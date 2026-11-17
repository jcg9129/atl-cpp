#pragma once
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief A node store the `splay_engine` can drive: nodes are `uint32`
	 *        ids into the user's storage, id 0 is the null sentinel.
	 *
	 * `child(x, d)` is x's child on side `d` (0 = left, 1 = right); the setters
	 * write one link and nothing else. The engine never writes a link of node 0
	 * and never follows one, so its storage may hold the identity aggregate a
	 * branchless `pull` folds through.
	 */
	template<typename Store>
	concept splay_store = requires(Store &s, Store const &cs, uint32 x, uint32 c, bool d) {
		{ cs.child(x, d) } -> std::convertible_to<uint32>;
		{ cs.parent(x) } -> std::convertible_to<uint32>;
		s.set_child(x, d, c);
		s.set_parent(x, c);
	};

	/**
	 * @brief A `splay_store` that also reports subtree sizes, which the
	 *        positional operations (`splay_at`, `split_at`, `insert_at`,
	 *        `max_right`) navigate by. Keeping them right is the store's `pull`;
	 *        `size(0)` must be 0.
	 */
	template<typename Store>
	concept sized_splay_store = splay_store<Store> && requires(Store const &cs, uint32 x) {
		{ cs.size(x) } -> std::convertible_to<uint32>;
	};

	/**
	 * @brief Where `splay_engine::descend` goes from the node a probe has just
	 *        seen. `stop` is not "equal": a lower-bound probe stops at a leaf
	 *        side it cannot enter, so a three-way order would be the wrong type.
	 */
	enum class splay_step : uint8 {
		left,  ///< Descend to the left child.
		right, ///< Descend to the right child.
		stop   ///< Accept the current node.
	};

	/**
	 * @brief A probe for `splay_engine::descend`: called on every visited
	 *        node (already pushed), root to leaf, through one shared lvalue, so
	 *        it may carry state (a remaining rank, an accumulated fold).
	 */
	template<typename Probe>
	concept splay_probe = std::invocable<Probe &, uint32> && std::same_as<std::invoke_result_t<Probe &, uint32>, splay_step>;

	/** @brief The result of `splay_engine::max_right`. */
	struct splay_prefix {
		uint32 root;   ///< the tree's new root (the last node the search visited)
		uint32 length; ///< the number of elements in the longest accepted prefix
	};

	namespace detail {
		/** @brief Store hook: settle x's pending tags one level down. */
		template<typename Store>
		concept splay_has_push = requires(Store &s, uint32 x) { s.push(x); };

		/** @brief Store hook: recompute x's size and aggregates from its children. */
		template<typename Store>
		concept splay_has_pull = requires(Store &s, uint32 x) { s.pull(x); };

		/** @brief Store hook: the splay stop test (link-cut path-parent pointers). */
		template<typename Store>
		concept splay_has_root_test = requires(Store &s, uint32 x) {
			{ s.is_splay_root(x) } -> std::convertible_to<bool>;
		};

		/** @brief Call the store's `push(x)` hook if it has one. */
		template<splay_store Store>
		constexpr void splay_push(Store &s, uint32 x) {
			if constexpr (splay_has_push<Store>) s.push(x);
		}

		/** @brief Call the store's `pull(x)` hook if it has one. */
		template<splay_store Store>
		constexpr void splay_pull(Store &s, uint32 x) {
			if constexpr (splay_has_pull<Store>) s.pull(x);
		}

		/** @brief The splay stop test: the store's `is_splay_root(x)` hook, else `parent(x) == 0`. */
		template<splay_store Store>
		constexpr bool splay_is_root(Store &s, uint32 x) {
			if constexpr (splay_has_root_test<Store>)
				return s.is_splay_root(x);
			else
				return s.parent(x) == 0;
		}

		/**
		 * @brief Rotate x over its parent p, pulling p only: x is pulled once,
		 *        when it stops rising. Both must be pushed.
		 */
		template<splay_store Store>
		constexpr void splay_rotate_up(Store &s, uint32 x) {
			uint32 const p = s.parent(x), g = s.parent(p);
			bool const side = s.child(p, 1) == x;
			uint32 const inner = s.child(x, !side);
			// tested before any relink: a link-cut root test reads p's parent's children
			if (!splay_is_root(s, p)) s.set_child(g, s.child(g, 1) == p, x);
			s.set_parent(x, g); // past a splay root this keeps the path-parent pointer
			s.set_child(p, side, inner);
			if (inner != 0) s.set_parent(inner, p);
			s.set_child(x, !side, p);
			s.set_parent(p, x);
			splay_pull(s, p);
		}

		/** @brief Bottom-up splay of x, whose whole ancestor path is already pushed. */
		template<splay_store Store>
		constexpr void splay_pushed(Store &s, uint32 x) {
			while (!splay_is_root(s, x)) {
				uint32 const p = s.parent(x);
				if (!splay_is_root(s, p)) {
					uint32 const g = s.parent(p);
					bool const is_zig_zig = (s.child(g, 1) == p) == (s.child(p, 1) == x);
					splay_rotate_up(s, is_zig_zig ? p : x);
				}
				splay_rotate_up(s, x);
			}
			splay_pull(s, x);
		}

		/**
		 * @brief The current thread's ancestor buffer for handle splays:
		 *        reused, so a splay allocates nothing once it has grown.
		 */
		inline std::vector<uint32> &splay_ancestor_buffer() {
			static thread_local std::vector<uint32> buffer;
			return buffer;
		}

		/** @brief Push x's ancestors (and x) top-down, recording the path in @p path. */
		template<splay_store Store>
			requires splay_has_push<Store>
		constexpr void splay_push_path_with(Store &s, uint32 x, std::vector<uint32> &path) {
			path.clear();
			for (uint32 y = x;; y = s.parent(y)) {
				path.push_back(y);
				if (splay_is_root(s, y)) break;
			}
			for (std::size_t i = path.size(); i-- > 0;) s.push(path[i]);
		}

		/** @brief Push the path from x's splay root down to x, top-down. */
		template<splay_store Store>
		constexpr void splay_push_path(Store &s, uint32 x) {
			if constexpr (splay_has_push<Store>) {
				if (splay_is_root(s, x)) {
					s.push(x);
					return;
				}
				if consteval {
					std::vector<uint32> path;
					splay_push_path_with(s, x, path);
				} else {
					splay_push_path_with(s, x, splay_ancestor_buffer());
				}
			}
		}

		/**
		 * @brief The operations every splay engine builds from its own
		 *        `descend` and `splay`, written once (CRTP): the library
		 *        `atl::splay_engine` and the experimental contenders share them.
		 */
		template<typename Engine>
			requires std::is_class_v<Engine> // incomplete here (CRTP): its members are checked at each use
		struct splay_composites {
		private:
			splay_composites() = default;
			friend Engine;

		public:
			/**
			 * @brief Splay the node of rank @p k (0-based, in order) to the root.
			 * @pre `k < size(root)` (asserted) @complexity amortized O(log n)
			 */
			template<sized_splay_store Store>
			static constexpr uint32 splay_at(Store &s, uint32 root, uint32 k) {
				atl_assert(root != 0 && k < s.size(root));
				return Engine::descend(s, root, [&s, k](uint32 x) mutable {
					uint32 const left_size = s.size(s.child(x, 0));
					if (k < left_size) return splay_step::left;
					if (k == left_size) return splay_step::stop;
					k -= left_size + 1;
					return splay_step::right;
				});
			}

			/**
			 * @brief Join two trees, every element of @p a before every element
			 *        of @p b; either may be 0. Splays a's last node to the root
			 *        and hangs @p b to its right.
			 * @pre `a` and `b` are roots of different trees
			 * @complexity amortized O(log n)
			 */
			template<splay_store Store>
			static constexpr uint32 join(Store &s, uint32 a, uint32 b) {
				if (a == 0) return b;
				if (b == 0) return a;
				uint32 const last = Engine::descend(s, a, [](uint32) { return splay_step::right; });
				s.set_child(last, 1, b);
				s.set_parent(b, last);
				splay_pull(s, last);
				return last;
			}

			/**
			 * @brief Split into the first @p k elements and the rest (either
			 *        part may be 0). The rest's root keeps the old root's parent.
			 * @pre `k <= size(root)` (asserted) @complexity amortized O(log n)
			 */
			template<sized_splay_store Store>
			static constexpr std::pair<uint32, uint32> split_at(Store &s, uint32 root, uint32 k) {
				atl_assert(k <= s.size(root));
				if (k == 0) return {0, root};
				if (k == s.size(root)) return {root, 0};
				uint32 const first = Engine::splay_at(s, root, k);
				return {detach(s, first, 0), first};
			}

			/**
			 * @brief Split around handle @p x: (everything before x, x and
			 *        everything after). @pre `x != 0` (asserted)
			 * @complexity amortized O(log n)
			 */
			template<splay_store Store>
			static constexpr std::pair<uint32, uint32> split_before(Store &s, uint32 x) {
				Engine::splay(s, x);
				return {detach(s, x, 0), x};
			}

			/**
			 * @brief Split around handle @p x: (everything up to x, everything
			 *        after x). @pre `x != 0` (asserted)
			 * @complexity amortized O(log n)
			 */
			template<splay_store Store>
			static constexpr std::pair<uint32, uint32> split_after(Store &s, uint32 x) {
				Engine::splay(s, x);
				uint32 const after = detach(s, x, 1);
				return {x, after};
			}

			/**
			 * @brief Link the fresh node @p x at rank @p k (it ends up with k
			 *        elements before it) and return it, the new root.
			 * @pre `x` is detached, both children and parent 0; `k <= size(root)`
			 *      (asserted)
			 * @complexity amortized O(log n)
			 */
			template<sized_splay_store Store>
			static constexpr uint32 insert_at(Store &s, uint32 root, uint32 k, uint32 x) {
				atl_assert(x != 0 && s.child(x, 0) == 0 && s.child(x, 1) == 0 && s.parent(x) == 0);
				auto const [before, after] = split_at(s, root, k);
				s.set_child(x, 0, before);
				if (before != 0) s.set_parent(before, x);
				s.set_child(x, 1, after);
				if (after != 0) s.set_parent(after, x);
				splay_pull(s, x);
				return x;
			}

			/**
			 * @brief Unlink handle @p x and return the root of what remains (its
			 *        former parent pointer is carried over). x is left detached and
			 *        pulled, ready to reuse; its storage stays the caller's.
			 * @pre `x != 0` (asserted) @complexity amortized O(log n)
			 */
			template<splay_store Store>
			static constexpr uint32 erase(Store &s, uint32 x) {
				Engine::splay(s, x);
				uint32 const above = s.parent(x);
				uint32 const before = detach(s, x, 0), after = detach(s, x, 1);
				s.set_parent(x, 0);
				uint32 const rest = join(s, before, after);
				if (rest != 0) s.set_parent(rest, above);
				splay_pull(s, x);
				return rest;
			}

			/**
			 * @brief The longest prefix whose fold satisfies @p pred: starting
			 *        from @p acc (the identity, which @p pred must accept), each
			 *        step folds `op(acc, aggregate(left child))` and then
			 *        `op(acc, value(x))`. The last visited node becomes the root.
			 *
			 * `aggregate(x)` is the store's subtree fold and `value(x)` the node's
			 * own element, both in the fold's type. @p pred must be monotone along
			 * the sequence (true on a prefix of the prefixes).
			 * @complexity amortized O(log n) folds and predicate calls
			 */
			template<sized_splay_store Store, typename Value, typename Op, typename Pred>
				requires requires(Store &s, uint32 x) {
					{ s.aggregate(x) } -> std::convertible_to<Value const &>;
					{ s.value(x) } -> std::convertible_to<Value const &>;
				} && std::regular_invocable<Op &, Value const &, Value const &>
					  && (std::same_as<std::invoke_result_t<Op &, Value const &, Value const &>, Value>
						  || std::convertible_to<std::invoke_result_t<Op &, Value const &, Value const &>, Value>)
					  && std::predicate<Pred &, Value const &>
					  && requires(Value &acc, Value &candidate) { acc = std::move(candidate); }
			[[nodiscard]] static constexpr splay_prefix max_right(Store &s, uint32 root, Value acc, Op &&op, Pred &&pred) {
				if (root == 0) return {0, 0};
				uint32 length = 0;
				uint32 const top = Engine::descend(s, root, [&](uint32 x) {
					if (uint32 const left = s.child(x, 0); left != 0) {
						Value candidate = std::invoke(op, std::as_const(acc), static_cast<Value const &>(s.aggregate(left)));
						if (!std::invoke(pred, std::as_const(candidate))) return splay_step::left;
						acc = std::move(candidate);
						length += s.size(left);
					}
					Value candidate = std::invoke(op, std::as_const(acc), static_cast<Value const &>(s.value(x)));
					if (!std::invoke(pred, std::as_const(candidate))) return splay_step::stop;
					acc = std::move(candidate);
					++length;
					return splay_step::right;
				});
				return {top, length};
			}

			/**
			 * @brief Link the nodes @p ids, in order, into a perfectly balanced
			 *        tree and return its root (0 when empty): an explicit range
			 *        stack, then pulls in reverse creation order (children first).
			 *        Every link of every id is overwritten, so recycled nodes need
			 *        no reset. Used by the splay
			 *        sequence and map for sorted construction and copies,
			 *        @ref atl::top_tree for rake trees, and the splay-based
			 *        Euler-tour contender for @ref atl::euler_tour_tree.
			 * @pre the ids are distinct nodes in no other tree; fewer than 2^32
			 *      ids (asserted)
			 * @complexity O(n)
			 */
			template<splay_store Store, typename R>
				requires std::ranges::random_access_range<R const> && std::ranges::sized_range<R const>
					  && requires(std::ranges::range_reference_t<R const> id) { static_cast<uint32>(id); }
			static constexpr uint32 build_balanced(Store &s, R const &ids) {
				std::size_t const count = std::ranges::size(ids);
				if (count == 0) return 0;
				atl_assert(count < (std::numeric_limits<uint32>::max)());

				/** @brief A pending half-open id range [lo, hi) and where its middle node hangs. */
				struct range {
					uint32 lo, hi, parent;
					bool side;
				};

				std::vector<range> pending{{0, uint32(count), 0, false}};
				std::vector<uint32> order;
				order.reserve(count);
				auto const first = std::ranges::begin(ids);
				uint32 root = 0;
				while (!pending.empty()) {
					auto const [lo, hi, above, side] = pending.back();
					pending.pop_back();
					uint32 const mid = lo + (hi - lo) / 2, x = uint32(first[mid]);
					s.set_parent(x, above);
					s.set_child(x, 0, 0);
					s.set_child(x, 1, 0);
					if (above != 0)
						s.set_child(above, side, x);
					else
						root = x;
					order.push_back(x);
					if (lo < mid) pending.push_back({lo, mid, x, false});
					if (mid + 1 < hi) pending.push_back({mid + 1, hi, x, true});
				}
				for (std::size_t i = order.size(); i-- > 0;) splay_pull(s, order[i]);
				return root;
			}

		protected:
			/** @brief Cut x's child on @p side loose (x must be pushed) and return it. */
			template<splay_store Store>
			static constexpr uint32 detach(Store &s, uint32 x, bool side) {
				uint32 const c = s.child(x, side);
				if (c == 0) return 0;
				s.set_child(x, side, 0);
				s.set_parent(c, 0);
				splay_pull(s, x);
				return c;
			}
		};

		/** @brief The links record of `atl::splay_links` / `atl::sized_splay_links`. */
		struct splay_link {
			uint32 child[2]{}; ///< Left and right child ids.
			uint32 parent = 0; ///< Parent id, zero for a standalone root.
		};

		/** @brief `splay_link` plus the subtree size. */
		struct sized_splay_link : splay_link {
			uint32 size = 0; ///< Subtree node count, zero at the sentinel.
		};

		/**
		 * @brief The links half of a splay store, payload left to the user:
		 *        one record per node (`child[2]`, `parent`, and the size when
		 *        @p IsSized), id 0 preallocated as the null sentinel.
		 */
		template<bool IsSized>
		struct splay_links_base {
			/** @brief Link record type exposed by this mixin. */
			using link = std::conditional_t<IsSized, sized_splay_link, splay_link>;

			/** @brief Node link records including null sentinel zero; preserve id and link invariants when editing. */
			std::vector<link> links = std::vector<link>(1);

			/** @brief Read one child id. @pre `x < links.size()`. @complexity O(1) */
			[[nodiscard]] constexpr uint32 child(uint32 x, bool side) const { return links[x].child[side]; }

			/** @brief Read the parent id. @pre `x < links.size()`. @complexity O(1) */
			[[nodiscard]] constexpr uint32 parent(uint32 x) const { return links[x].parent; }

			/** @brief Write one child id without updating other links or sizes. @pre Valid ids. @complexity O(1) */
			constexpr void set_child(uint32 x, bool side, uint32 c) { links[x].child[side] = c; }

			/** @brief Write one parent id without updating other links or sizes. @pre Valid ids. @complexity O(1) */
			constexpr void set_parent(uint32 x, uint32 p) { links[x].parent = p; }

			/** @brief Read a subtree size. @pre `x < links.size()`. @complexity O(1) */
			[[nodiscard]] constexpr uint32 size(uint32 x) const
				requires IsSized
			{
				return links[x].size;
			}

			/** @brief Recompute x's size from its children: call it from `pull`. */
			constexpr void pull_size(uint32 x)
				requires IsSized
			{
				link &l = links[x];
				l.size = links[l.child[0]].size + 1 + links[l.child[1]].size;
			}

			/** @brief Append a detached node (size 1 when sized) and return its id. */
			constexpr uint32 add_node() {
				atl_assert(links.size() < (std::numeric_limits<uint32>::max)()); // ids are 32-bit
				link &l = links.emplace_back();
				if constexpr (IsSized) l.size = 1;
				return uint32(links.size() - 1);
			}

			/** @brief The number of nodes, the null sentinel excluded. */
			[[nodiscard]] constexpr uint32 node_count() const { return uint32(links.size() - 1); }

			/** @brief Room for @p nodes nodes besides the sentinel. */
			constexpr void reserve(std::size_t nodes) {
				atl_assert(nodes < (std::numeric_limits<uint32>::max)() && nodes < (std::numeric_limits<std::size_t>::max)());
				links.reserve(nodes + 1);
			}
		};
	}

	/**
	 * @brief Reusable links for a `splay_store:` derive from it and add the
	 *        payload, `pull` and `push`. Links live in their own array, apart
	 *        from the payload (`links[x]`: `child[2]`, `parent`).
	 */
	using splay_links = detail::splay_links_base<false>;

	/**
	 * @brief `splay_links` with subtree sizes, a `sized_splay_store` once
	 *        the derived `pull` calls `pull_size(x)`.
	 */
	using sized_splay_links = detail::splay_links_base<true>;

	/**
	 * @brief Recursion-free splay-tree algorithms over any `splay_store:`
	 *        the structure half of a splay tree, the user's store being the
	 *        payload half (sequences, ordered maps, link-cut and Euler tour
	 *        trees, custom applications).
	 *
	 * Roots are passed and returned as ids, 0 being the empty tree. Optional
	 * store hooks, detected and otherwise no-ops:
	 *  - `push(x)`: settle x's pending tags one level; called on every node
	 *    before its children are read or relinked;
	 *  - `pull(x)`: recompute x's size and aggregates from its children; called
	 *    bottom-up after every relink;
	 *  - `is_splay_root(x)`: the splay stop test, default `parent(x) == 0`; a
	 *    link-cut store answers true also when x's parent does not list x as a
	 *    child (a path-parent pointer), and a splay then keeps that pointer.
	 *
	 * Every `splay` costs amortized O(log n) on a tree of n nodes (the access
	 * lemma) and every operation is O(1) splays plus the descent it splays.
	 * A handle splay pushes the path above the handle first, through a reused
	 * thread-local buffer (a local one in constant evaluation), so the hooks
	 * must not themselves splay.
	 */
	struct splay_engine : detail::splay_composites<splay_engine> {
		/**
		 * @brief The current thread's ancestor buffer, the one a handle
		 *        `splay` records its push path in: shared with multi-level
		 *        clients (@ref atl::top_tree pushes its compress and rake chain), so they allocate nothing once it has
		 *        grown. The contents are scratch: any engine call may overwrite
		 *        them, so a client fills and drains it between engine calls.
		 *        @note Not constexpr (uses a thread-local buffer). In constant evaluation a
		 *        client takes a local vector instead.
		 */
		[[nodiscard]] static std::vector<uint32> &ancestor_buffer() noexcept { return detail::splay_ancestor_buffer(); }

		/**
		 * @brief Splay @p x to the root of its splay tree when the caller has
		 *        already pushed every node from that root down to x: the second
		 *        half of `splay`, without its push pass. For clients that
		 *        settle the chain themselves, as part of a walk they make anyway:
		 *        after a descent that pushed on the way down, or after one
		 *        top-down push across several trees through
		 *        `ancestor_buffer` (@ref atl::top_tree pushes its compress and rake
		 *        chain once, then splays each tree with this).
		 * @pre `x != 0`; no node on x's splay path to its root carries a
		 *      pending tag (not checkable: a stale tag is silently relinked
		 *      under the wrong children)
		 * @complexity amortized O(log n)
		 */
		template<splay_store Store>
		static constexpr void splay_pushed(Store &s, uint32 x) {
			atl_assert(x != 0);
			detail::splay_pushed(s, x);
		}

		/**
		 * @brief One rotation of @p x over its parent, both pushed here; both
		 *        pulled. @pre `x != 0`, x is not a splay root, and the nodes
		 *        above x's parent carry no pending tags (all asserted but the last)
		 */
		template<splay_store Store>
		static constexpr void rotate(Store &s, uint32 x) {
			atl_assert(x != 0 && !detail::splay_is_root(s, x));
			detail::splay_push(s, s.parent(x));
			detail::splay_push(s, x);
			detail::splay_rotate_up(s, x);
			detail::splay_pull(s, x);
		}

		/**
		 * @brief Make @p x the root of its splay tree. Any node id is valid:
		 *        the path above x is pushed first.
		 * @pre `x != 0` (asserted) @complexity amortized O(log n)
		 */
		template<splay_store Store>
		static constexpr void splay(Store &s, uint32 x) {
			atl_assert(x != 0);
			detail::splay_push_path(s, x);
			detail::splay_pushed(s, x);
		}

		/**
		 * @brief Walk down from @p root, asking @p probe at each node (already
		 *        pushed) which way to go, splaying the path on the way (top-down),
		 *        and return the last node visited, the new root. A step towards
		 *        an empty side stops.
		 *
		 * Simplified top-down splay: nodes passed on the left are hung on a left
		 * tree by their right links, those on the right on a right tree by their
		 * left links (a zig-zig rotates first; a zig-zag is two zigs); the last
		 * node adopts both trees, whose spines are then pulled bottom-up by
		 * parent pointers. The root's parent pointer (a link-cut path-parent)
		 * passes to the new root. One pass down plus the spine pulls, where a
		 * bottom-up splay walks the path twice. Descent and layout tradeoffs are
		 * measured in docs/splay-engine.md.
		 *
		 * @p root may also be a proper subtree of a splay tree whose ancestors
		 * are pushed (a link-cut predecessor search below an accessed node):
		 * then the reached node is splayed bottom-up to the top of the whole
		 * splay tree, and is that tree's new root.
		 *
		 * The probe is shared by reference and sees every visited node exactly
		 * once, root to leaf: a stateful probe (a remaining rank, a lower-bound
		 * candidate) is the intended use.
		 * @complexity amortized O(log n) plus the probe calls, one per level
		 */
		template<splay_store Store, splay_probe Probe>
		static constexpr uint32 descend(Store &s, uint32 root, Probe &&probe) {
			if (root == 0) return 0;
			if (!detail::splay_is_root(s, root)) {
				// a subtree inside a larger splay tree: walk, then splay bottom-up to the top
				uint32 x = root;
				for (;;) {
					detail::splay_push(s, x);
					splay_step const step = std::invoke(probe, x);
					if (step == splay_step::stop) break;
					uint32 const next = s.child(x, step == splay_step::right);
					if (next == 0) break;
					x = next;
				}
				detail::splay_pushed(s, x);
				return x;
			}
			uint32 const above = s.parent(root);
			// tree[0] collects nodes before the target (linked by right
			// children), tree[1] nodes after it (linked by left children)
			uint32 tree_root[2]{}, tree_tail[2]{};
			auto const hang = [&](uint32 x, bool after) {
				if (tree_root[after] == 0) {
					tree_root[after] = x;
				} else {
					s.set_child(tree_tail[after], !after, x);
					s.set_parent(x, tree_tail[after]);
				}
				tree_tail[after] = x;
			};
			uint32 t = root;
			detail::splay_push(s, t);
			splay_step step = std::invoke(probe, t);
			while (step != splay_step::stop) {
				bool const side = step == splay_step::right;
				uint32 const c = s.child(t, side);
				if (c == 0) break;
				detail::splay_push(s, c);
				splay_step const next = std::invoke(probe, c);
				if (next == step) {
					if (uint32 const grandchild = s.child(c, side); grandchild != 0) {
						// zig-zig: rotate c over t, hang c, continue below it
						uint32 const inner = s.child(c, !side);
						s.set_child(t, side, inner);
						if (inner != 0) s.set_parent(inner, t);
						s.set_child(c, !side, t);
						s.set_parent(t, c);
						detail::splay_pull(s, t);
						hang(c, !side);
						t = grandchild;
						detail::splay_push(s, t);
						step = std::invoke(probe, t);
						continue;
					}
				}
				hang(t, !side);
				t = c;
				step = next;
			}
			for (bool const after : {false, true}) {
				if (tree_root[after] == 0) continue;
				uint32 const inner = s.child(t, after);
				s.set_child(tree_tail[after], !after, inner);
				if (inner != 0) s.set_parent(inner, tree_tail[after]);
				s.set_child(t, after, tree_root[after]);
				s.set_parent(tree_root[after], t);
				for (uint32 y = tree_tail[after];; y = s.parent(y)) {
					detail::splay_pull(s, y);
					if (y == tree_root[after]) break;
				}
			}
			s.set_parent(t, above);
			detail::splay_pull(s, t);
			return t;
		}

		/**
		 * @brief In-order visit of every node id in @p root's tree, pushing on
		 *        the way down; stackless (it climbs by parent pointers).
		 *        @p visit must not relink nodes.
		 * @complexity O(n)
		 */
		template<splay_store Store, typename Visit>
			requires std::invocable<Visit &, uint32>
		static constexpr void for_each(Store &s, uint32 root, Visit &&visit) {
			if (root == 0) return;
			auto const leftmost = [&s](uint32 x) {
				detail::splay_push(s, x);
				for (uint32 c; (c = s.child(x, 0)) != 0; x = c) detail::splay_push(s, c);
				return x;
			};
			uint32 x = leftmost(root);
			for (;;) {
				std::invoke(visit, x);
				if (uint32 const right = s.child(x, 1); right != 0) {
					x = leftmost(right);
					continue;
				}
				for (;;) { // climb to the first ancestor entered from its left
					if (x == root) return;
					uint32 const p = s.parent(x);
					bool const is_left = s.child(p, 0) == x;
					x = p;
					if (is_left) break;
				}
			}
		}
	};
}
