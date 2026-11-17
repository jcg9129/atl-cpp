#pragma once
#include <atl/ds/detail/algebra.hpp>
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/treap_base.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Implicit treap: a sequence with expected O(log n) surgery anywhere.
	 *
	 * A randomized balanced tree ordered by *position* (no keys): insert or erase
	 * at any index, split off a suffix, concatenate — all expected O(log n) —
	 * with the segment-tree algebra available on top. Capability is opt-in
	 * through the template arguments, in the `lazy_segment_tree` vocabulary:
	 *
	 *  - `treap_sequence<T>` — the plain rope: surgery only;
	 *  - `treap_sequence<T, Op>` — adds `fold(l, r)` range folds and
	 *    `max_right`, with `(T, op, s_identity)` a monoid;
	 *  - `treap_sequence<T, Op, F, Mapping, Compose>` — adds `apply(l, r, f)`
	 *    lazy tags, under the same laws as the lazy tree (mapping distributes
	 *    over op and the identity tag acts as identity; carry widths inside `T` when a tag's effect depends on them).
	 *
	 * Sequences derived from one origin (via `split` / copies) share a node
	 * pool; each handle owns its nodes (destruction recycles them). `concat`
	 * requires the shared pool. Copying is a deep O(n) clone — use `split` /
	 * moves for the cheap paths. A moved-from handle is empty and only good for
	 * destruction or assignment. Priorities are seeded per process (see `atl::set_hash_seed`), expected bounds assume inputs independent of the seed; worst-case depth is linear.
	 *
	 * @note References returned by `at` point into the shared pool's node
	 *       storage: any call that allocates a node (`insert`, `push_back`,
	 *       `push_front`, copying, building) on this handle **or any handle
	 *       sharing its pool** may reallocate it and dangle them. Copy the
	 *       value out before mutating.
	 *
	 * @note Constructor algebra arguments are non-deduced; const-qualified callable
	 *       types accept ordinary callable arguments. Defaulted template parameters
	 *       keep disabled tiers from declaring void parameters.
	 *
	 * @tparam T       The element type.
	 * @tparam Op      Optional element fold: any `op(T, T)` convertible to `T`.
	 * @tparam F       Optional tag type applied to ranges (requires `Op`).
	 * @tparam Mapping Tag application `mapping(F, T)` (with `F`).
	 * @tparam Compose Tag composition `compose(F, F)`, newer first (with `F`).
	 * @tparam IsReversible Whether reversal and reverse aggregates are enabled.
	 */
	template<std::semiregular T, typename Op = void, typename F = void, typename Mapping = void,
			 typename Compose = void, bool IsReversible = false>
		requires((std::is_void_v<Op> || binary_operation<Op, T>)
				 && (std::is_void_v<F> == std::is_void_v<Mapping>)
				 && (std::is_void_v<F> == std::is_void_v<Compose>)
				 && (std::is_void_v<F> || (!std::is_void_v<Op> && std::semiregular<F>))
				 && (std::is_void_v<Mapping> || action<Mapping, F, T>)
				 && (std::is_void_v<Compose> || binary_operation<Compose, F>))
	class treap_sequence {
		static constexpr bool has_op = !std::is_void_v<Op>;
		static constexpr bool has_tags = !std::is_void_v<F>;

		struct no_slot {};

		struct no_agg_slot {};

		struct no_tag_slot {};

		struct no_rev_slot {};

		struct no_agg_rev_slot {};

		struct aggregate_slot {
			T aggregate;
		};

		struct tag_slot {
			F tag;
		};

		struct reversed_slot {
			bool is_reversed = false;
		};

		struct reversed_aggregate_slot {
			T reversed_aggregate; ///< fold in reversed order, for correct folds under pending flips
		};

		struct node : std::conditional_t<has_op, aggregate_slot, no_agg_slot>,
					  std::conditional_t<has_tags, tag_slot, no_tag_slot>,
					  std::conditional_t<IsReversible, reversed_slot, no_rev_slot>,
					  std::conditional_t<IsReversible && has_op, reversed_aggregate_slot, no_agg_rev_slot> {
			T value;
			uint32 left = 0, right = 0, size = 1, priority = 0;
		};

		template<typename X>
		using slot = std::conditional_t<std::is_void_v<X>, no_slot, X>;

		/**
		 * @brief The shared node pool: storage, recycling, algebra, and the
		 * `detail::treap_store` interface the engine drives.
		 */
		struct pool {
			std::vector<node> nodes;
			std::vector<uint32> recycled;
			uint64 rng;
			[[no_unique_address]] slot<Op> op;
			[[no_unique_address]] slot<Mapping> mapping;
			[[no_unique_address]] slot<Compose> compose;
			[[no_unique_address]] std::conditional_t<has_op, T, no_slot> element_identity;
			[[no_unique_address]] slot<F> tag_identity;

			pool(slot<Op> fold = {}, slot<Mapping> map = {}, slot<Compose> composition = {},
				 std::conditional_t<has_op, T, no_slot> identity = {}, slot<F> tag_unit = {})
				: rng(detail::hash_seed() ^ 0x9e3779b97f4a7c15ULL), op(detail::move_or_copy(fold)), mapping(detail::move_or_copy(map)), compose(detail::move_or_copy(composition)),
				  element_identity(std::move(identity)), tag_identity(std::move(tag_unit)) {
				nodes.emplace_back(); // id 0 = null: size 0, never read otherwise
				nodes[0].size = 0;
				if constexpr (has_op) nodes[0].aggregate = element_identity;
				if constexpr (IsReversible && has_op) nodes[0].reversed_aggregate = element_identity;
			}

			decltype(auto) fold_values(auto &&a, auto &&b) const
				requires has_op && requires { detail::invoke_algebra<T>(op, std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)); }
			{ return detail::invoke_algebra<T>(op, std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)); }

			decltype(auto) map_value(auto &&tag_value, auto &&value) const
				requires has_tags && requires { detail::invoke_algebra<T>(mapping, std::forward<decltype(tag_value)>(tag_value), std::forward<decltype(value)>(value)); }
			{ return detail::invoke_algebra<T>(mapping, std::forward<decltype(tag_value)>(tag_value), std::forward<decltype(value)>(value)); }

			decltype(auto) compose_tags(auto &&a, auto &&b) const
				requires has_tags && requires { detail::invoke_algebra<slot<F>>(compose, std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)); }
			{ return detail::invoke_algebra<slot<F>>(compose, std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)); }

			uint32 left(uint32 t) const { return nodes[t].left; }

			uint32 right(uint32 t) const { return nodes[t].right; }

			uint32 size(uint32 t) const { return nodes[t].size; }

			uint32 priority(uint32 t) const { return nodes[t].priority; }

			// Branchless in-order aggregate: the null sentinel's aggregate holds
			// s_identity (established at construction, never rebound), so folding
			// straight through null sides is lawful without per-child conditions.
			void pull(uint32 t) {
				node &nd = nodes[t];
				nd.size = 1 + nodes[nd.left].size + nodes[nd.right].size;
				if constexpr (has_op) nd.aggregate = fold_values(fold_values(nodes[nd.left].aggregate, nd.value), nodes[nd.right].aggregate);
				if constexpr (IsReversible && has_op)
					nd.reversed_aggregate = fold_values(fold_values(nodes[nd.right].reversed_aggregate, nd.value), nodes[nd.left].reversed_aggregate);
			}

			uint32 rebind(uint32 t, uint32 l, uint32 r) {
				nodes[t].left = l;
				nodes[t].right = r;
				pull(t);
				return t;
			}

			uint32 rebind_left(uint32 t, uint32 l) {
				nodes[t].left = l;
				pull(t);
				return t;
			}

			uint32 rebind_right(uint32 t, uint32 r) {
				nodes[t].right = r;
				pull(t);
				return t;
			}

			/**
			 * @brief Fold @p f into node @p t: value and aggregate updated now, tag
			 * composed for the subtree below (same convention as the lazy tree).
			 */
			template<typename FArg = F>
				requires(has_tags && std::same_as<FArg, F>)
			void apply_tag(uint32 t, FArg const &f) {
				node &nd = nodes[t];
				nd.value = map_value(f, nd.value);
				if constexpr (has_op) nd.aggregate = map_value(f, nd.aggregate);
				if constexpr (IsReversible && has_op) nd.reversed_aggregate = map_value(f, nd.reversed_aggregate);
				nd.tag = compose_tags(f, nd.tag);
			}

			/**
			 * @brief Mark subtree @p t as reversed: its own aggregates flip now, the
			 * structural swap stays pending (settled by `push`).
			 */
			void flip(uint32 t)
				requires(IsReversible)
			{
				if (t == 0) return;
				node &nd = nodes[t];
				nd.is_reversed = !nd.is_reversed;
				if constexpr (has_op) std::swap(nd.aggregate, nd.reversed_aggregate);
			}

			uint32 push(uint32 t) {
				if constexpr (IsReversible) {
					if (node &nd = nodes[t]; nd.is_reversed) {
						std::swap(nd.left, nd.right);
						flip(nd.left);
						flip(nd.right);
						nd.is_reversed = false;
					}
				}
				if constexpr (has_tags) {
					node &nd = nodes[t];
					// An identity tag has nothing to settle — skip the child
					// mapping/compose calls when F can be compared.
					if constexpr (std::equality_comparable<F>)
						if (nd.tag == tag_identity) return t;
					if (nd.left != 0) apply_tag(nd.left, nd.tag);
					if (nd.right != 0) apply_tag(nd.right, nd.tag);
					nd.tag = tag_identity;
				}
				return t;
			}

			/**
			 * @brief A fresh detached node holding `T(args...)`: one construction
			 *        plus one move into the slot.
			 */
			template<typename... Args>
				requires std::constructible_from<T, Args...>
			uint32 make(Args &&...args) {
				// built before the node array may grow: an argument can alias an
				// element of this very pool (clone, `emplace_back(s.at(0))`)
				T value(std::forward<Args>(args)...);
				uint32 id;
				if (!recycled.empty()) {
					id = recycled.back();
					recycled.pop_back();
				} else {
					// ids are 32-bit; id 0 is the null sentinel
					atl_assert(nodes.size() < (std::numeric_limits<uint32>::max)());
					nodes.emplace_back();
					id = uint32(nodes.size() - 1);
				}
				node &nd = nodes[id];
				nd.value = std::move(value);
				nd.left = nd.right = 0;
				nd.size = 1;
				nd.priority = uint32(detail::splitmix64(rng += 0x9e3779b97f4a7c15ULL) >> 32);
				if constexpr (has_op) nd.aggregate = nd.value;
				if constexpr (IsReversible && has_op) nd.reversed_aggregate = nd.value;
				if constexpr (IsReversible) nd.is_reversed = false;
				if constexpr (has_tags) nd.tag = tag_identity;
				return id;
			}

			void recycle(uint32 t) {
				if (t == 0) return;
				recycle(nodes[t].left);
				recycle(nodes[t].right);
				// a heap-owning element is released now, not when the id is reused
				if constexpr (!std::is_trivially_destructible_v<T>) {
					node &nd = nodes[t];
					nd.value = T{};
					if constexpr (has_op) nd.aggregate = T{};
					if constexpr (IsReversible && has_op) nd.reversed_aggregate = T{};
				}
				recycled.push_back(t);
			}

			/**
			 * @brief Deep structural copy of a subtree (priorities kept: the copy is an
			 * independent tree, so sharing them is harmless).
			 */
			uint32 clone(uint32 t) {
				if (t == 0) return 0;
				uint32 const l = clone(nodes[t].left), r = clone(nodes[t].right);
				uint32 const id = make(nodes[t].value);
				node &nd = nodes[id];
				node const &src = nodes[t];
				nd.priority = src.priority;
				if constexpr (has_tags) nd.tag = src.tag;
				if constexpr (IsReversible) nd.is_reversed = src.is_reversed;
				nd.left = l;
				nd.right = r;
				pull(id);
				if constexpr (has_op) nd.aggregate = src.aggregate; // pull is tag-blind; the source aggregate is authoritative
				if constexpr (IsReversible && has_op) nd.reversed_aggregate = src.reversed_aggregate;
				return id;
			}
		};

		using base = detail::treap_base;

		std::shared_ptr<pool> pool_;
		uint32 root_ = 0;

		treap_sequence(std::shared_ptr<pool> shared, uint32 root) : pool_(std::move(shared)), root_(root) {}

		std::pair<uint32, uint32> split_prefix(uint32 t, std::size_t k) {
			return base::split_at(*pool_, t, uint32(k));
		}

		// (prefix [0, l), middle [l, r), suffix [r, n))
		struct parts {
			uint32 prefix, middle, suffix;
		};

		parts split_range(std::size_t l, std::size_t r) {
			auto [ab, c] = split_prefix(root_, r);
			auto [a, b] = split_prefix(ab, l);
			return {a, b, c};
		}

		void rejoin(uint32 a, uint32 b, uint32 c) {
			root_ = base::merge(*pool_, base::merge(*pool_, a, b), c);
		}

		// Fold of the non-empty [l, r) inside subtree t (subtree-local indices),
		// accumulating pending path tags instead of settling them, and — on the
		// reversible tier — the parity `is_flipped` of ancestors' pending flips: odd
		// parity mirrors child roles and reads reversed aggregates. `acc`
		// excludes t's own tag: a node's value/aggregate already include it.
		template<typename... Acc>
			requires(sizeof...(Acc) == (has_tags ? 1 : 0) && (std::convertible_to<Acc const &, slot<F>> && ...))
		T fold_range(uint32 t, uint32 l, uint32 r, bool is_flipped, Acc const &...acc) const {
			node const &nd = pool_->nodes[t];
			if (l == 0 && r == nd.size) {
				T const &whole = [&]() -> T const & {
					if constexpr (IsReversible && has_op)
						return is_flipped ? nd.reversed_aggregate : nd.aggregate;
					else
						return nd.aggregate;
				}();
				if constexpr (has_tags)
					return pool_->map_value(acc..., whole);
				else
					return whole;
			}
			bool is_child_flipped = is_flipped;
			if constexpr (IsReversible) is_child_flipped = is_flipped != nd.is_reversed;
			uint32 const lc = is_child_flipped ? nd.right : nd.left, rc = is_child_flipped ? nd.left : nd.right;
			uint32 const ls = pool_->nodes[lc].size;
			auto const fold = [&](uint32 child, uint32 cl, uint32 cr) {
				if constexpr (has_tags)
					return fold_range(child, cl, cr, is_child_flipped, pool_->compose_tags(acc..., nd.tag));
				else
					return fold_range(child, cl, cr, is_child_flipped);
			};
			if (r <= ls) return fold(lc, l, r);
			if (l > ls) return fold(rc, l - ls - 1, r - ls - 1);
			auto value_of = [&] {
				if constexpr (has_tags)
					return pool_->map_value(acc..., nd.value);
				else
					return nd.value;
			};
			if (l == ls) { // value, maybe right part
				T v = value_of();
				return r > ls + 1 ? pool_->fold_values(std::move(v), fold(rc, 0, r - ls - 1)) : v;
			}
			T left_part = fold(lc, l, ls);
			T v = pool_->fold_values(std::move(left_part), value_of());
			return r > ls + 1 ? pool_->fold_values(std::move(v), fold(rc, 0, r - ls - 1)) : v;
		}

		/**
		 * @brief O(n) bulk build: maintain the right spine as a stack; each new
		 * node evicts the lower-priority tail of the spine as its left subtree.
		 * Evicted nodes and, finally, the remaining spine are pulled
		 * children-first, so every augmentation is computed exactly once.
		 * @tparam Source the constructor's range type: elements are moved or
		 * copied out of `[it, last)` by the library rule in
		 * `detail::forward_element`.
		 * @complexity O(n)
		 */
		template<std::ranges::range Source, std::input_iterator It, std::sentinel_for<It> Se>
			requires requires(It it) { static_cast<T>(detail::forward_element<Source>(*it)); }
		uint32 build_tree(It it, Se last) {
			std::vector<uint32> spine;
			for (; it != last; ++it) {
				uint32 const id = pool_->make(static_cast<T>(detail::forward_element<Source>(*it)));
				uint32 evicted = 0;
				while (!spine.empty() && pool_->nodes[spine.back()].priority < pool_->nodes[id].priority) {
					evicted = spine.back();
					pool_->pull(evicted);
					spine.pop_back();
				}
				pool_->nodes[id].left = evicted;
				if (!spine.empty()) pool_->nodes[spine.back()].right = id;
				spine.push_back(id);
			}
			for (std::size_t i = spine.size(); i-- > 0;) pool_->pull(spine[i]);
			return spine.empty() ? 0 : spine.front();
		}

		template<typename Visit>
			requires std::invocable<Visit &, T const &>
		void visit_subtree(uint32 t, bool is_flipped, slot<F> const &acc, Visit &visit) const {
			if (t == 0) return;
			node const &nd = pool_->nodes[t];
			if constexpr (IsReversible) is_flipped = is_flipped != nd.is_reversed;
			slot<F> const below = [&] {
				if constexpr (has_tags)
					return pool_->compose_tags(acc, nd.tag);
				else
					return slot<F>{};
			}();
			visit_subtree(is_flipped ? nd.right : nd.left, is_flipped, below, visit);
			if constexpr (has_tags) {
				T const shown = pool_->map_value(acc, nd.value);
				std::invoke(visit, shown);
			} else {
				std::invoke(visit, nd.value);
			}
			visit_subtree(is_flipped ? nd.left : nd.right, is_flipped, below, visit);
		}

	public:
		/** @brief Stored element type. */
		using value_type = T;
		/** @brief Type used for positions and element counts. */
		using size_type = std::size_t;

		/** @brief An empty sequence (plain tier; algebra tiers start from values). @note Not constexpr (uses a shared_ptr node pool). */
		treap_sequence()
			requires(!has_op)
			: pool_(std::make_shared<pool>()) {}

		/** @brief @p n copies of @p fill (plain tier). @complexity O(n) @note Not constexpr (uses a shared_ptr node pool). */
		treap_sequence(std::size_t n, T fill)
			requires(!has_op)
			: pool_(std::make_shared<pool>()) {
			auto const same = std::views::repeat(fill, n);
			root_ = build_tree<decltype(same)>(std::ranges::begin(same), std::ranges::end(same));
		}

		/** @brief From initial values (plain tier). @complexity O(n) @note Not constexpr (uses a shared_ptr node pool). */
		template<std::ranges::sized_range R>
			requires(!has_op && requires(detail::forwarded_element_t<R> element) { static_cast<T>(std::forward<detail::forwarded_element_t<R>>(element)); })
		explicit treap_sequence(R &&values) : pool_(std::make_shared<pool>()) {
			root_ = build_tree<R>(std::ranges::begin(values), std::ranges::end(values));
		}

		/** @brief @p n elements, all equal to @p s_identity (fold tier). @complexity O(n) @note Not constexpr (uses a shared_ptr node pool). */
		template<typename OpArg = Op>
			requires(has_op && !has_tags && std::same_as<OpArg, Op>)
		treap_sequence(std::size_t n, T s_identity, std::type_identity_t<OpArg> op) : pool_(std::make_shared<pool>(detail::move_or_copy(op), slot<Mapping>{}, slot<Compose>{}, std::move(s_identity))) {
			auto const same = std::views::repeat(pool_->element_identity, n);
			root_ = build_tree<decltype(same)>(std::ranges::begin(same), std::ranges::end(same));
		}

		/**
		 * @brief From initial values with a fold algebra; `T` is non-deduced so class
		 * deduction goes through the guide below, which rejects narrowing.
		 * @complexity O(n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<std::ranges::sized_range R, typename OpArg = Op>
			requires(has_op && !has_tags && std::same_as<OpArg, Op>
					 && requires(detail::forwarded_element_t<R> element) { static_cast<T>(std::forward<detail::forwarded_element_t<R>>(element)); })
		treap_sequence(R &&values, std::type_identity_t<T> s_identity, std::type_identity_t<OpArg> op) : pool_(std::make_shared<pool>(detail::move_or_copy(op), slot<Mapping>{}, slot<Compose>{}, std::move(s_identity))) {
			root_ = build_tree<R>(std::ranges::begin(values), std::ranges::end(values));
		}

		/**
		 * @brief @p n elements equal to @p s_identity, with the full tag algebra.
		 * @complexity O(n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<typename OpArg = Op, typename FArg = F, typename MapArg = Mapping, typename CompArg = Compose>
			requires(has_tags && std::same_as<OpArg, Op> && std::same_as<FArg, F> && std::same_as<MapArg, Mapping>
					 && std::same_as<CompArg, Compose>)
		treap_sequence(std::size_t n, T s_identity, std::type_identity_t<FArg> f_identity, std::type_identity_t<OpArg> op, std::type_identity_t<MapArg> mapping, std::type_identity_t<CompArg> compose)
			: pool_(std::make_shared<pool>(detail::move_or_copy(op), detail::move_or_copy(mapping), detail::move_or_copy(compose), std::move(s_identity), std::move(f_identity))) {
			auto const same = std::views::repeat(pool_->element_identity, n);
			root_ = build_tree<decltype(same)>(std::ranges::begin(same), std::ranges::end(same));
		}

		/**
		 * @brief From initial values with the full tag algebra; narrowing-guarded as
		 * above. @complexity O(n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<std::ranges::sized_range R, typename OpArg = Op, typename FArg = F, typename MapArg = Mapping,
				 typename CompArg = Compose>
			requires(has_tags && std::same_as<OpArg, Op> && std::same_as<FArg, F> && std::same_as<MapArg, Mapping>
					 && std::same_as<CompArg, Compose> && requires(detail::forwarded_element_t<R> element) { static_cast<T>(std::forward<detail::forwarded_element_t<R>>(element)); })
		treap_sequence(R &&values, std::type_identity_t<T> s_identity, std::type_identity_t<FArg> f_identity, std::type_identity_t<OpArg> op, std::type_identity_t<MapArg> mapping,
					   std::type_identity_t<CompArg> compose)
			: pool_(std::make_shared<pool>(detail::move_or_copy(op), detail::move_or_copy(mapping), detail::move_or_copy(compose), std::move(s_identity), std::move(f_identity))) {
			root_ = build_tree<R>(std::ranges::begin(values), std::ranges::end(values));
		}

		/** @brief Clone the source nodes into its shared pool. @param o Source handle. @complexity O(source size) @note Not constexpr (uses a shared_ptr node pool). */
		treap_sequence(treap_sequence const &o) : pool_(o.pool_), root_(o.pool_ ? o.pool_->clone(o.root_) : 0) {}

		/** @brief Transfer the pool and tree root. @param o Source, empty on return. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
		treap_sequence(treap_sequence &&o) noexcept : pool_(std::move(o.pool_)), root_(o.root_) { o.root_ = 0; }

		/** @brief Recycle this tree, then clone the source into its pool. @param o Source tree. @return `*this`. @complexity O(destination size + source size) @note Not constexpr (uses a shared_ptr node pool). */
		treap_sequence &operator=(treap_sequence const &o) {
			if (this != &o) {
				if (pool_) pool_->recycle(root_);
				pool_ = o.pool_;
				root_ = pool_ ? pool_->clone(o.root_) : 0;
			}
			return *this;
		}

		/** @brief Recycle this tree, then transfer the source handle. @param o Source, empty on return. @return `*this`. @complexity O(destination size) @note Not constexpr (uses a shared_ptr node pool). */
		treap_sequence &operator=(treap_sequence &&o) noexcept {
			if (this != &o) {
				if (pool_) pool_->recycle(root_);
				pool_ = std::move(o.pool_);
				root_ = o.root_;
				o.root_ = 0;
			}
			return *this;
		}

		/** @brief Recycle this handle's nodes into its shared pool. @complexity O(n) @note Not constexpr (uses a shared_ptr node pool). */
		~treap_sequence() {
			if (pool_) pool_->recycle(root_);
		}

		/** @brief Number of elements in this handle; zero for a moved-from handle. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
		std::size_t size() const noexcept { return pool_ ? pool_->nodes[root_].size : 0; }

		/** @brief Whether this handle contains no elements. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
		bool empty() const noexcept { return size() == 0; }

		/**
		 * @brief Insert @p x before position @p i. @pre `i <= size()`
		 * @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		void insert(std::size_t i, T x) {
			atl_assert(pool_ && i <= size());
			auto [l, r] = split_prefix(root_, i);
			root_ = base::merge(*pool_, base::merge(*pool_, l, pool_->make(std::move(x))), r);
		}

		/** @brief Remove the element at @p i. @pre `i < size()` @complexity expected O(log n) @note Not constexpr (uses a shared_ptr node pool). */
		void erase(std::size_t i) { erase(i, i + 1); }

		/** @brief Remove `[l, r)`. @pre `l <= r <= size()` @complexity expected O(log n + r - l) @note Not constexpr (uses a shared_ptr node pool). */
		void erase(std::size_t l, std::size_t r) {
			atl_assert(pool_ && l <= r && r <= size());
			auto [a, b, c] = split_range(l, r);
			pool_->recycle(b);
			root_ = base::merge(*pool_, a, c);
		}

		/** @brief Append @p x. @complexity expected O(log n) @note Not constexpr (uses a shared_ptr node pool). */
		void push_back(T x) {
			atl_assert(pool_);
			root_ = base::merge(*pool_, root_, pool_->make(std::move(x)));
		}

		/** @brief Prepend @p x. @complexity expected O(log n) @note Not constexpr (uses a shared_ptr node pool). */
		void push_front(T x) {
			atl_assert(pool_);
			root_ = base::merge(*pool_, pool_->make(std::move(x)), root_);
		}

		/**
		 * @brief Insert `T(args...)` before position @p i: built once from
		 *        @p args and moved into the pool's node (no copy, one move).
		 * @pre `i <= size()` @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		void emplace(std::size_t i, Args &&...args) {
			atl_assert(pool_ && i <= size());
			auto [l, r] = split_prefix(root_, i);
			root_ = base::merge(*pool_, base::merge(*pool_, l, pool_->make(std::forward<Args>(args)...)), r);
		}

		/** @brief Append `T(args...)`. @complexity expected O(log n) @note Not constexpr (uses a shared_ptr node pool). */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		void emplace_back(Args &&...args) {
			atl_assert(pool_);
			root_ = base::merge(*pool_, root_, pool_->make(std::forward<Args>(args)...));
		}

		/** @brief Prepend `T(args...)`. @complexity expected O(log n) @note Not constexpr (uses a shared_ptr node pool). */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		void emplace_front(Args &&...args) {
			atl_assert(pool_);
			root_ = base::merge(*pool_, pool_->make(std::forward<Args>(args)...), root_);
		}

		/** @brief Remove the last element. @pre `!empty()`. @complexity expected O(log n) @note Not constexpr (uses a shared_ptr node pool). */
		void pop_back() { erase(size() - 1); }

		/** @brief Remove the first element. @pre `!empty()`. @complexity expected O(log n) @note Not constexpr (uses a shared_ptr node pool). */
		void pop_front() { erase(0); }

		/**
		 * @brief The value at @p i (plain reference — no tags exist to settle).
		 * @pre `i < size()` @complexity expected O(log n)
		 * @note The reference is invalidated by any node allocation on a
		 *       handle sharing this pool (see the class note).
		 */
		[[nodiscard]] T const &at(std::size_t i) const
			requires(!has_tags)
		{
			atl_assert(pool_ && i < size());
			uint32 t = root_;
			auto k = uint32(i);
			[[maybe_unused]] bool is_flipped = false;
			for (;;) {
				node const &nd = pool_->nodes[t];
				if constexpr (IsReversible) is_flipped = is_flipped != nd.is_reversed;
				uint32 const lc = is_flipped ? nd.right : nd.left;
				uint32 const ls = pool_->nodes[lc].size;
				if (k < ls)
					t = lc;
				else if (k == ls)
					return nd.value;
				else
					k -= ls + 1, t = is_flipped ? nd.left : nd.right;
			}
		}

		/**
		 * @brief The value at @p i, with pending path tags folded in on the fly —
		 * read-only. @pre `i < size()` @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		[[nodiscard]] T at(std::size_t i) const
			requires(has_tags)
		{
			atl_assert(pool_ && i < size());
			uint32 t = root_;
			auto k = uint32(i);
			F acc = pool_->tag_identity;
			[[maybe_unused]] bool is_flipped = false;
			for (;;) {
				node const &nd = pool_->nodes[t];
				if constexpr (IsReversible) is_flipped = is_flipped != nd.is_reversed;
				uint32 const lc = is_flipped ? nd.right : nd.left;
				uint32 const ls = pool_->nodes[lc].size;
				if (k == ls) return pool_->map_value(acc, nd.value);
				acc = pool_->compose_tags(acc, nd.tag); // descending: nd's tag is pending for its children
				if (k < ls)
					t = lc;
				else
					k -= ls + 1, t = is_flipped ? nd.left : nd.right;
			}
		}

		/** @brief Assign @p x to position @p i. @pre `i < size()` @complexity expected O(log n) @note Not constexpr (uses a shared_ptr node pool). */
		void set(std::size_t i, T x) {
			atl_assert(pool_ && i < size());
			auto [a, b, c] = split_range(i, i + 1); // b: a single, settled node
			node &nd = pool_->nodes[b];
			nd.value = std::move(x);
			if constexpr (has_op) nd.aggregate = nd.value;
			if constexpr (IsReversible && has_op) nd.reversed_aggregate = nd.value;
			rejoin(a, b, c);
		}

		/**
		 * @brief Fold of `[l, r)`, or `s_identity` if empty. A read-only two-boundary
		 * descent (pending tags accumulated, never settled) — const, no
		 * restructuring. @pre `l <= r <= size()` @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		[[nodiscard]] T fold(std::size_t l, std::size_t r) const
			requires(has_op)
		{
			atl_assert(pool_ && l <= r && r <= size());
			if (l == r) return pool_->element_identity;
			if constexpr (has_tags)
				return fold_range(root_, uint32(l), uint32(r), false, pool_->tag_identity);
			else
				return fold_range(root_, uint32(l), uint32(r), false);
		}

		/** @brief Fold of the whole sequence. @complexity O(1) @note Not constexpr (uses a shared_ptr node pool). */
		[[nodiscard]] T fold() const
			requires(has_op)
		{
			atl_assert(pool_);
			return root_ != 0 ? pool_->nodes[root_].aggregate : pool_->element_identity;
		}

		/**
		 * @brief Apply tag @p f to every element of `[l, r)`. @pre `l <= r <= size()`
		 * @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<typename FArg = F>
			requires(has_tags && std::same_as<FArg, F>)
		void apply(std::size_t l, std::size_t r, FArg const &f) {
			atl_assert(pool_ && l <= r && r <= size());
			if (l == r) return;
			auto [a, b, c] = split_range(l, r);
			pool_->apply_tag(b, f);
			rejoin(a, b, c);
		}

		/**
		 * @brief Reverse the order of `[l, r)`. @pre `l <= r <= size()`
		 * @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		void reverse(std::size_t l, std::size_t r)
			requires(IsReversible)
		{
			atl_assert(pool_ && l <= r && r <= size());
			if (r - l < 2) return;
			auto [a, b, c] = split_range(l, r);
			pool_->flip(b);
			rejoin(a, b, c);
		}

		/**
		 * @brief Largest `r` in `[l, size()]` with `pred(fold(l, r))` true. Requires
		 * `pred` monotone and `pred(s_identity)`. @pre `l <= size()`
		 * @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		template<typename Pred>
			requires std::predicate<Pred &, T const &>
		[[nodiscard]] std::size_t max_right(std::size_t l, Pred pred)
			requires(has_op)
		{
			atl_assert(pool_ && l <= size() && std::invoke(pred, std::as_const(pool_->element_identity)));
			auto [a, b] = split_prefix(root_, l);
			std::size_t count = 0;
			T acc = pool_->element_identity;
			for (uint32 t = b; t != 0;) {
				t = pool_->push(t);
				node const &nd = pool_->nodes[t];
				if (nd.left != 0) {
					if (T cand = pool_->fold_values(acc, pool_->nodes[nd.left].aggregate); std::invoke(pred, std::as_const(cand))) {
						acc = std::move(cand);
						count += pool_->nodes[nd.left].size;
					} else {
						t = nd.left;
						continue;
					}
				}
				if (T cand = pool_->fold_values(acc, nd.value); std::invoke(pred, std::as_const(cand))) {
					acc = std::move(cand);
					++count;
					t = nd.right;
				} else {
					break;
				}
			}
			root_ = base::merge(*pool_, a, b);
			return l + count;
		}

		/**
		 * @brief Detach and return `[i, size())`; this sequence keeps `[0, i)`. The two
		 * share the pool (so `concat` between them stays O(log n)).
		 * @pre `i <= size()` @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		[[nodiscard]] treap_sequence split(std::size_t i) {
			atl_assert(pool_ && i <= size());
			auto [l, r] = split_prefix(root_, i);
			root_ = l;
			return treap_sequence(pool_, r);
		}

		/**
		 * @brief An empty sequence sharing this one's pool — a valid `concat`
		 * partner, in O(1) (unlike copying-then-clearing).
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		[[nodiscard]] treap_sequence sibling() const {
			atl_assert(pool_ != nullptr);
			return treap_sequence(pool_, 0);
		}

		/**
		 * @brief Append @p other (emptied by the call). @pre Distinct source on the same pool (a sequence
		 * derived from this one via split/copy). @complexity expected O(log n)
		 * @note Not constexpr (uses a shared_ptr node pool).
		 */
		void concat(treap_sequence &&other) {
			atl_assert(this != &other && pool_ && other.pool_ == pool_);
			root_ = base::merge(*pool_, root_, other.root_);
			other.root_ = 0;
		}

		/** @brief In-order visit of every element's current value. @complexity O(n) @note Not constexpr (uses a shared_ptr node pool). */
		template<typename Visit>
			requires std::invocable<Visit &, T const &>
		void for_each(Visit visit) const {
			if (!pool_ || root_ == 0) return;
			visit_subtree(root_, false, pool_->tag_identity, visit);
		}

		/** @brief Empty the sequence, recycling its nodes. @complexity O(n) @note Not constexpr (uses a shared_ptr node pool). */
		void clear() {
			if (pool_) pool_->recycle(root_);
			root_ = 0;
		}
	};

	/**
	 * @brief The reversible tier: same class, `reverse(l, r)` enabled (flip tags, and a
	 * second reversed aggregate per node when `Op` is present).
	 */
	template<typename T, typename Op = void, typename F = void, typename Mapping = void, typename Compose = void>
	using reversible_treap_sequence = treap_sequence<T, Op, F, Mapping, Compose, true>;

	/** @brief Deduce a plain non-reversible sequence from a sized range. */
	template<std::ranges::sized_range R>
	treap_sequence(R &&) -> treap_sequence<std::ranges::range_value_t<R>>;
	/** @brief Deduce the plain element type from the fill value. */
	template<typename T>
	treap_sequence(std::size_t, T) -> treap_sequence<T>;
	/** @brief Deduce the element and fold types from a count and identity. */
	template<typename T, binary_operation<T> Op>
	treap_sequence(std::size_t, T, Op) -> treap_sequence<T, Op>;
	/**
	 * @brief Deduces `T` from the identity; rejected if that would narrow the range's
	 * values (spell the class arguments explicitly if intended).
	 */
	template<std::ranges::sized_range R, typename T, binary_operation<T> Op>
		requires requires(std::ranges::range_value_t<R> v) { T{std::move(v)}; }
	treap_sequence(R &&, T, Op) -> treap_sequence<T, Op>;
	/** @brief Deduce a tagged sequence from a count and both identities. */
	template<typename T, typename F, binary_operation<T> Op, action<F, T> Mapping, binary_operation<F> Compose>
	treap_sequence(std::size_t, T, F, Op, Mapping, Compose) -> treap_sequence<T, Op, F, Mapping, Compose>;
	/** @brief Deduce a tagged sequence from a range, rejecting narrowing to the identity type. */
	template<std::ranges::sized_range R, typename T, typename F, binary_operation<T> Op, action<F, T> Mapping,
			 binary_operation<F> Compose>
		requires requires(std::ranges::range_value_t<R> v) { T{std::move(v)}; }
	treap_sequence(R &&, T, F, Op, Mapping, Compose) -> treap_sequence<T, Op, F, Mapping, Compose>;
}
