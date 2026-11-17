#pragma once
#include <atl/ds/detail/algebra.hpp>
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/persistent_arena.hpp>
#include <atl/ds/detail/treap_base.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <bit>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Fully persistent implicit treap: O(log n) sequence surgery where
	 *        every version is kept.
	 *
	 * The path-copying counterpart of @ref atl::treap_sequence, driven by the *same*
	 * `detail::treap_base` engine — the store's `rebind`/`push` return fresh
	 * ids from a `detail::persistent_arena` instead of mutating, chosen at
	 * compile time. The tags tier is the exception: it joins and splits with its
	 * own accumulator-carrying descents (pending tags are composed down rather
	 * than pushed into shared nodes), bypassing the engine. Mutators are `const`
	 * and return new versions; any version stays usable; copies are O(1) handle
	 * copies (structure is shared, never aliased mutably). Reads (`at`, `fold`)
	 * accumulate pending tags instead of settling them: const and
	 * allocation-free.
	 *
	 * Tiers and algebra exactly as @ref atl::treap_sequence (plain / `Op` folds /
	 * `F` tags), minus reversal and `max_right` (add on demand).
	 *
	 * @note Not constexpr (versions own a std::shared_ptr pool); this applies to all members.
	 * @note Constructor algebra arguments are non-deduced; const-qualified callable
	 *       types accept ordinary callable arguments. Defaulted template parameters
	 *       keep disabled tiers from declaring void parameters.
	 *
	 * @tparam T       The element type.
	 * @tparam Op      Optional element fold.
	 * @tparam F       Optional tag type (requires `Op`).
	 * @tparam Mapping Tag application (with `F`).
	 * @tparam Compose Tag composition, newer first (with `F`).
	 */
	template<std::semiregular T, typename Op = void, typename F = void, typename Mapping = void,
			 typename Compose = void>
		requires((std::is_void_v<Op> || binary_operation<Op, T>)
				 && (std::is_void_v<F> == std::is_void_v<Mapping>)
				 && (std::is_void_v<F> == std::is_void_v<Compose>)
				 && (std::is_void_v<F> || (!std::is_void_v<Op> && std::semiregular<F>))
				 && (std::is_void_v<Mapping> || action<Mapping, F, T>)
				 && (std::is_void_v<Compose> || binary_operation<Compose, F>))
	class persistent_treap_sequence {
		static constexpr bool has_op = !std::is_void_v<Op>;
		static constexpr bool has_tags = !std::is_void_v<F>;

		/**
		 * @brief Stored sizes cap here; logical sizes beyond it (repeated self-concat)
		 * are all reported as this value.
		 */
		static constexpr uint32 saturated = 0xFFFFFFFFu;

		struct no_slot {};

		struct no_agg_slot {};

		struct no_tag_slot {};

		struct aggregate_slot {
			T aggregate;
		};

		struct tag_slot {
			F tag;
		};

		// No priority field: the persistent tree joins with a size-weighted coin
		// per call (see treap_base::merge_weighted) — stored priorities would be
		// correlated by version sharing (e.g. concatenating a list with itself).
		struct node : std::conditional_t<has_op, aggregate_slot, no_agg_slot>,
					  std::conditional_t<has_tags, tag_slot, no_tag_slot> {
			T value;
			uint32 left = 0, right = 0, size = 1;
		};

		template<typename X>
		using slot = std::conditional_t<std::is_void_v<X>, no_slot, X>;

		/**
		 * @brief Path-copying treap_store: `rebind`/`push` allocate copies in the
		 * arena and return fresh ids — old versions never change.
		 */
		struct pool {
			uint64 rng;
			[[no_unique_address]] slot<Op> op;
			[[no_unique_address]] slot<Mapping> mapping;
			[[no_unique_address]] slot<Compose> compose;
			[[no_unique_address]] std::conditional_t<has_op, T, no_slot> element_identity;
			[[no_unique_address]] slot<F> tag_identity;
			detail::persistent_arena<node> nodes;

			pool() : rng(detail::hash_seed() ^ 0x9e3779b97f4a7c15ULL), nodes(sentinel()) {}

			pool(T identity, slot<Op> operation)
				requires has_op && (!has_tags)
				: rng(detail::hash_seed() ^ 0x9e3779b97f4a7c15ULL), op(detail::move_or_copy(operation)),
				  element_identity(std::move(identity)), nodes(sentinel()) {
				nodes[0].aggregate = element_identity;
			}

			pool(T identity, slot<F> tag, slot<Op> operation, slot<Mapping> map, slot<Compose> composition)
				requires has_tags
				: rng(detail::hash_seed() ^ 0x9e3779b97f4a7c15ULL), op(detail::move_or_copy(operation)), mapping(detail::move_or_copy(map)),
				  compose(detail::move_or_copy(composition)), element_identity(std::move(identity)), tag_identity(std::move(tag)), nodes(sentinel()) {
				nodes[0].aggregate = element_identity;
			}

			static node sentinel() {
				node nd{};
				nd.size = 0;
				return nd;
			}

			uint32 left(uint32 t) const { return nodes[t].left; }

			uint32 right(uint32 t) const { return nodes[t].right; }

			uint32 size(uint32 t) const { return nodes[t].size; }

			uint32 priority(uint32) const { return 0; } // unused: weighted joins only

			uint64 rand() { return detail::splitmix64(rng += 0x9e3779b97f4a7c15ULL); }

			// Sizes saturate at uint32 max: versions can grow beyond 2^32 logical
			// elements (repeated self-concat). Positional ops (insert/split_at)
			// remain exact only below the cap; the structural end-splits never
			// look at sizes, and saturated joins switch to the spine-pinning
			// merge (see joined).
			void pull_into(node &nd) const {
				uint64 const total = 1 + uint64(nodes[nd.left].size) + nodes[nd.right].size;
				nd.size = total > uint64(saturated) ? saturated : uint32(total);
				if constexpr (has_op) nd.aggregate = op(op(nodes[nd.left].aggregate, nd.value), nodes[nd.right].aggregate);
			}

			uint32 rebind(uint32 t, uint32 l, uint32 r) {
				node nd = nodes[t]; // copied out per the arena contract
				nd.left = l;
				nd.right = r;
				pull_into(nd);
				return nodes.make(std::move(nd));
			}

			/**
			 * @brief A copy of @p t with @p f folded in (value/aggregate now, tag composed
			 * for the subtree below).
			 */
			template<typename FArg = F>
				requires(has_tags && std::same_as<FArg, F>)
			uint32 with_tag(uint32 t, FArg const &f) {
				node nd = nodes[t];
				nd.value = mapping(f, nd.value);
				if constexpr (has_op) nd.aggregate = mapping(f, nd.aggregate);
				nd.tag = compose(f, nd.tag);
				return nodes.make(std::move(nd));
			}

			uint32 push(uint32 t) {
				if constexpr (has_tags) {
					node nd = nodes[t];
					// An identity tag has nothing to settle: skipping it turns
					// the ~5 copies per descent level (self + both children,
					// then the caller's rebind) into the rebind's single copy.
					// Only detectable when F can be compared.
					if constexpr (std::equality_comparable<F>)
						if (nd.tag == tag_identity) return t;
					if (nd.left != 0) nd.left = with_tag(nd.left, nd.tag);
					if (nd.right != 0) nd.right = with_tag(nd.right, nd.tag);
					nd.tag = tag_identity;
					return nodes.make(std::move(nd));
				} else {
					return t;
				}
			}

			uint32 make(T value) {
				node nd{};
				nd.value = std::move(value);
				if constexpr (has_op) nd.aggregate = nd.value;
				if constexpr (has_tags) nd.tag = tag_identity;
				return nodes.make(std::move(nd));
			}

			// The engine's push-then-rebind settles a pending tag with up to four
			// node copies per descent level (self, both children, the rebind).
			// These carry the composed ancestor tag down as `acc` instead and
			// fold it into the path copy the operation makes anyway: one copy
			// per level, plus one for the off-path child under a live tag.

			/**
			 * @brief Whether @p g is knowably the identity tag: detectable only when F
			 * is equality-comparable — otherwise every tag is treated as live,
			 * which is correct but settles more.
			 */
			template<typename FArg = F>
				requires(has_tags && std::same_as<FArg, F>)
			bool is_identity(FArg const &g) const {
				if constexpr (std::equality_comparable<F>)
					return g == tag_identity;
				else
					return false;
			}

			/** @brief @p t with @p g absorbed when @p is_live (and @p t non-null). */
			template<typename FArg = F>
				requires(has_tags && std::same_as<FArg, F>)
			uint32 with_tag_if_live(uint32 t, FArg const &g, bool is_live) {
				return t == 0 || !is_live ? t : with_tag(t, g);
			}

			/**
			 * @brief Copy of subtree @p t with @p f folded over its `[l, r)` (non-empty,
			 * relative). Applies change no structure, so — exactly like a
			 * persistent lazy segment tree — nothing is settled: `acc` carries
			 * the composed ancestor tags into each path copy.
			 */
			template<typename FArg = F>
				requires(has_tags && std::same_as<FArg, F>)
			uint32 apply_range(uint32 t, uint32 l, uint32 r, FArg const &f, FArg const &acc, bool is_live) {
				node nd = nodes[t];
				if (is_live) {
					nd.value = mapping(acc, nd.value);
					if constexpr (has_op) nd.aggregate = mapping(acc, nd.aggregate);
					nd.tag = compose(acc, nd.tag);
				}
				if (l == 0 && r == nd.size) {
					nd.value = mapping(f, nd.value);
					if constexpr (has_op) nd.aggregate = mapping(f, nd.aggregate);
					nd.tag = compose(f, nd.tag);
					return nodes.make(std::move(nd));
				}
				F const down = nd.tag;
				bool const is_down_live = !is_identity(down);
				nd.tag = tag_identity;
				uint32 const ls = nodes[nd.left].size;
				if (r <= ls) {
					nd.left = apply_range(nd.left, l, r, f, down, is_down_live);
					nd.right = with_tag_if_live(nd.right, down, is_down_live);
				} else if (l > ls) {
					nd.right = apply_range(nd.right, l - ls - 1, r - ls - 1, f, down, is_down_live);
					nd.left = with_tag_if_live(nd.left, down, is_down_live);
				} else { // spans this node's own element (l <= ls < r)
					nd.left = l < ls ? apply_range(nd.left, l, ls, f, down, is_down_live) : with_tag_if_live(nd.left, down, is_down_live);
					nd.right = r > ls + 1 ? apply_range(nd.right, 0, r - ls - 1, f, down, is_down_live)
										  : with_tag_if_live(nd.right, down, is_down_live);
					nd.value = mapping(f, nd.value);
				}
				pull_into(nd);
				return nodes.make(std::move(nd));
			}

			/** @brief As treap_base::split_at, with `acc` carried instead of settling. */
			template<typename FArg = F>
				requires(has_tags && std::same_as<FArg, F>)
			std::pair<uint32, uint32> split_at_tagged(uint32 t, uint32 k, FArg const &acc, bool is_live) {
				if (k == 0) return {0, with_tag_if_live(t, acc, is_live)};
				node nd = nodes[t];
				if (k >= nd.size) return {with_tag_if_live(t, acc, is_live), 0};
				if (is_live) {
					nd.value = mapping(acc, nd.value); // aggregate recomputed by the pull
					nd.tag = compose(acc, nd.tag);
				}
				F const down = nd.tag;
				bool const is_down_live = !is_identity(down);
				nd.tag = tag_identity;
				uint32 const ls = nodes[nd.left].size;
				if (k <= ls) {
					auto const parts = split_at_tagged(nd.left, k, down, is_down_live);
					uint32 const a = parts.first;
					nd.left = parts.second;
					nd.right = with_tag_if_live(nd.right, down, is_down_live);
					pull_into(nd);
					return {a, nodes.make(std::move(nd))};
				}
				auto const parts = split_at_tagged(nd.right, k - ls - 1, down, is_down_live);
				uint32 const b = parts.second;
				nd.right = parts.first;
				nd.left = with_tag_if_live(nd.left, down, is_down_live);
				pull_into(nd);
				return {nodes.make(std::move(nd)), b};
			}

			/**
			 * @brief As treap_base::merge_weighted, with per-side accumulators carried
			 * instead of settling.
			 */
			template<typename FArg = F>
				requires(has_tags && std::same_as<FArg, F>)
			uint32 merge_tagged(uint32 a, uint32 b, FArg const &fa, bool is_a_live, FArg const &fb, bool is_b_live) {
				if (a == 0 || b == 0) return with_tag_if_live(a | b, a == 0 ? fb : fa, a == 0 ? is_b_live : is_a_live);
				uint64 const sa = nodes[a].size, total = sa + nodes[b].size;
				if (detail::treap_base::bounded_draw(rand(), total) < sa) {
					node nd = nodes[a];
					if (is_a_live) {
						nd.value = mapping(fa, nd.value);
						nd.tag = compose(fa, nd.tag);
					}
					F const down = nd.tag;
					bool const is_down_live = !is_identity(down);
					nd.tag = tag_identity;
					nd.left = with_tag_if_live(nd.left, down, is_down_live);
					nd.right = merge_tagged(nd.right, b, down, is_down_live, fb, is_b_live);
					pull_into(nd);
					return nodes.make(std::move(nd));
				}
				node nd = nodes[b];
				if (is_b_live) {
					nd.value = mapping(fb, nd.value);
					nd.tag = compose(fb, nd.tag);
				}
				F const down = nd.tag;
				bool const is_down_live = !is_identity(down);
				nd.tag = tag_identity;
				nd.right = with_tag_if_live(nd.right, down, is_down_live);
				nd.left = merge_tagged(a, nd.left, fa, is_a_live, down, is_down_live);
				pull_into(nd);
				return nodes.make(std::move(nd));
			}
		};

		using base = detail::treap_base;

		std::shared_ptr<pool> pool_;

		uint32 joined(uint32 a, uint32 b) const {
			// Saturated sizes make the weighted coin ~50/50 and an interleaving
			// merge then compounds both spines; the spine-pinning join keeps
			// giant merges O(shorter opposing spine) instead.
			if (pool_->size(a) == saturated || pool_->size(b) == saturated)
				return base::merge_spines(*pool_, a, b);
			if constexpr (has_tags)
				return pool_->merge_tagged(a, b, pool_->tag_identity, false, pool_->tag_identity, false);
			else
				return base::merge_weighted(*pool_, a, b, [p = pool_.get()] { return p->rand(); });
		}

		// Split off the first k elements — the accumulator-carrying descent when
		// tags exist, the engine's otherwise.
		std::pair<uint32, uint32> cut(uint32 root, std::size_t k) const {
			if constexpr (has_tags)
				return pool_->split_at_tagged(root, uint32(k), pool_->tag_identity, false);
			else
				return base::split_at(*pool_, root, uint32(k));
		}

		uint32 root_ = 0;

		persistent_treap_sequence(std::shared_ptr<pool> shared, uint32 root)
			: pool_(std::move(shared)), root_(root) {}

		/** @brief The root of `insert`'s result; the public overloads wrap it with a copied or moved pool. */
		uint32 insert_root(std::size_t i, T x) const {
			atl_assert(pool_ && i <= size());
			atl_assert(!is_saturated());
			auto [l, r] = base::split_at(*pool_, root_, uint32(i));
			return joined(joined(l, pool_->make(std::move(x))), r);
		}

		/** @brief The root of `erase`'s result; the public overloads wrap it with a copied or moved pool. */
		uint32 erase_root(std::size_t l, std::size_t r) const {
			atl_assert(pool_ && l <= r && r <= size());
			atl_assert(!is_saturated());
			auto [a, b, c] = split_range(root_, l, r);
			return joined(a, c);
		}

		/** @brief The root of `push_back`'s result; the public overloads wrap it with a copied or moved pool. */
		uint32 push_back_root(T x) const {
			atl_assert(pool_ != nullptr);
			return joined(root_, pool_->make(std::move(x)));
		}

		/** @brief The root of `push_front`'s result; the public overloads wrap it with a copied or moved pool. */
		uint32 push_front_root(T x) const {
			atl_assert(pool_ != nullptr);
			return joined(pool_->make(std::move(x)), root_);
		}

		/** @brief The root of `set`'s result; the public overloads wrap it with a copied or moved pool. */
		uint32 set_root(std::size_t i, T x) const {
			atl_assert(pool_ && i < size());
			atl_assert(!is_saturated());
			auto [a, b, c] = split_range(root_, i, i + 1);
			node nd = pool_->nodes[b]; // a settled single node; rebuild it
			nd.value = std::move(x);
			if constexpr (has_op) nd.aggregate = nd.value;
			uint32 const fresh = pool_->nodes.make(std::move(nd));
			return joined(joined(a, fresh), c);
		}

		/** @brief The root of `apply`'s result; the public overloads wrap it with a copied or moved pool. */
		template<typename FArg = F>
			requires(has_tags && std::same_as<FArg, F>)
		uint32 apply_root(std::size_t l, std::size_t r, FArg const &f) const {
			atl_assert(pool_ && l <= r && r <= size());
			atl_assert(!is_saturated());
			if (l == r) return root_;
			// A direct descent, not split + with_tag + merge: applies change no
			// structure, so no tag ever needs settling — O(log n) path copies.
			return pool_->apply_range(root_, uint32(l), uint32(r), f, pool_->tag_identity, false);
		}

		// O(n) bulk build: perfectly balanced midpoint recursion (fresh nodes, so
		// in-place writes are fine; subsequent randomized joins keep the shape
		// balanced in expectation).
		template<std::ranges::input_range R, std::input_iterator It>
			requires std::constructible_from<T, detail::forwarded_element_t<R>>
		std::pair<uint32, It> build_tree(It it, std::size_t count) {
			if (count == 0) return {0, std::move(it)};
			std::size_t const half = count / 2;
			auto left = build_tree<R>(std::move(it), half);
			uint32 const id = pool_->make(static_cast<T>(detail::forward_element<R>(*left.second)));
			++left.second;
			auto right = build_tree<R>(std::move(left.second), count - half - 1);
			pool_->nodes[id].left = left.first;
			pool_->nodes[id].right = right.first;
			pool_->pull_into(pool_->nodes[id]);
			return {id, std::move(right.second)};
		}

		// Balanced rebuild of a copied end-path: path[lo, hi) are the settled
		// spine nodes root-first, each keeping its off-spine subtree; `edge` is
		// the subtree at the sequence's inner end (past the deepest node). The
		// path nodes must be copied regardless, so shaping the copies as a
		// balanced tree is free — a long end spine costs its length once and
		// O(log) on every later end-split.
		uint32 rebuild_path(std::vector<uint32> const &path, std::size_t lo, std::size_t hi, uint32 edge,
							bool is_front) const {
			if (lo == hi) return edge;
			std::size_t const mid = lo + (hi - lo) / 2;
			if (is_front) {
				// left spine: sequence order is deepest-first, i.e. descending
				// indices — everything past mid goes left, the rest right
				uint32 const l = rebuild_path(path, mid + 1, hi, edge, is_front);
				uint32 const r = rebuild_path(path, lo, mid, pool_->right(path[mid]), is_front);
				return pool_->rebind(path[mid], l, r);
			}
			uint32 const l = rebuild_path(path, lo, mid, pool_->left(path[mid]), is_front);
			uint32 const r = rebuild_path(path, mid + 1, hi, edge, is_front);
			return pool_->rebind(path[mid], l, r);
		}

		// Detach the extreme element on side `is_front`: (that element as a fresh
		// singleton, the rest). Pure descent — no size arithmetic.
		std::pair<uint32, uint32> split_off_end(uint32 t, bool is_front) const {
			std::vector<uint32> path;
			for (;;) {
				t = pool_->push(t); // settle a possible pending tag one level (copies)
				uint32 const into = is_front ? pool_->left(t) : pool_->right(t);
				if (into == 0) break;
				path.push_back(t);
				t = into;
			}
			node const nd = pool_->nodes[t]; // copied out per the arena contract
			uint32 const single = pool_->make(nd.value);
			uint32 const rest = rebuild_path(path, 0, path.size(), is_front ? nd.right : nd.left, is_front);
			return {single, rest};
		}

		struct parts {
			uint32 prefix, middle, suffix;
		};

		/**
		 * @brief Whether this version's stored size has saturated: its logical
		 *        length may exceed 2^32 - 1, so positional descents would do
		 *        arithmetic on capped child sizes. (A version of exactly
		 *        2^32 - 1 elements also reports saturated.)
		 */
		bool is_saturated() const { return pool_->size(root_) == saturated; }

		parts split_range(uint32 root, std::size_t l, std::size_t r) const {
			auto [ab, c] = cut(root, r);
			auto [a, b] = cut(ab, l);
			return {a, b, c};
		}

		// Fold of the non-empty [l, r) inside subtree t — read-only, tags
		// accumulated (identical to treap_sequence's descent).
		template<typename... Acc>
			requires(sizeof...(Acc) <= 1 && (std::same_as<Acc, F> && ...))
		T fold_range(uint32 t, uint32 l, uint32 r, Acc const &...acc) const {
			node const &nd = pool_->nodes[t];
			if (l == 0 && r == nd.size) {
				if constexpr (has_tags)
					return pool_->mapping(acc..., nd.aggregate);
				else
					return nd.aggregate;
			}
			uint32 const ls = pool_->nodes[nd.left].size;
			auto const fold = [&](uint32 child, uint32 cl, uint32 cr) {
				if constexpr (has_tags)
					return fold_range(child, cl, cr, pool_->compose(acc..., nd.tag));
				else
					return fold_range(child, cl, cr);
			};
			if (r <= ls) return fold(nd.left, l, r);
			if (l > ls) return fold(nd.right, l - ls - 1, r - ls - 1);
			auto value_of = [&] {
				if constexpr (has_tags)
					return pool_->mapping(acc..., nd.value);
				else
					return nd.value;
			};
			if (l == ls) {
				T v = value_of();
				return r > ls + 1 ? pool_->op(std::move(v), fold(nd.right, 0, r - ls - 1)) : v;
			}
			T left_part = fold(nd.left, l, ls);
			T v = pool_->op(std::move(left_part), value_of());
			return r > ls + 1 ? pool_->op(std::move(v), fold(nd.right, 0, r - ls - 1)) : v;
		}

	public:
		/** @brief Element type. */
		using value_type = T;
		/** @brief Lazy tag type. */
		using tag_type = F;
		/** @brief Type of element counts and positions. */
		using size_type = std::size_t;

		/** @brief An empty sequence (plain tier). */
		persistent_treap_sequence()
			requires(!has_op)
			: pool_(std::make_shared<pool>()) {}

		/** @brief From initial values (plain tier). @complexity O(n) */
		template<std::ranges::sized_range R>
			requires(!has_op && std::ranges::input_range<R> && std::convertible_to<std::ranges::range_value_t<R>, T>
					 && requires { static_cast<T>(std::declval<detail::forwarded_element_t<R>>()); })
		explicit persistent_treap_sequence(R &&values) : pool_(std::make_shared<pool>()) {
			auto it = std::ranges::begin(values);
			atl_assert(std::ranges::size(values) <= saturated);
			root_ = build_tree<R>(std::move(it), std::ranges::size(values)).first;
		}

		/**
		 * @brief From initial values with a fold algebra (narrowing-guarded deduction).
		 * @complexity O(n)
		 */
		template<std::ranges::sized_range R, typename OpArg = Op>
			requires(has_op && !has_tags && std::same_as<OpArg, Op>
					 && std::ranges::input_range<R> && std::convertible_to<std::ranges::range_value_t<R>, T>
					 && requires { static_cast<T>(std::declval<detail::forwarded_element_t<R>>()); })
		persistent_treap_sequence(R &&values, std::type_identity_t<T> s_identity, std::type_identity_t<OpArg> op)
			: pool_(std::make_shared<pool>(std::move(s_identity), detail::move_or_copy(op))) {
			auto it = std::ranges::begin(values);
			atl_assert(std::ranges::size(values) <= saturated);
			root_ = build_tree<R>(std::move(it), std::ranges::size(values)).first;
		}

		/** @brief From initial values with the full tag algebra. @complexity O(n) */
		template<std::ranges::sized_range R, typename OpArg = Op, typename FArg = F, typename MapArg = Mapping,
				 typename CompArg = Compose>
			requires(has_tags && std::same_as<OpArg, Op> && std::same_as<FArg, F> && std::same_as<MapArg, Mapping>
					 && std::same_as<CompArg, Compose> && std::ranges::input_range<R> && std::convertible_to<std::ranges::range_value_t<R>, T>
					 && requires { static_cast<T>(std::declval<detail::forwarded_element_t<R>>()); })
		persistent_treap_sequence(R &&values, std::type_identity_t<T> s_identity, std::type_identity_t<FArg> f_identity, std::type_identity_t<OpArg> op,
								  std::type_identity_t<MapArg> mapping, std::type_identity_t<CompArg> compose)
			: pool_(std::make_shared<pool>(std::move(s_identity), std::move(f_identity), detail::move_or_copy(op), detail::move_or_copy(mapping), detail::move_or_copy(compose))) {
			auto it = std::ranges::begin(values);
			atl_assert(std::ranges::size(values) <= saturated);
			root_ = build_tree<R>(std::move(it), std::ranges::size(values)).first;
		}

		/** @brief Number of elements; saturates at 2^32 - 1 (see the class notes). */
		std::size_t size() const noexcept { return pool_ ? pool_->nodes[root_].size : 0; }

		/** @brief Whether this version contains no elements. @complexity O(1) */
		bool empty() const noexcept { return size() == 0; }

		/**
		 * @brief The version with @p x inserted before position @p i.
		 * @pre `i <= size()`, and `size()` is not saturated (positions are
		 *      exact only below 2^32 - 1)
		 * @complexity O(log n) time and new nodes
		 */
		[[nodiscard]] persistent_treap_sequence insert(std::size_t i, T x) const & {
			uint32 const root = insert_root(i, std::move(x));
			return {pool_, root};
		}

		/** @brief `insert` consuming this version: the pool is moved, not copied (no atomic refcount traffic), so `s = std::move(s).insert(...)` is the cheap linear idiom. The moved-from sequence may only be destroyed or assigned to. Otherwise as the lvalue overload. */
		[[nodiscard]] persistent_treap_sequence insert(std::size_t i, T x) && {
			uint32 const root = insert_root(i, std::move(x));
			return {std::move(pool_), root};
		}

		/**
		 * @brief The version with `[l, r)` removed. @pre `l <= r <= size()`, `size()`
		 * not saturated @complexity O(log n)
		 */
		[[nodiscard]] persistent_treap_sequence erase(std::size_t l, std::size_t r) const & {
			uint32 const root = erase_root(l, r);
			return {pool_, root};
		}

		/** @brief `erase` consuming this version: the pool is moved, not copied (no atomic refcount traffic), so `s = std::move(s).erase(...)` is the cheap linear idiom. The moved-from sequence may only be destroyed or assigned to. Otherwise as the lvalue overload. */
		[[nodiscard]] persistent_treap_sequence erase(std::size_t l, std::size_t r) && {
			uint32 const root = erase_root(l, r);
			return {std::move(pool_), root};
		}

		/**
		 * @brief The version with element @p i removed.
		 * @pre `i < size()`, `size()` not saturated
		 * @complexity O(log n)
		 */
		[[nodiscard]] persistent_treap_sequence erase(std::size_t i) const & { return erase(i, i + 1); }

		/** @brief `erase` of one element consuming this version (see the rvalue `erase`). */
		[[nodiscard]] persistent_treap_sequence erase(std::size_t i) && { return std::move(*this).erase(i, i + 1); }

		/** @brief The version with @p x appended. @complexity O(log n) expected */
		[[nodiscard]] persistent_treap_sequence push_back(T x) const & {
			uint32 const root = push_back_root(std::move(x));
			return {pool_, root};
		}

		/** @brief `push_back` consuming this version: the pool is moved, not copied (no atomic refcount traffic), so `s = std::move(s).push_back(...)` is the cheap linear idiom. The moved-from sequence may only be destroyed or assigned to. Otherwise as the lvalue overload. */
		[[nodiscard]] persistent_treap_sequence push_back(T x) && {
			uint32 const root = push_back_root(std::move(x));
			return {std::move(pool_), root};
		}

		/** @brief The version with @p x prepended. @complexity O(log n) expected */
		[[nodiscard]] persistent_treap_sequence push_front(T x) const & {
			uint32 const root = push_front_root(std::move(x));
			return {pool_, root};
		}

		/** @brief `push_front` consuming this version: the pool is moved, not copied (no atomic refcount traffic), so `s = std::move(s).push_front(...)` is the cheap linear idiom. The moved-from sequence may only be destroyed or assigned to. Otherwise as the lvalue overload. */
		[[nodiscard]] persistent_treap_sequence push_front(T x) && {
			uint32 const root = push_front_root(std::move(x));
			return {std::move(pool_), root};
		}

		/**
		 * @brief The version with element @p i replaced. @pre `i < size()`, `size()`
		 * not saturated @complexity O(log n)
		 */
		[[nodiscard]] persistent_treap_sequence set(std::size_t i, T x) const & {
			uint32 const root = set_root(i, std::move(x));
			return {pool_, root};
		}

		/** @brief `set` consuming this version: the pool is moved, not copied (no atomic refcount traffic), so `s = std::move(s).set(...)` is the cheap linear idiom. The moved-from sequence may only be destroyed or assigned to. Otherwise as the lvalue overload. */
		[[nodiscard]] persistent_treap_sequence set(std::size_t i, T x) && {
			uint32 const root = set_root(i, std::move(x));
			return {std::move(pool_), root};
		}

		/**
		 * @brief The value at @p i (pending tags folded on the fly).
		 * @pre `i < size()`, `size()` not saturated
		 * @complexity O(log n), allocation-free
		 */
		[[nodiscard]] T at(std::size_t i) const {
			atl_assert(pool_ && i < size());
			atl_assert(!is_saturated());
			uint32 t = root_;
			auto k = uint32(i);
			[[maybe_unused]] slot<F> acc;
			if constexpr (has_tags) acc = pool_->tag_identity;
			for (;;) {
				node const &nd = pool_->nodes[t];
				uint32 const ls = pool_->nodes[nd.left].size;
				if (k == ls) {
					if constexpr (has_tags)
						return pool_->mapping(acc, nd.value);
					else
						return nd.value;
				}
				if constexpr (has_tags) acc = pool_->compose(acc, nd.tag);
				if (k < ls)
					t = nd.left;
				else
					k -= ls + 1, t = nd.right;
			}
		}

		/**
		 * @brief Fold of `[l, r)` in this version — const, allocation-free.
		 * @pre `l <= r <= size()`, `size()` not saturated @complexity O(log n)
		 */
		[[nodiscard]] T fold(std::size_t l, std::size_t r) const
			requires(has_op)
		{
			atl_assert(pool_ && l <= r && r <= size());
			atl_assert(!is_saturated());
			if (l == r) return pool_->element_identity;
			if constexpr (has_tags)
				return fold_range(root_, uint32(l), uint32(r), pool_->tag_identity);
			else
				return fold_range(root_, uint32(l), uint32(r));
		}

		/** @brief Fold of the whole sequence. @complexity O(1) */
		[[nodiscard]] T fold() const
			requires(has_op)
		{
			atl_assert(pool_ != nullptr);
			return root_ != 0 ? pool_->nodes[root_].aggregate : pool_->element_identity;
		}

		/**
		 * @brief The version with tag @p f applied to `[l, r)`.
		 * @pre `l <= r <= size()`, `size()` not saturated @complexity O(log n)
		 */
		template<typename FArg = F>
			requires(has_tags && std::same_as<FArg, F>)
		[[nodiscard]] persistent_treap_sequence apply(std::size_t l, std::size_t r, FArg const &f) const & {
			uint32 const root = apply_root(l, r, f);
			return {pool_, root};
		}

		/** @brief `apply` consuming this version: the pool is moved, not copied (no atomic refcount traffic), so `s = std::move(s).apply(...)` is the cheap linear idiom. The moved-from sequence may only be destroyed or assigned to. Otherwise as the lvalue overload. */
		template<typename FArg = F>
			requires(has_tags && std::same_as<FArg, F>)
		[[nodiscard]] persistent_treap_sequence apply(std::size_t l, std::size_t r, FArg const &f) && {
			uint32 const root = apply_root(l, r, f);
			return {std::move(pool_), root};
		}

		/**
		 * @brief The versions `[first element]` and `[the rest]` — structural descent,
		 * valid regardless of how large the sequence's logical size is.
		 * @pre `!empty()` @complexity O(log n)
		 */
		[[nodiscard]] std::pair<persistent_treap_sequence, persistent_treap_sequence> split_first() const & {
			atl_assert(pool_ && !empty());
			auto [f, rest] = split_off_end(root_, true);
			return {persistent_treap_sequence(pool_, f), persistent_treap_sequence(pool_, rest)};
		}

		/** @brief `split_first` consuming this version: the pool is moved into the second result, copied into the first. The moved-from sequence may only be destroyed or assigned to. */
		[[nodiscard]] std::pair<persistent_treap_sequence, persistent_treap_sequence> split_first() && {
			atl_assert(pool_ && !empty());
			auto [f, rest] = split_off_end(root_, true);
			return {persistent_treap_sequence(pool_, f), persistent_treap_sequence(std::move(pool_), rest)};
		}

		/**
		 * @brief The versions `[all but last]` and `[the last element]` — structural.
		 * @pre `!empty()` @complexity O(log n)
		 */
		[[nodiscard]] std::pair<persistent_treap_sequence, persistent_treap_sequence> split_last() const & {
			atl_assert(pool_ && !empty());
			auto [l, rest] = split_off_end(root_, false);
			return {persistent_treap_sequence(pool_, rest), persistent_treap_sequence(pool_, l)};
		}

		/** @brief `split_last` consuming this version: the pool is moved into the second result, copied into the first. The moved-from sequence may only be destroyed or assigned to. */
		[[nodiscard]] std::pair<persistent_treap_sequence, persistent_treap_sequence> split_last() && {
			atl_assert(pool_ && !empty());
			auto [l, rest] = split_off_end(root_, false);
			return {persistent_treap_sequence(pool_, rest), persistent_treap_sequence(std::move(pool_), l)};
		}

		/**
		 * @brief The two versions `[0, i)` and `[i, size())` — this one is unchanged.
		 * @pre `i <= size()`, `size()` not saturated @complexity O(log n)
		 */
		[[nodiscard]] std::pair<persistent_treap_sequence, persistent_treap_sequence> split(std::size_t i) const & {
			atl_assert(pool_ && i <= size());
			atl_assert(!is_saturated());
			auto [l, r] = cut(root_, i);
			return {persistent_treap_sequence(pool_, l), persistent_treap_sequence(pool_, r)};
		}

		/** @brief `split` consuming this version: the pool is moved into the second result, copied into the first. The moved-from sequence may only be destroyed or assigned to. */
		[[nodiscard]] std::pair<persistent_treap_sequence, persistent_treap_sequence> split(std::size_t i) && {
			atl_assert(pool_ && i <= size());
			atl_assert(!is_saturated());
			auto [l, r] = cut(root_, i);
			return {persistent_treap_sequence(pool_, l), persistent_treap_sequence(std::move(pool_), r)};
		}

		/** @brief The version `this ++ other`. @pre same arena. @complexity O(log n) */
		[[nodiscard]] persistent_treap_sequence concat(persistent_treap_sequence const &other) const & {
			atl_assert(pool_ && other.pool_ == pool_);
			return {pool_, joined(root_, other.root_)};
		}

		/** @brief `concat` consuming this version: the pool is moved, not copied. The moved-from sequence may only be destroyed or assigned to. */
		[[nodiscard]] persistent_treap_sequence concat(persistent_treap_sequence const &other) && {
			atl_assert(pool_ && other.pool_ == pool_);
			uint32 const root = joined(root_, other.root_);
			return {std::move(pool_), root};
		}

		/**
		 * @brief In-order visit of every element, `visit(element)`, with pending
		 *        tags composed down on the fly — const, never settles or copies
		 *        a node (an explicit stack, so long spines cannot overflow the
		 *        call stack).
		 * @complexity O(n)
		 */
		template<typename Visit>
			requires std::invocable<Visit &, T const &>
		void for_each(Visit visit) const {
			if (!pool_) return;

			struct frame {
				uint32 t;
				[[no_unique_address]] slot<F> acc; // composed tags of t's ancestors
			};

			std::vector<frame> stack;
			uint32 t = root_;
			slot<F> acc{};
			if constexpr (has_tags) acc = pool_->tag_identity;
			for (;;) {
				for (; t != 0; t = pool_->nodes[t].left) {
					stack.push_back({t, acc});
					if constexpr (has_tags) acc = pool_->compose(acc, pool_->nodes[t].tag);
				}
				if (stack.empty()) return;
				frame const top = stack.back();
				stack.pop_back();
				node const &nd = pool_->nodes[top.t];
				t = nd.right;
				if constexpr (has_tags) {
					acc = pool_->compose(top.acc, nd.tag);
					T const value = pool_->mapping(top.acc, nd.value);
					visit(value);
				} else {
					visit(nd.value);
				}
			}
		}

		/**
		 * @brief This version rebuilt perfectly balanced, pending tags settled,
		 *        in a fresh arena of its own — sheds every dead path a long
		 *        history left in the shared arena (the old arena is freed once
		 *        no version references it). The result shares nothing with this
		 *        version, so it cannot be `concat`ed with it.
		 * @pre `size()` not saturated
		 * @pre Each active algebra callable satisfies std::copy_constructible (constrained).
		 * @complexity O(n)
		 */
		[[nodiscard]] persistent_treap_sequence compacted() const
			requires((!has_op || std::copy_constructible<Op>) && (!has_tags || (std::copy_constructible<Mapping> && std::copy_constructible<Compose>)))
		{
			atl_assert(pool_ != nullptr);
			atl_assert(!is_saturated());
			auto fresh = [&] {
				if constexpr (has_tags)
					return std::make_shared<pool>(pool_->element_identity, pool_->tag_identity, pool_->op, pool_->mapping, pool_->compose);
				else if constexpr (has_op)
					return std::make_shared<pool>(pool_->element_identity, pool_->op);
				else
					return std::make_shared<pool>();
			}();
			std::vector<T> values;
			values.reserve(size());
			for_each([&](T const &v) { values.push_back(v); });
			fresh->nodes.reserve_extra(values.size());
			persistent_treap_sequence out(std::move(fresh), 0);
			auto it = values.begin();
			out.root_ = out.template build_tree<std::vector<T>>(std::move(it), values.size()).first;
			return out;
		}

		/**
		 * @brief Number of nodes in the shared arena — versions' total structural
		 * footprint (diagnostics / capacity planning).
		 */
		std::size_t arena_size() const noexcept { return pool_ ? pool_->nodes.size() : 0; }

		/** @brief Reserve arena headroom for @p ops further mutations. */
		void reserve(std::size_t ops) {
			std::size_t const levels = std::size_t(std::bit_width(uint64(size()) + 1));
			atl_assert(ops <= (std::numeric_limits<std::size_t>::max)() / 6 / (levels + 1));
			pool_->nodes.reserve_extra(ops * 6 * (levels + 1));
		}
	};

	/** @brief Deduce the container types from constructor arguments and their algebra. */
	template<std::ranges::sized_range R>
	persistent_treap_sequence(R &&) -> persistent_treap_sequence<std::ranges::range_value_t<R>>;
	/** @brief Narrowing-guarded, as in @ref atl::treap_sequence. */
	template<std::ranges::sized_range R, typename T, binary_operation<T> Op>
		requires requires(std::ranges::range_value_t<R> v) { T{std::move(v)}; }
	persistent_treap_sequence(R &&, T, Op) -> persistent_treap_sequence<T, Op>;
	/** @brief Deduce the tagged tier while rejecting narrowing range values. */
	template<std::ranges::sized_range R, typename T, typename F, binary_operation<T> Op, action<F, T> Mapping,
			 binary_operation<F> Compose>
		requires requires(std::ranges::range_value_t<R> v) { T{std::move(v)}; }
	persistent_treap_sequence(R &&, T, F, Op, Mapping, Compose) -> persistent_treap_sequence<T, Op, F, Mapping, Compose>;
}
