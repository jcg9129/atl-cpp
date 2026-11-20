#pragma once

/**
 * @file
 * @brief Path-copying heaps by rank-leftist, weight-biased or randomized meld,
 * with optional lazy shifts and branch/select and fresh-node reuse knobs.
 * Vector heapify allocates a temporary buffer and moves values twice. It takes
 * 11.3 / 11.2 ms to build 1e6 values versus in-arena heapify's 10.0 / 9.2;
 * chained class push_range(3) takes 32.5 / 32.4 versus 30.8 / 30.1.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-11-01,
 * -O2 / -O2 -march=native (also -g -DNDEBUG -std=c++23),
 * benchmarks/ds/persistent_heap.cpp; docs/persistent-heap.md has full tables.
 * Rank-leftist improves keep-all versions (748 / 748 ms versus randomized
 * 1014 / 1015) but loses on linear push/pop (124 / 125 versus 101 / 98.2).
 * These records are 2026-10-23 / 2026-10-24, with the same CPU and flags.
 * Persistent skew and pairing heaps are excluded because repeated expensive
 * operations on one version defeat their amortized bounds. On k-shortest random n=1e5 m=3e5 heap construction, class vector heapify
 * takes 19.5 / 20.2 ms versus in-arena 16.7 / 16.4; forest takes
 * 16.0 / 16.5 versus 13.0 / 13.2 (2026-11-01, same CPU/compiler/flags).
 * Complete forest-walk tuning with K=3e5 takes 174 / 179 ms by rank-leftist,
 * 184 / 183 by weight-biased and 170 / 168 by randomized meld with branch/reuse
 * (2026-10-23, same CPU/compiler/flags). Tests certify all variants by a multiset per version.
 */

#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/persistent_arena.hpp>
#include <atl/ds/meldable_heap_forest.hpp>
#include <atl/ds/persistent_heap.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::experimental {
	/** @brief The meld algorithm of a @ref atl::experimental::persistent_heap_by_meld_policy. */
	enum class persistent_heap_algorithm : uint8 {
		/** @brief Leftist by rank (null-path length, `uint8`): meld down both right spines, then a fix-up pass. */
		rank_leftist,
		/** @brief Leftist by subtree size (`uint32`): the merged sizes are known up front, so one top-down pass. */
		weight_biased,
		/** @brief Gambin-Malinowski: a coin flip picks the child to meld into; no balance field. */
		randomized,
	};

	namespace detail {

		/** @brief The elements of `values` in a buffer, moved or copied by the library's range-forwarding rule. */
		template<std::semiregular T, std::ranges::input_range R>
			requires std::convertible_to<atl::detail::forwarded_element_t<R>, T>
		constexpr std::vector<T> collect_heap_values(R &&values) {
			std::vector<T> buffer;
			if constexpr (std::ranges::sized_range<R>) buffer.reserve(std::ranges::size(values));
			for (auto &&v : values) buffer.push_back(static_cast<T>(atl::detail::forward_element<R>(std::forward<decltype(v)>(v))));
			return buffer;
		}

		/** @brief The balance field of a node: a rank, a size, or nothing. */
		template<persistent_heap_algorithm Algorithm>
		using persistent_heap_meta_t = std::conditional_t<
			Algorithm == persistent_heap_algorithm::rank_leftist, uint8,
			std::conditional_t<Algorithm == persistent_heap_algorithm::weight_biased, uint32, atl::detail::no_heap_link>>;

		/**
		 * @brief The node store of every contender: an append-only @ref
		 *        atl::detail::persistent_arena (id 0 the null sentinel, balance
		 *        field 0) and the order, plus the meld, pop, add-to-all and
		 *        heapify that build new roots out of old ones by path copying.
		 *
		 * Every operation records `fresh`, the arena size at its start: a node
		 * with an id at or past it was made by this operation, is shared with
		 * no version, and is changed in place instead of copied (the pushed
		 * singleton, a child copied by a push-down).
		 *
		 * A pending shift on a node covers its descendants (its own value is
		 * already shifted, so `top` reads a stored value). Copying a node with a
		 * pending shift onto a path hands the shift to both children, which
		 * copies them too.
		 */
		template<std::semiregular T, typename Compare, persistent_heap_algorithm Algorithm, typename Shift,
				 typename Mapping, typename Compose, bool IsBranchLight, bool ReusesFreshNodes>
			requires meldable_heap_parameters<T, Compare, Shift, Mapping, Compose>
		struct persistent_heap_core {
			using tag_type = atl::detail::heap_tag_t<Shift>;
			using meta_type = persistent_heap_meta_t<Algorithm>;
			static constexpr bool has_shift = !std::is_void_v<Shift>;

			/** @brief A heap node: its value, children `child[0]` (left) and `child[1]` (right), balance field, pending shift. */
			struct node {
				T value;
				uint32 child[2];
				[[no_unique_address]] meta_type meta;
				[[no_unique_address]] tag_type tag;
			};

			atl::detail::persistent_arena<node> nodes;
			[[no_unique_address]] Compare comp;
			[[no_unique_address]] Mapping mapping;
			[[no_unique_address]] Compose compose;
			[[no_unique_address]] tag_type identity;
			// a fixed seed, unlike the library pool's per-process one, so race runs replay the same coins
			uint64 random_state = 0x9e3779b97f4a7c15ULL;
			uint64 random_bits = 0;
			uint32 random_left = 0;

			/** @brief An empty pool: the null node, the order and, with a lazy add, its identity and callables. */
			explicit persistent_heap_core(Compare order = {}, tag_type shift_identity = {}, Mapping map = {},
										  Compose comp_shift = {})
				: nodes(node{T{}, {0, 0}, meta_type{}, shift_identity}),
				  comp(std::move(order)),
				  mapping(std::move(map)),
				  compose(std::move(comp_shift)),
				  identity(std::move(shift_identity)) {}

			/** @brief One fair coin (splitmix64 output, consumed a bit at a time). */
			uint32 coin() {
				if (random_left == 0) {
					random_bits = atl::detail::splitmix64(random_state += 0x9e3779b97f4a7c15ULL);
					random_left = 64;
				}
				--random_left;
				uint32 const bit = uint32(random_bits & 1);
				random_bits >>= 1;
				return bit;
			}

			/** @brief A fresh leaf holding `value`. */
			uint32 leaf(T &&value) {
				meta_type meta{};
				if constexpr (Algorithm != persistent_heap_algorithm::randomized) meta = 1;
				return nodes.make(node{std::move(value), {0, 0}, meta, identity});
			}

			/** @brief Node `x` with `shift` applied to its value and its subtree: in place when fresh, else a copy. */
			uint32 shifted(uint32 x, tag_type const &shift, uint32 fresh)
				requires has_shift
			{
				if (!ReusesFreshNodes || x < fresh) x = nodes.make(nodes[x]);
				node &n = nodes[x];
				n.value = mapping(shift, std::move(n.value));
				n.tag = compose(shift, n.tag);
				return x;
			}

			/** @brief Node `x` made writable for this operation (copied unless fresh), its pending shift handed down. */
			uint32 own(uint32 x, uint32 fresh) {
				if (!ReusesFreshNodes || x < fresh) x = nodes.make(nodes[x]);
				if constexpr (has_shift) {
					if (!atl::detail::is_identity_shift(nodes[x].tag, identity)) {
						tag_type const shift = nodes[x].tag;
						for (uint32 side = 0; side < 2; ++side)
							if (uint32 const c = nodes[x].child[side]; c != 0) {
								uint32 const moved = shifted(c, shift, fresh);
								nodes[x].child[side] = moved;
							}
						nodes[x].tag = identity;
					}
				}
				return x;
			}

			/** @brief Order `(first, second)` so that `first` holds the comp-larger root (the tuning knob: a select or a branch). */
			void order(uint32 &first, uint32 &second) {
				if constexpr (IsBranchLight) {
					bool const is_swapped = comp(nodes[first].value, nodes[second].value);
					uint32 const better = is_swapped ? second : first;
					second = is_swapped ? first : second;
					first = better;
				} else {
					if (comp(nodes[first].value, nodes[second].value)) std::swap(first, second);
				}
			}

			/** @brief The meld of the heaps rooted at `a` and `b` (either may be 0), sharing every untouched subtree. */
			uint32 meld(uint32 a, uint32 b, uint32 fresh) {
				if (a == 0) return b;
				if (b == 0) return a;
				if constexpr (Algorithm == persistent_heap_algorithm::rank_leftist) {
					// down both right spines, the copies linked upwards through
					// their (now free) right links; ranks are fixed on the way up
					uint32 above = 0;
					while (a != 0 && b != 0) {
						order(a, b);
						uint32 const c = own(a, fresh);
						node &n = nodes[c];
						a = n.child[1];
						n.child[1] = above;
						above = c;
					}
					uint32 merged = a | b;
					while (above != 0) {
						node &n = nodes[above];
						uint32 const parent = n.child[1];
						uint8 const left_rank = nodes[n.child[0]].meta, merged_rank = nodes[merged].meta;
						if (left_rank < merged_rank) {
							n.child[1] = n.child[0];
							n.child[0] = merged;
							n.meta = uint8(left_rank + 1);
						} else {
							n.child[1] = merged;
							n.meta = uint8(merged_rank + 1);
						}
						merged = above;
						above = parent;
					}
					return merged;
				} else {
					uint32 root = 0, parent = 0, side = 0;
					while (a != 0 && b != 0) {
						order(a, b);
						uint32 total = 0;
						if constexpr (Algorithm == persistent_heap_algorithm::weight_biased)
							total = nodes[a].meta + nodes[b].meta;
						uint32 const c = own(a, fresh);
						if (parent == 0)
							root = c;
						else
							nodes[parent].child[side] = c;
						node &n = nodes[c];
						if constexpr (Algorithm == persistent_heap_algorithm::weight_biased) {
							// the meld of the right child with b weighs size(right) + size(b);
							// it goes left when it outweighs the left child
							n.meta = total;
							uint32 const left = n.child[0], right = n.child[1];
							if (nodes[left].meta < nodes[right].meta + nodes[b].meta) {
								n.child[1] = left;
								side = 0;
							} else
								side = 1;
							a = right;
						} else {
							side = coin();
							a = n.child[side];
						}
						parent = c;
					}
					nodes[parent].child[side] = a | b;
					return root;
				}
			}

			/** @brief The heap rooted at `root` without its top. @pre `root != 0` */
			uint32 pop(uint32 root, uint32 fresh) {
				uint32 left = nodes[root].child[0], right = nodes[root].child[1];
				if constexpr (has_shift) {
					if (!atl::detail::is_identity_shift(nodes[root].tag, identity)) {
						tag_type const shift = nodes[root].tag;
						if (left != 0) left = shifted(left, shift, fresh);
						if (right != 0) right = shifted(right, shift, fresh);
					}
				}
				return meld(left, right, fresh);
			}

			/**
			 * @brief A heap of `values` (consumed) in O(n): `std::ranges::make_heap`,
			 *        then the array laid out as its implicit complete tree, which is
			 *        leftist by rank and by size; returns its root (0 when empty).
			 */
			uint32 heapify(std::vector<T> &values) {
				std::size_t const count = values.size();
				if (count == 0) return 0;
				atl_assert(uint64(count) + nodes.size() < (std::numeric_limits<uint32>::max)());
				std::ranges::make_heap(values, std::ref(comp));
				uint32 const k = uint32(count), base = uint32(nodes.size());
				nodes.reserve_extra(count);
				for (uint32 i = 0; i < k; ++i) {
					uint32 const l = 2 * uint64(i) + 1 < k ? base + 2 * i + 1 : 0;
					uint32 const r = 2 * uint64(i) + 2 < k ? base + 2 * i + 2 : 0;
					nodes.make(node{std::move(values[i]), {l, r}, meta_type{}, identity});
				}
				if constexpr (Algorithm != persistent_heap_algorithm::randomized)
					for (uint32 i = k; i > 0; --i) {
						node &n = nodes[base + i - 1];
						if constexpr (Algorithm == persistent_heap_algorithm::rank_leftist)
							n.meta = uint8(nodes[n.child[1]].meta + 1);
						else
							n.meta = nodes[n.child[0]].meta + nodes[n.child[1]].meta + 1;
					}
				return base;
			}

			/** @brief The heap rooted at `root` with `shift` applied to every element. */
			uint32 add_to_all(uint32 root, tag_type const &shift)
				requires has_shift
			{
				return root == 0 ? 0 : shifted(root, shift, uint32(nodes.size()));
			}
		};
	}

	/**
	 * @brief A fully persistent meldable heap (the class form): every version
	 *        is a self-contained object holding a handle on the node pool it
	 *        shares with the versions it was derived from.
	 *
	 * @tparam T, Compare, Shift, Mapping, Compose As for @ref atl::meldable_heap_forest;
	 *         `top` is the comp-largest element.
	 * @tparam Algorithm The meld (@ref persistent_heap_algorithm).
	 * @tparam IsBranchLight Tuning knob: `true` orders the two roots of a meld
	 *         step by a select instead of a branch around `std::swap`.
	 * @tparam ReusesFreshNodes Tuning knob: `true` changes a node made by the
	 *         running operation in place; `false` copies it like any other.

	 */
	template<std::semiregular T, typename Compare = std::less<>,
			 persistent_heap_algorithm Algorithm = persistent_heap_algorithm::rank_leftist, typename Shift = void,
			 typename Mapping = shift_by_addition, typename Compose = compose_by_addition, bool IsBranchLight = false,
			 bool ReusesFreshNodes = true>
		requires meldable_heap_parameters<T, Compare, Shift, Mapping, Compose>
	class persistent_heap_by_meld_policy {
		using core = detail::persistent_heap_core<T, Compare, Algorithm, Shift, Mapping, Compose, IsBranchLight,
												  ReusesFreshNodes>;
		using tag_type = typename core::tag_type;
		static constexpr bool has_shift = core::has_shift;

		std::shared_ptr<core> pool_;
		uint32 root_ = 0;
		uint32 size_ = 0;

		persistent_heap_by_meld_policy(std::shared_ptr<core> shared, uint32 root, uint32 size)
			: pool_(std::move(shared)), root_(root), size_(size) {}

		/** @brief `push` with the pool passed by value (as @ref atl::persistent_heap). */
		persistent_heap_by_meld_policy push_in(std::shared_ptr<core> shared, T value) const {
			atl_assert(size_ + 1 < (std::numeric_limits<uint32>::max)());
			uint32 const fresh = uint32(shared->nodes.size());
			uint32 const x = shared->leaf(std::move(value));
			uint32 const root = shared->meld(root_, x, fresh);
			return {std::move(shared), root, size_ + 1};
		}

		/** @brief `pop` with the pool passed by value. */
		persistent_heap_by_meld_policy pop_in(std::shared_ptr<core> shared) const {
			atl_assert(size_ > 0);
			uint32 const root = shared->pop(root_, uint32(shared->nodes.size()));
			return {std::move(shared), root, size_ - 1};
		}

		/** @brief `meld` with the pool passed by value (the other version's root and size given). */
		persistent_heap_by_meld_policy meld_in(std::shared_ptr<core> shared, uint32 other_root, uint32 other_size) const {
			atl_assert(uint64(size_) + other_size < (std::numeric_limits<uint32>::max)());
			uint32 const root = shared->meld(root_, other_root, uint32(shared->nodes.size()));
			return {std::move(shared), root, size_ + other_size};
		}

		/** @brief `push_range` with the pool passed by value. */
		template<std::ranges::input_range R>
			requires std::convertible_to<atl::detail::forwarded_element_t<R>, T>
		persistent_heap_by_meld_policy push_range_in(std::shared_ptr<core> shared, R &&values) const {
			uint32 const fresh = uint32(shared->nodes.size());
			uint32 built, count;
			std::vector<T> buffer = detail::collect_heap_values<T>(std::forward<R>(values));
			atl_assert(buffer.size() < (std::numeric_limits<uint32>::max)());
			count = uint32(buffer.size());
			built = shared->heapify(buffer);

			atl_assert(uint64(size_) + count < (std::numeric_limits<uint32>::max)());
			uint32 const root = shared->meld(root_, built, fresh);
			return {std::move(shared), root, size_ + count};
		}

	public:
		using value_type = T;

		/**
		 * @brief A read-only position in a version's heap tree (empty past a
		 *        leaf), as @ref atl::persistent_heap::node_view; only without a
		 *        lazy add, whose stored values below the root are not real.
		 */
		class node_view {
			core const *pool_ = nullptr;
			uint32 id_ = 0;

			friend class persistent_heap_by_meld_policy;

			node_view(core const *pool, uint32 id) : pool_(pool), id_(id) {}

		public:
			/** @brief An empty view. */
			node_view() = default;

			/** @brief Whether there is no node here. */
			bool empty() const noexcept { return id_ == 0; }

			/** @brief The node's element. @pre `!empty()` (asserted) */
			T const &value() const {
				atl_assert(id_ != 0);
				return pool_->nodes[id_].value;
			}

			/** @brief The left child (empty when none). @pre `!empty()` (asserted) */
			node_view left() const {
				atl_assert(id_ != 0);
				return {pool_, pool_->nodes[id_].child[0]};
			}

			/** @brief The right child (empty when none). @pre `!empty()` (asserted) */
			node_view right() const {
				atl_assert(id_ != 0);
				return {pool_, pool_->nodes[id_].child[1]};
			}
		};

		/** @brief An empty heap with a new pool. */
		persistent_heap_by_meld_policy() : pool_(std::make_shared<core>()) {}

		/** @brief An empty heap ordered by `comp`, with a new pool. */
		explicit persistent_heap_by_meld_policy(Compare comp) : pool_(std::make_shared<core>(std::move(comp))) {}

		/** @brief An empty heap with the lazy add `(shift_identity, mapping, compose)`, with a new pool. */
		persistent_heap_by_meld_policy(Compare comp, tag_type shift_identity, Mapping mapping = {}, Compose compose = {})
			requires has_shift
			: pool_(std::make_shared<core>(std::move(comp), std::move(shift_identity), std::move(mapping),
										   std::move(compose))) {}

		/** @brief A heap of `values`, with a new pool, in O(n). */
		template<std::ranges::input_range R>
			requires std::convertible_to<atl::detail::forwarded_element_t<R>, T>
		explicit persistent_heap_by_meld_policy(R &&values, Compare comp = {})
			: pool_(std::make_shared<core>(std::move(comp))) {
			std::vector<T> buffer = detail::collect_heap_values<T>(std::forward<R>(values));
			atl_assert(buffer.size() < (std::numeric_limits<uint32>::max)());
			size_ = uint32(buffer.size());
			root_ = pool_->heapify(buffer);
		}

		/** @brief The number of elements. */
		std::size_t size() const noexcept { return size_; }

		/** @brief Whether the heap holds no element. */
		bool empty() const noexcept { return size_ == 0; }

		/** @brief The comp-largest element, by reference into the pool. @pre `!empty()` (asserted) */
		T const &top() const {
			atl_assert(size_ > 0);
			return pool_->nodes[root_].value;
		}

		/** @brief The root of the heap tree. */
		node_view root() const
			requires(!has_shift)
		{
			return {pool_.get(), root_};
		}

		/** @brief The number of nodes in the shared pool, the null node included. */
		std::size_t arena_size() const noexcept { return pool_->nodes.size(); }

		/**
		 * @brief Pool room for `operations` further operations (an operation
		 *        count, unlike @ref atl::experimental::persistent_heap_forest_by_meld_policy::reserve) on
		 *        versions of about this size.
		 */
		void reserve(std::size_t operations) {
			std::size_t const per_operation = 2 * std::size_t(std::bit_width(size_ + uint64(1))) + 1;
			atl_assert(operations <= (std::numeric_limits<uint32>::max)() / per_operation);
			pool_->nodes.reserve_extra(operations * per_operation);
		}

		/** @brief The heap with `value` added. */
		[[nodiscard]] persistent_heap_by_meld_policy push(T value) const & { return push_in(pool_, std::move(value)); }

		/** @brief `push` consuming this version (the pool is moved). */
		[[nodiscard]] persistent_heap_by_meld_policy push(T value) && { return push_in(std::move(pool_), std::move(value)); }

		/** @brief `push(T(args...))`. */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] persistent_heap_by_meld_policy emplace(Args &&...args) const & {
			return push_in(pool_, T(std::forward<Args>(args)...));
		}

		/** @brief `emplace` consuming this version. */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] persistent_heap_by_meld_policy emplace(Args &&...args) && {
			return push_in(std::move(pool_), T(std::forward<Args>(args)...));
		}

		/** @brief The heap without its top. @pre `!empty()` (asserted) */
		[[nodiscard]] persistent_heap_by_meld_policy pop() const & { return pop_in(pool_); }

		/** @brief `pop` consuming this version. */
		[[nodiscard]] persistent_heap_by_meld_policy pop() && { return pop_in(std::move(pool_)); }

		/** @brief The elements of both versions. @pre both share one pool (asserted) */
		[[nodiscard]] persistent_heap_by_meld_policy meld(persistent_heap_by_meld_policy const &other) const & {
			atl_assert(pool_ == other.pool_);
			return meld_in(pool_, other.root_, other.size_);
		}

		/** @brief `meld` consuming this version. */
		[[nodiscard]] persistent_heap_by_meld_policy meld(persistent_heap_by_meld_policy const &other) && {
			atl_assert(pool_ == other.pool_);
			return meld_in(std::move(pool_), other.root_, other.size_);
		}

		/** @brief Every element of `values` added: one O(k) array-built heap, then one meld. */
		template<std::ranges::input_range R>
			requires std::convertible_to<atl::detail::forwarded_element_t<R>, T>
		[[nodiscard]] persistent_heap_by_meld_policy push_range(R &&values) const & {
			return push_range_in(pool_, std::forward<R>(values));
		}

		/** @brief `push_range` consuming this version. */
		template<std::ranges::input_range R>
			requires std::convertible_to<atl::detail::forwarded_element_t<R>, T>
		[[nodiscard]] persistent_heap_by_meld_policy push_range(R &&values) && {
			return push_range_in(std::move(pool_), std::forward<R>(values));
		}

		/** @brief `mapping(shift, x)` for every element `x`: one copied root (the lazy-add variant only). */
		[[nodiscard]] persistent_heap_by_meld_policy add_to_all(tag_type const &shift) const &
			requires has_shift
		{
			return {pool_, pool_->add_to_all(root_, shift), size_};
		}

		/** @brief `add_to_all` consuming this version. */
		[[nodiscard]] persistent_heap_by_meld_policy add_to_all(tag_type const &shift) &&
			requires has_shift
		{
			uint32 const root = pool_->add_to_all(root_, shift);
			return {std::move(pool_), root, size_};
		}
	};

	/** @brief A heap of a range holds the range's value type, ordered by `std::less<>`. */
	template<std::ranges::input_range R>
	persistent_heap_by_meld_policy(R &&) -> persistent_heap_by_meld_policy<std::ranges::range_value_t<R>>;

	/** @brief A heap of a range holds the range's value type, ordered by the given `Compare`. */
	template<std::ranges::input_range R, typename Compare>
		requires std::strict_weak_order<Compare &, std::ranges::range_value_t<R> const &,
										std::ranges::range_value_t<R> const &>
	persistent_heap_by_meld_policy(R &&, Compare) -> persistent_heap_by_meld_policy<std::ranges::range_value_t<R>, Compare>;

	/**
	 * @brief The forest form: one node pool, heap versions named by `uint32`
	 *        ids (a version is its root node; 0 is the empty heap). Every
	 *        subtree of a version is itself a version, so `left` / `right`
	 *        walk the heap tree by ids.
	 *
	 * No element count per version (it would need a record per id); `size`
	 * exists for the weight-biased meld, whose nodes store their subtree sizes.
	 *
	 * @tparam T, Compare, Algorithm, Shift, Mapping, Compose, IsBranchLight, ReusesFreshNodes
	 *         As for @ref atl::experimental::persistent_heap_by_meld_policy.
	 */
	template<std::semiregular T, typename Compare = std::less<>,
			 persistent_heap_algorithm Algorithm = persistent_heap_algorithm::rank_leftist, typename Shift = void,
			 typename Mapping = shift_by_addition, typename Compose = compose_by_addition, bool IsBranchLight = false,
			 bool ReusesFreshNodes = true>
		requires meldable_heap_parameters<T, Compare, Shift, Mapping, Compose>
	class persistent_heap_forest_by_meld_policy {
		using core = detail::persistent_heap_core<T, Compare, Algorithm, Shift, Mapping, Compose, IsBranchLight,
												  ReusesFreshNodes>;
		using tag_type = typename core::tag_type;
		static constexpr bool has_shift = core::has_shift;

		core pool_;

		/** @brief The arena size: nodes at or past it are made by the running operation. */
		uint32 fresh() const { return uint32(pool_.nodes.size()); }

	public:
		using value_type = T;

		/** @brief The id of the empty heap. */
		static constexpr uint32 empty_heap = 0;

		/** @brief An empty forest ordered by `comp`. */
		explicit persistent_heap_forest_by_meld_policy(Compare comp = {}) : pool_(std::move(comp)) {}

		/** @brief Whether version `heap` holds no element. */
		bool empty(uint32 heap) const noexcept { return heap == 0; }

		/** @brief The element count of version `heap` (its root's subtree size). */
		std::size_t size(uint32 heap) const
			requires(Algorithm == persistent_heap_algorithm::weight_biased)
		{
			return pool_.nodes[heap].meta;
		}

		/** @brief The comp-largest element of version `heap`. @pre nonempty (asserted) */
		T const &top(uint32 heap) const {
			atl_assert(heap != 0 && heap < pool_.nodes.size());
			return pool_.nodes[heap].value;
		}

		/** @brief The version formed by the left subtree of `heap`'s root. @pre nonempty (asserted) */
		uint32 left(uint32 heap) const
			requires(!has_shift)
		{
			atl_assert(heap != 0 && heap < pool_.nodes.size());
			return pool_.nodes[heap].child[0];
		}

		/** @brief The version formed by the right subtree of `heap`'s root. @pre nonempty (asserted) */
		uint32 right(uint32 heap) const
			requires(!has_shift)
		{
			atl_assert(heap != 0 && heap < pool_.nodes.size());
			return pool_.nodes[heap].child[1];
		}

		/** @brief The number of nodes in the pool, the null node included. */
		std::size_t arena_size() const noexcept { return pool_.nodes.size(); }

		/** @brief Pool room for `nodes` further nodes (a node count, unlike @ref atl::experimental::persistent_heap_by_meld_policy::reserve). */
		void reserve(std::size_t nodes) { pool_.nodes.reserve_extra(nodes); }

		/** @brief Version `heap` with `value` added. */
		[[nodiscard]] uint32 push(uint32 heap, T value) {
			atl_assert(heap < pool_.nodes.size());
			uint32 const f = fresh();
			uint32 const x = pool_.leaf(std::move(value));
			return pool_.meld(heap, x, f);
		}

		/** @brief `push(heap, T(args...))`. */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] uint32 emplace(uint32 heap, Args &&...args) {
			return push(heap, T(std::forward<Args>(args)...));
		}

		/** @brief Version `heap` without its top. @pre nonempty (asserted) */
		[[nodiscard]] uint32 pop(uint32 heap) {
			atl_assert(heap != 0 && heap < pool_.nodes.size());
			return pool_.pop(heap, fresh());
		}

		/** @brief The version holding the elements of `a` and `b`. */
		[[nodiscard]] uint32 meld(uint32 a, uint32 b) {
			atl_assert(a < pool_.nodes.size() && b < pool_.nodes.size());
			return pool_.meld(a, b, fresh());
		}

		/** @brief Version `heap` with every element of `values` added: one O(k) array-built heap, then one meld. */
		template<std::ranges::input_range R>
			requires std::convertible_to<atl::detail::forwarded_element_t<R>, T>
		[[nodiscard]] uint32 push_range(uint32 heap, R &&values) {
			atl_assert(heap < pool_.nodes.size());
			uint32 const f = fresh();
			uint32 built;
			std::vector<T> buffer = detail::collect_heap_values<T>(std::forward<R>(values));
			built = pool_.heapify(buffer);

			return pool_.meld(heap, built, f);
		}

		/** @brief `mapping(shift, x)` for every element of version `heap` (the lazy-add variant only). */
		[[nodiscard]] uint32 add_to_all(uint32 heap, tag_type const &shift)
			requires has_shift
		{
			return pool_.add_to_all(heap, shift);
		}
	};
}
