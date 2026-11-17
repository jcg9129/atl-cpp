#pragma once

/**
 * @file
 * @brief Contenders for @ref atl::meldable_heap_forest.
 *
 * - @ref atl::experimental::meldable_heap_forest_by_rank_leftist — ranks, O(log n) worst-case meld down the two
 *   right spines (iterative, pointer reversal). Lost: 1.3-3.5x slower than
 *   the library pairing heap. Kept as the heap that persists naturally.
 * - @ref atl::experimental::meldable_heap_forest_by_skew — the same walk with no ranks, amortized O(log n),
 *   iterative only (a spine can be O(n)). Lost: 1.1-4.2x slower on the
 *   tree melds at 1e6, tied on the random-tree slope pattern (259 vs 256
 *   ms); wins only the single-heap push / pop mix (14.5 vs 17.2 ms).
 * - @ref atl::experimental::meldable_heap_forest_by_pairing — the library algorithm with tuning knobs (link
 *   by select, no free list, no identity skip); its defaults are the
 *   library class without handles.
 * - @ref atl::experimental::meldable_heap_forest_by_split_pairing — pairing over split arrays (values,
 *   tags, links). Lost the tuning: 15.9 / 15.1 vs 12.6 / 13.2 ms (2e5 tree).
 * - @ref atl::experimental::meldable_heap_forest_by_binary — the baseline, `std::vector` binary heaps
 *   melded smaller-into-larger. Lost: 3.1-4.1x slower on the tree melds at
 *   1e6.
 *
 * Race verdict (benchmarks/ds/meldable_heap_forest.cpp, measurements of 2026-10-16, Xeon Platinum 8280, ms, plain -O2 / -march=native;
 * int64; tree = bottom-up melds, push 3 / pop 2 per vertex; slope = min-heaps
 * with a lazy add; mix = 1e6 random operations):
 *
 * | workload | pairing (library) | skew | leftist | binary |
 * |------------------------------|-------------------|---------------|-------------|-------------|
 * | tree path 1e6 | 69.3 / 70.3 | 93.1 / 91.9 | 127 / 128 | 220 / 221 |
 * | tree star 1e6 | 35.8 / 36.2 | 152 / 151 | 125 / 121 | 147 / 148 |
 * | tree caterpillar 1e6 | 58.9 / 59.9 | 82.9 / 81.1 | 91.3 / 92.3 | 185 / 185 |
 * | tree random 1e6 | 117 / 119 | 131 / 134 | 154 / 157 | 408 / 406 |
 * | slope path 1e6 | 73.1 / 75.2 | 97.6 / 103 | 131 / 132 | 127 / 128 |
 * | slope random 1e6 | 256 / 255 | 259 / 262 | 380 / 379 | 431 / 435 |
 * | mix 1 heap, push/pop | 17.2 / 17.5 | 14.5 / 15.2 | 21.6 / 23.3 | 25.6 / 25.4 |
 * | mix 1e5 heaps, 40/20/40 | 25.0 / 25.6 | 26.2 / 26.3 | 39.3 / 39.8 | 53.8 / 52.6 |
 *
 * The full table (2e5 sizes, caterpillar slope, 1000-heap mix) and the
 * tuning table are in docs/meldable-heap-forest.md.
 */

#include <atl/ds/meldable_heap_forest.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::experimental {
	/**
	 * @brief Leftist heaps on one node pool: each node stores its rank (the
	 *        length of its rightmost path), every left child ranks at least
	 *        its sibling, so the right spine has O(log n) nodes and a meld
	 *        walks only the two right spines.
	 *
	 * Kept for a second reason besides its race numbers: it is the
	 * contender that persists naturally. A meld changes only the O(log n)
	 * nodes on the right spines, so path copying gives a persistent leftist
	 * heap at O(log n) new nodes per operation (pairing and skew heaps have
	 * no such bound per operation). A persistent leftist heap is planned as
	 * a separate component.
	 *
	 * @tparam T, Compare, Shift, Mapping, Compose As for @ref atl::meldable_heap_forest.
	 * @tparam Rank The rank's storage type; a rank never exceeds
	 *         `log2(n + 1) <= 32`, so `uint8` fits.
	 * @tparam IsRecursive `true` melds recursively down the right spines;
	 *         `false` walks them in a loop, reversing the right links on the
	 *         way down (no stack) and restoring them, with the ranks, on the
	 *         way up.
	 */
	template<typename T, typename Compare = std::less<>, typename Shift = void, typename Mapping = shift_by_addition,
			 typename Compose = compose_by_addition, std::unsigned_integral Rank = uint8,
			 bool IsRecursive = false>
		requires meldable_heap_parameters<T, Compare, Shift, Mapping, Compose>
	class meldable_heap_forest_by_rank_leftist : public atl::detail::heap_directory {
		using tag_type = atl::detail::heap_tag_t<Shift>;
		static constexpr bool has_shift = !std::is_void_v<Shift>;

		struct node {
			T value;
			[[no_unique_address]] tag_type tag{};
			uint32 left = 0, right = 0;
			Rank rank = 1;
		};

		std::vector<node> nodes_;
		uint32 free_ = 0;
		[[no_unique_address]] Compare comp_;
		[[no_unique_address]] Mapping mapping_;
		[[no_unique_address]] Compose compose_;
		[[no_unique_address]] tag_type shift_identity_{};

		/** @brief Node `x` (1-based; 0 means none). */
		constexpr node &at(uint32 x) { return nodes_[x - 1]; }

		/** @brief Node `x` (1-based; 0 means none). */
		constexpr node const &at(uint32 x) const { return nodes_[x - 1]; }

		/** @brief The rank of `x`, 0 for none. */
		constexpr Rank rank(uint32 x) { return x == 0 ? Rank(0) : at(x).rank; }

		/** @brief Shift `x`'s own value and add the shift to what it owes its descendants. */
		constexpr void apply(uint32 x, tag_type const &shift) {
			node &n = at(x);
			n.value = mapping_(shift, std::move(n.value));
			n.tag = compose_(shift, n.tag);
		}

		/** @brief Hand `x`'s pending add (which covers its descendants, not `x`) to its children. */
		constexpr void push_down(uint32 x) {
			if constexpr (has_shift) {
				node &n = at(x);
				if (atl::detail::is_identity_shift(n.tag, shift_identity_)) return;
				if (n.left != 0) apply(n.left, n.tag);
				if (n.right != 0) apply(n.right, n.tag);
				n.tag = shift_identity_;
			}
		}

		/** @brief A fresh node holding `T(args...)`, from the free list when it has one. */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		constexpr uint32 allocate(Args &&...args) {
			if (free_ != 0) {
				uint32 const x = free_;
				node &n = at(x);
				free_ = n.left;
				n.value = T(std::forward<Args>(args)...);
				n.tag = shift_identity_;
				n.left = n.right = 0;
				n.rank = 1;
				return x;
			}
			atl_assert(nodes_.size() + 1 < (std::numeric_limits<uint32>::max)());
			nodes_.push_back(node{T(std::forward<Args>(args)...), shift_identity_});
			return uint32(nodes_.size());
		}

		/** @brief Meld the heaps rooted at `a` and `b` (either may be 0); returns the new root. */
		constexpr uint32 merge(uint32 a, uint32 b) {
			if (a == 0) return b;
			if (b == 0) return a;
			if constexpr (IsRecursive) {
				if (comp_(at(a).value, at(b).value)) std::swap(a, b);
				push_down(a);
				uint32 const merged = merge(at(a).right, b);
				node &n = at(a);
				Rank const left_rank = rank(n.left), merged_rank = rank(merged);
				if (left_rank < merged_rank) {
					n.right = n.left;
					n.left = merged;
					n.rank = Rank(left_rank + 1);
				} else {
					n.right = merged;
					n.rank = Rank(merged_rank + 1);
				}
				return a;
			} else {
				uint32 above = 0; // the path so far, linked upwards through `right`
				while (a != 0 && b != 0) {
					if (comp_(at(a).value, at(b).value)) std::swap(a, b);
					push_down(a);
					uint32 const next = at(a).right;
					at(a).right = above;
					above = a;
					a = next;
				}
				uint32 merged = a != 0 ? a : b;
				while (above != 0) {
					node &n = at(above);
					uint32 const parent = n.right;
					Rank const left_rank = rank(n.left), merged_rank = rank(merged);
					if (left_rank < merged_rank) {
						n.right = n.left;
						n.left = merged;
						n.rank = Rank(left_rank + 1);
					} else {
						n.right = merged;
						n.rank = Rank(merged_rank + 1);
					}
					merged = above;
					above = parent;
				}
				return merged;
			}
		}

	public:
		using value_type = T;
		using shift_type = Shift;

		/**
		 * @brief `heap_count` empty heaps, ids `[0, heap_count)`; with a lazy add,
		 *        its identity is `Shift{}` and the callables are value-initialized.
		 */
		constexpr explicit meldable_heap_forest_by_rank_leftist(std::size_t heap_count = 0, Compare comp = {})
			requires(!has_shift
					 || (std::default_initializable<Shift> && std::default_initializable<Mapping>
						 && std::default_initializable<Compose>))
			: heap_directory(heap_count), comp_(std::move(comp)) {}

		/** @brief `heap_count` empty heaps with the lazy add `(shift_identity, mapping, compose)`. */
		constexpr meldable_heap_forest_by_rank_leftist(std::size_t heap_count, Compare comp, tag_type shift_identity, Mapping mapping = {},
													   Compose compose = {})
			requires has_shift
			: heap_directory(heap_count),
			  comp_(std::move(comp)),
			  mapping_(std::move(mapping)),
			  compose_(std::move(compose)),
			  shift_identity_(std::move(shift_identity)) {}

		/** @brief Room for `node_count` live elements without reallocating the pool. */
		constexpr void reserve(std::size_t node_count) { nodes_.reserve(node_count); }

		/** @brief The comp-largest element of heap `heap`. @pre the heap is not empty (asserted) */
		constexpr T const &top(std::size_t heap) const {
			atl_assert(record(checked_heap_id(heap)).size != 0);
			return at(record(checked_heap_id(heap)).root).value;
		}

		/** @brief Insert `T(args...)` into heap `heap`; returns its node id (no promote / erase here). @complexity O(log n) */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		constexpr uint32 emplace(std::size_t heap, Args &&...args) {
			atl::detail::heap_record &h = record(checked_heap_id(heap));
			uint32 const x = allocate(std::forward<Args>(args)...);
			h.root = merge(h.root, x);
			++h.size;
			return x;
		}

		/** @brief Insert `value` into heap `heap`. @complexity O(log n) */
		constexpr uint32 push(std::size_t heap, T value) { return emplace(heap, std::move(value)); }

		/** @brief Remove and return the top of heap `heap`. @pre not empty (asserted) @complexity O(log n) */
		constexpr T pop(std::size_t heap) {
			atl::detail::heap_record &h = record(checked_heap_id(heap));
			atl_assert(h.size != 0);
			uint32 const x = h.root;
			push_down(x);
			T top_value = std::move(at(x).value);
			h.root = merge(at(x).left, at(x).right);
			--h.size;
			at(x).left = free_;
			free_ = x;
			return top_value;
		}

		/** @brief Move every element of `from` into `into`; `from` is left empty (a no-op when they are the same heap). */
		constexpr void meld(std::size_t into, std::size_t from) {
			if (into == from) return;
			atl::detail::heap_record &a = record(checked_heap_id(into));
			atl::detail::heap_record &b = record(checked_heap_id(from));
			a.root = merge(a.root, b.root);
			a.size += b.size;
			b = {};
		}

		/** @brief `value = mapping(shift, value)` for every element now in heap `heap`. @complexity O(1) */
		constexpr void add_to_all(std::size_t heap, tag_type const &shift)
			requires has_shift
		{
			uint32 const x = record(checked_heap_id(heap)).root;
			if (x != 0) apply(x, shift);
		}
	};

	/**
	 * @brief Skew heaps on one node pool: a meld walks the two right spines
	 *        top-down in one loop and swaps the children of every node it
	 *        passes. A single meld can take O(n) steps (the right spine is
	 *        unbounded), amortized O(log n), so it is iterative only: a
	 *        recursive meld would overflow the stack.
	 *
	 * @tparam T, Compare, Shift, Mapping, Compose As for @ref atl::meldable_heap_forest.
	 * @tparam IsBranchLight `true` picks the next node by a conditional
	 *         select instead of a branch around `std::swap`.
	 */
	template<typename T, typename Compare = std::less<>, typename Shift = void, typename Mapping = shift_by_addition,
			 typename Compose = compose_by_addition, bool IsBranchLight = false>
		requires meldable_heap_parameters<T, Compare, Shift, Mapping, Compose>
	class meldable_heap_forest_by_skew : public atl::detail::heap_directory {
		using tag_type = atl::detail::heap_tag_t<Shift>;
		static constexpr bool has_shift = !std::is_void_v<Shift>;

		struct node {
			T value;
			[[no_unique_address]] tag_type tag{};
			uint32 left = 0, right = 0;
		};

		std::vector<node> nodes_;
		uint32 free_ = 0;
		[[no_unique_address]] Compare comp_;
		[[no_unique_address]] Mapping mapping_;
		[[no_unique_address]] Compose compose_;
		[[no_unique_address]] tag_type shift_identity_{};

		/** @brief Node `x` (1-based; 0 means none). */
		constexpr node &at(uint32 x) { return nodes_[x - 1]; }

		/** @brief Node `x` (1-based; 0 means none). */
		constexpr node const &at(uint32 x) const { return nodes_[x - 1]; }

		/** @brief Shift `x`'s own value and add the shift to what it owes its descendants. */
		constexpr void apply(uint32 x, tag_type const &shift) {
			node &n = at(x);
			n.value = mapping_(shift, std::move(n.value));
			n.tag = compose_(shift, n.tag);
		}

		/** @brief Hand `x`'s pending add (which covers its descendants, not `x`) to its children. */
		constexpr void push_down(uint32 x) {
			if constexpr (has_shift) {
				node &n = at(x);
				if (atl::detail::is_identity_shift(n.tag, shift_identity_)) return;
				if (n.left != 0) apply(n.left, n.tag);
				if (n.right != 0) apply(n.right, n.tag);
				n.tag = shift_identity_;
			}
		}

		/** @brief A fresh node holding `T(args...)`, from the free list when it has one. */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		constexpr uint32 allocate(Args &&...args) {
			if (free_ != 0) {
				uint32 const x = free_;
				node &n = at(x);
				free_ = n.left;
				n.value = T(std::forward<Args>(args)...);
				n.tag = shift_identity_;
				n.left = n.right = 0;
				return x;
			}
			atl_assert(nodes_.size() + 1 < (std::numeric_limits<uint32>::max)());
			nodes_.push_back(node{T(std::forward<Args>(args)...), shift_identity_});
			return uint32(nodes_.size());
		}

		/** @brief Order `(first, second)` so that `first` is not worse than `second`. */
		constexpr void order(uint32 &first, uint32 &second) {
			if constexpr (IsBranchLight) {
				bool const is_swapped = comp_(at(first).value, at(second).value);
				uint32 const better = is_swapped ? second : first;
				second = is_swapped ? first : second;
				first = better;
			} else {
				if (comp_(at(first).value, at(second).value)) std::swap(first, second);
			}
		}

		/** @brief Meld the heaps rooted at `a` and `b` (either may be 0); returns the new root. */
		constexpr uint32 merge(uint32 a, uint32 b) {
			if (a == 0) return b;
			if (b == 0) return a;
			order(a, b);
			uint32 const root = a;
			// `a` is placed; its new left child is the meld of its old right
			// child with `b`, and its old left child becomes its right
			for (;;) {
				push_down(a);
				node &n = at(a);
				uint32 next = n.right;
				n.right = n.left;
				if (next == 0) {
					n.left = b;
					return root;
				}
				order(next, b);
				n.left = next;
				a = next;
			}
		}

	public:
		using value_type = T;
		using shift_type = Shift;

		/**
		 * @brief `heap_count` empty heaps, ids `[0, heap_count)`; with a lazy add,
		 *        its identity is `Shift{}` and the callables are value-initialized.
		 */
		constexpr explicit meldable_heap_forest_by_skew(std::size_t heap_count = 0, Compare comp = {})
			requires(!has_shift
					 || (std::default_initializable<Shift> && std::default_initializable<Mapping>
						 && std::default_initializable<Compose>))
			: heap_directory(heap_count), comp_(std::move(comp)) {}

		/** @brief `heap_count` empty heaps with the lazy add `(shift_identity, mapping, compose)`. */
		constexpr meldable_heap_forest_by_skew(std::size_t heap_count, Compare comp, tag_type shift_identity, Mapping mapping = {},
											   Compose compose = {})
			requires has_shift
			: heap_directory(heap_count),
			  comp_(std::move(comp)),
			  mapping_(std::move(mapping)),
			  compose_(std::move(compose)),
			  shift_identity_(std::move(shift_identity)) {}

		/** @brief Room for `node_count` live elements without reallocating the pool. */
		constexpr void reserve(std::size_t node_count) { nodes_.reserve(node_count); }

		/** @brief The comp-largest element of heap `heap`. @pre the heap is not empty (asserted) */
		constexpr T const &top(std::size_t heap) const {
			atl_assert(record(checked_heap_id(heap)).size != 0);
			return at(record(checked_heap_id(heap)).root).value;
		}

		/** @brief Insert `T(args...)` into heap `heap`; returns its node id (no promote / erase here). @complexity O(log n) amortized */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		constexpr uint32 emplace(std::size_t heap, Args &&...args) {
			atl::detail::heap_record &h = record(checked_heap_id(heap));
			uint32 const x = allocate(std::forward<Args>(args)...);
			h.root = merge(h.root, x);
			++h.size;
			return x;
		}

		/** @brief Insert `value` into heap `heap`. @complexity O(log n) amortized */
		constexpr uint32 push(std::size_t heap, T value) { return emplace(heap, std::move(value)); }

		/** @brief Remove and return the top of heap `heap`. @pre not empty (asserted) @complexity O(log n) amortized */
		constexpr T pop(std::size_t heap) {
			atl::detail::heap_record &h = record(checked_heap_id(heap));
			atl_assert(h.size != 0);
			uint32 const x = h.root;
			push_down(x);
			T top_value = std::move(at(x).value);
			h.root = merge(at(x).left, at(x).right);
			--h.size;
			at(x).left = free_;
			free_ = x;
			return top_value;
		}

		/** @brief Move every element of `from` into `into`; `from` is left empty (a no-op when they are the same heap). */
		constexpr void meld(std::size_t into, std::size_t from) {
			if (into == from) return;
			atl::detail::heap_record &a = record(checked_heap_id(into));
			atl::detail::heap_record &b = record(checked_heap_id(from));
			a.root = merge(a.root, b.root);
			a.size += b.size;
			b = {};
		}

		/** @brief `value = mapping(shift, value)` for every element now in heap `heap`. @complexity O(1) */
		constexpr void add_to_all(std::size_t heap, tag_type const &shift)
			requires has_shift
		{
			uint32 const x = record(checked_heap_id(heap)).root;
			if (x != 0) apply(x, shift);
		}
	};

	/**
	 * @brief Pairing heaps on one node pool, children kept as a first-child /
	 *        next-sibling list: meld links two roots in O(1); pop pairs the
	 *        root's children left to right, then folds the pairs right to left
	 *        (two-pass pairing), in place through the sibling links.
	 *
	 * A pending add at a node covers its first-child and next-sibling
	 * subtrees in the binary view of the lists, which are exactly its
	 * descendants for a root, so handing it down is O(1).
	 *
	 * @tparam T, Compare, Shift, Mapping, Compose As for @ref atl::meldable_heap_forest.
	 * @tparam IsBranchLight `true` links by a conditional select instead of a
	 *         branch around `std::swap`.
	 * @tparam IsRecycling `false` never reuses a popped node (the pool grows
	 *         with every push), to measure what the free list costs or saves.
	 * @tparam IsSkippingIdentityShift `true` (with `==` on `Shift`) skips
	 *         handing down a pending add equal to the identity, which would
	 *         touch the child and sibling for nothing; `false` always hands
	 *         it down.
	 */
	template<typename T, typename Compare = std::less<>, typename Shift = void, typename Mapping = shift_by_addition,
			 typename Compose = compose_by_addition, bool IsBranchLight = false,
			 bool IsRecycling = true, bool IsSkippingIdentityShift = true>
		requires meldable_heap_parameters<T, Compare, Shift, Mapping, Compose>
	class meldable_heap_forest_by_pairing : public atl::detail::heap_directory {
		using tag_type = atl::detail::heap_tag_t<Shift>;
		static constexpr bool has_shift = !std::is_void_v<Shift>;

		struct node {
			T value;
			[[no_unique_address]] tag_type tag{};
			uint32 child = 0, sibling = 0;
		};

		std::vector<node> nodes_;
		uint32 free_ = 0;
		[[no_unique_address]] Compare comp_;
		[[no_unique_address]] Mapping mapping_;
		[[no_unique_address]] Compose compose_;
		[[no_unique_address]] tag_type shift_identity_{};

		/** @brief Node `x` (1-based; 0 means none). */
		constexpr node &at(uint32 x) { return nodes_[x - 1]; }

		/** @brief Node `x` (1-based; 0 means none). */
		constexpr node const &at(uint32 x) const { return nodes_[x - 1]; }

		/** @brief Shift `x`'s own value and add the shift to what it owes its descendants. */
		constexpr void apply(uint32 x, tag_type const &shift) {
			node &n = at(x);
			n.value = mapping_(shift, std::move(n.value));
			n.tag = compose_(shift, n.tag);
		}

		/** @brief Hand `x`'s pending add to its first child and next sibling. */
		constexpr void push_down(uint32 x) {
			if constexpr (has_shift) {
				node &n = at(x);
				if (IsSkippingIdentityShift && atl::detail::is_identity_shift(n.tag, shift_identity_)) return;
				if (n.child != 0) apply(n.child, n.tag);
				if (n.sibling != 0) apply(n.sibling, n.tag);
				n.tag = shift_identity_;
			}
		}

		/** @brief A fresh node holding `T(args...)`, from the free list when it has one. */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		constexpr uint32 allocate(Args &&...args) {
			if constexpr (IsRecycling) {
				if (free_ != 0) {
					uint32 const x = free_;
					node &n = at(x);
					free_ = n.child;
					n.value = T(std::forward<Args>(args)...);
					n.tag = shift_identity_;
					n.child = n.sibling = 0;
					return x;
				}
			}
			atl_assert(nodes_.size() + 1 < (std::numeric_limits<uint32>::max)());
			nodes_.push_back(node{T(std::forward<Args>(args)...), shift_identity_});
			return uint32(nodes_.size());
		}

		/** @brief Link two roots with no pending adds and no siblings; returns the new root. */
		constexpr uint32 link(uint32 a, uint32 b) {
			uint32 winner = a, loser = b;
			if constexpr (IsBranchLight) {
				bool const is_swapped = comp_(at(a).value, at(b).value);
				winner = is_swapped ? b : a;
				loser = is_swapped ? a : b;
			} else {
				if (comp_(at(a).value, at(b).value)) std::swap(winner, loser);
			}
			at(loser).sibling = at(winner).child;
			at(winner).child = loser;
			return winner;
		}

		/** @brief Link two roots (either may be 0 = empty), clearing their pending adds first. */
		constexpr uint32 meld_roots(uint32 a, uint32 b) {
			if (a == 0) return b;
			if (b == 0) return a;
			push_down(a);
			push_down(b);
			return link(a, b);
		}

		/** @brief Two-pass pairing of the sibling list starting at `first`; returns the new root. */
		constexpr uint32 pair_up(uint32 first) {
			if (first == 0) return 0;
			uint32 pairs = 0; // the pairs so far, last first, linked through `sibling`
			while (first != 0) {
				push_down(first);
				uint32 const second = at(first).sibling;
				if (second == 0) {
					at(first).sibling = pairs;
					pairs = first;
					break;
				}
				push_down(second);
				uint32 const rest = at(second).sibling;
				uint32 const winner = link(first, second);
				at(winner).sibling = pairs;
				pairs = winner;
				first = rest;
			}
			uint32 root = pairs;
			pairs = at(root).sibling;
			while (pairs != 0) {
				uint32 const rest = at(pairs).sibling;
				root = link(root, pairs);
				pairs = rest;
			}
			at(root).sibling = 0;
			return root;
		}

	public:
		using value_type = T;
		using shift_type = Shift;

		/**
		 * @brief `heap_count` empty heaps, ids `[0, heap_count)`; with a lazy add,
		 *        its identity is `Shift{}` and the callables are value-initialized.
		 */
		constexpr explicit meldable_heap_forest_by_pairing(std::size_t heap_count = 0, Compare comp = {})
			requires(!has_shift
					 || (std::default_initializable<Shift> && std::default_initializable<Mapping>
						 && std::default_initializable<Compose>))
			: heap_directory(heap_count), comp_(std::move(comp)) {}

		/** @brief `heap_count` empty heaps with the lazy add `(shift_identity, mapping, compose)`. */
		constexpr meldable_heap_forest_by_pairing(std::size_t heap_count, Compare comp, tag_type shift_identity, Mapping mapping = {},
												  Compose compose = {})
			requires has_shift
			: heap_directory(heap_count),
			  comp_(std::move(comp)),
			  mapping_(std::move(mapping)),
			  compose_(std::move(compose)),
			  shift_identity_(std::move(shift_identity)) {}

		/** @brief Room for `node_count` live elements without reallocating the pool. */
		constexpr void reserve(std::size_t node_count) { nodes_.reserve(node_count); }

		/** @brief The comp-largest element of heap `heap`. @pre the heap is not empty (asserted) */
		constexpr T const &top(std::size_t heap) const {
			atl_assert(record(checked_heap_id(heap)).size != 0);
			return at(record(checked_heap_id(heap)).root).value;
		}

		/** @brief Insert `T(args...)` into heap `heap`; returns its node id (no promote / erase here). @complexity O(1) */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		constexpr uint32 emplace(std::size_t heap, Args &&...args) {
			atl::detail::heap_record &h = record(checked_heap_id(heap));
			uint32 const x = allocate(std::forward<Args>(args)...);
			h.root = meld_roots(h.root, x);
			++h.size;
			return x;
		}

		/** @brief Insert `value` into heap `heap`. @complexity O(1) */
		constexpr uint32 push(std::size_t heap, T value) { return emplace(heap, std::move(value)); }

		/** @brief Remove and return the top of heap `heap`. @pre not empty (asserted) @complexity O(log n) amortized */
		constexpr T pop(std::size_t heap) {
			atl::detail::heap_record &h = record(checked_heap_id(heap));
			atl_assert(h.size != 0);
			uint32 const x = h.root;
			push_down(x);
			T top_value = std::move(at(x).value);
			h.root = pair_up(at(x).child);
			--h.size;
			if constexpr (IsRecycling) {
				at(x).child = free_;
				free_ = x;
			}
			return top_value;
		}

		/** @brief Move every element of `from` into `into`; `from` is left empty (a no-op when they are the same heap). @complexity O(1) */
		constexpr void meld(std::size_t into, std::size_t from) {
			if (into == from) return;
			atl::detail::heap_record &a = record(checked_heap_id(into));
			atl::detail::heap_record &b = record(checked_heap_id(from));
			a.root = meld_roots(a.root, b.root);
			a.size += b.size;
			b = {};
		}

		/** @brief `value = mapping(shift, value)` for every element now in heap `heap`. @complexity O(1) */
		constexpr void add_to_all(std::size_t heap, tag_type const &shift)
			requires has_shift
		{
			uint32 const x = record(checked_heap_id(heap)).root;
			if (x != 0) apply(x, shift);
		}
	};

	/**
	 * @brief @ref atl::experimental::meldable_heap_forest_by_pairing with the pool split into arrays
	 *        (structure of arrays): values, pending adds and links each in
	 *        their own vector, so a pass over links does not pull values into
	 *        cache and a comparison does not pull links.
	 *
	 * @tparam T, Compare, Shift, Mapping, Compose As for @ref atl::meldable_heap_forest.
	 */
	template<typename T, typename Compare = std::less<>, typename Shift = void, typename Mapping = shift_by_addition,
			 typename Compose = compose_by_addition>
		requires meldable_heap_parameters<T, Compare, Shift, Mapping, Compose>
	class meldable_heap_forest_by_split_pairing : public atl::detail::heap_directory {
		using tag_type = atl::detail::heap_tag_t<Shift>;
		static constexpr bool has_shift = !std::is_void_v<Shift>;

		struct links {
			uint32 child = 0, sibling = 0;
		};

		// index 0 of every array is the empty sentinel's slot, so node x sits at [x]
		std::vector<T> values_;
		std::vector<tag_type> tags_; // stays empty without a lazy add
		std::vector<links> links_{links{}};
		uint32 free_ = 0;
		[[no_unique_address]] Compare comp_;
		[[no_unique_address]] Mapping mapping_;
		[[no_unique_address]] Compose compose_;
		[[no_unique_address]] tag_type shift_identity_{};

		/** @brief Node `x`'s value (1-based; 0 means none). */
		constexpr T &value(uint32 x) { return values_[x - 1]; }

		/** @brief Node `x`'s value (1-based; 0 means none). */
		constexpr T const &value(uint32 x) const { return values_[x - 1]; }

		/** @brief Shift `x`'s own value and add the shift to what it owes its descendants. */
		constexpr void apply(uint32 x, tag_type const &shift) {
			values_[x - 1] = mapping_(shift, std::move(values_[x - 1]));
			tags_[x - 1] = compose_(shift, tags_[x - 1]);
		}

		/** @brief Hand `x`'s pending add to its first child and next sibling. */
		constexpr void push_down(uint32 x) {
			if constexpr (has_shift) {
				if (atl::detail::is_identity_shift(tags_[x - 1], shift_identity_)) return;
				links const l = links_[x];
				if (l.child != 0) apply(l.child, tags_[x - 1]);
				if (l.sibling != 0) apply(l.sibling, tags_[x - 1]);
				tags_[x - 1] = shift_identity_;
			}
		}

		/** @brief A fresh node holding `T(args...)`, from the free list when it has one. */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		constexpr uint32 allocate(Args &&...args) {
			if (free_ != 0) {
				uint32 const x = free_;
				free_ = links_[x].child;
				value(x) = T(std::forward<Args>(args)...);
				if constexpr (has_shift) tags_[x - 1] = shift_identity_;
				links_[x] = {};
				return x;
			}
			atl_assert(links_.size() < (std::numeric_limits<uint32>::max)());
			values_.push_back(T(std::forward<Args>(args)...));
			if constexpr (has_shift) tags_.push_back(shift_identity_);
			links_.emplace_back();
			return uint32(values_.size());
		}

		/** @brief Link two roots with no pending adds and no siblings; returns the new root. */
		constexpr uint32 link(uint32 a, uint32 b) {
			if (comp_(value(a), value(b))) std::swap(a, b);
			links_[b].sibling = links_[a].child;
			links_[a].child = b;
			return a;
		}

		/** @brief Link two roots, either possibly 0 (empty); returns the new root. */
		constexpr uint32 meld_roots(uint32 a, uint32 b) {
			if (a == 0) return b;
			if (b == 0) return a;
			push_down(a);
			push_down(b);
			return link(a, b);
		}

		/** @brief Two-pass pairing of the sibling list starting at `first`; returns the new root. */
		constexpr uint32 pair_up(uint32 first) {
			if (first == 0) return 0;
			uint32 pairs = 0;
			while (first != 0) {
				push_down(first);
				uint32 const second = links_[first].sibling;
				if (second == 0) {
					links_[first].sibling = pairs;
					pairs = first;
					break;
				}
				push_down(second);
				uint32 const rest = links_[second].sibling;
				uint32 const winner = link(first, second);
				links_[winner].sibling = pairs;
				pairs = winner;
				first = rest;
			}
			uint32 root = pairs;
			pairs = links_[root].sibling;
			while (pairs != 0) {
				uint32 const rest = links_[pairs].sibling;
				root = link(root, pairs);
				pairs = rest;
			}
			links_[root].sibling = 0;
			return root;
		}

	public:
		using value_type = T;
		using shift_type = Shift;

		/**
		 * @brief `heap_count` empty heaps, ids `[0, heap_count)`; with a lazy add,
		 *        its identity is `Shift{}` and the callables are value-initialized.
		 */
		constexpr explicit meldable_heap_forest_by_split_pairing(std::size_t heap_count = 0, Compare comp = {})
			requires(!has_shift
					 || (std::default_initializable<Shift> && std::default_initializable<Mapping>
						 && std::default_initializable<Compose>))
			: heap_directory(heap_count), comp_(std::move(comp)) {}

		/** @brief `heap_count` empty heaps with the lazy add `(shift_identity, mapping, compose)`. */
		constexpr meldable_heap_forest_by_split_pairing(std::size_t heap_count, Compare comp, tag_type shift_identity, Mapping mapping = {},
														Compose compose = {})
			requires has_shift
			: heap_directory(heap_count),
			  comp_(std::move(comp)),
			  mapping_(std::move(mapping)),
			  compose_(std::move(compose)),
			  shift_identity_(std::move(shift_identity)) {}

		/** @brief Room for `node_count` live elements without reallocating the pool. */
		constexpr void reserve(std::size_t node_count) {
			values_.reserve(node_count);
			if constexpr (has_shift) tags_.reserve(node_count);
			links_.reserve(node_count + 1);
		}

		/** @brief The comp-largest element of heap `heap`. @pre the heap is not empty (asserted) */
		constexpr T const &top(std::size_t heap) const {
			atl_assert(record(checked_heap_id(heap)).size != 0);
			return value(record(checked_heap_id(heap)).root);
		}

		/** @brief Insert `T(args...)` into heap `heap`; returns its node id (no promote / erase here). @complexity O(1) */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		constexpr uint32 emplace(std::size_t heap, Args &&...args) {
			atl::detail::heap_record &h = record(checked_heap_id(heap));
			uint32 const x = allocate(std::forward<Args>(args)...);
			h.root = meld_roots(h.root, x);
			++h.size;
			return x;
		}

		/** @brief Insert `value` into heap `heap`. @complexity O(1) */
		constexpr uint32 push(std::size_t heap, T value) { return emplace(heap, std::move(value)); }

		/** @brief Remove and return the top of heap `heap`. @pre not empty (asserted) @complexity O(log n) amortized */
		constexpr T pop(std::size_t heap) {
			atl::detail::heap_record &h = record(checked_heap_id(heap));
			atl_assert(h.size != 0);
			uint32 const x = h.root;
			push_down(x);
			T top_value = std::move(value(x));
			h.root = pair_up(links_[x].child);
			--h.size;
			links_[x].child = free_;
			free_ = x;
			return top_value;
		}

		/** @brief Move every element of `from` into `into`; `from` is left empty (a no-op when they are the same heap). @complexity O(1) */
		constexpr void meld(std::size_t into, std::size_t from) {
			if (into == from) return;
			atl::detail::heap_record &a = record(checked_heap_id(into));
			atl::detail::heap_record &b = record(checked_heap_id(from));
			a.root = meld_roots(a.root, b.root);
			a.size += b.size;
			b = {};
		}

		/** @brief `value = mapping(shift, value)` for every element now in heap `heap`. @complexity O(1) */
		constexpr void add_to_all(std::size_t heap, tag_type const &shift)
			requires has_shift
		{
			uint32 const x = record(checked_heap_id(heap)).root;
			if (x != 0) apply(x, shift);
		}
	};

	/** @brief The default inverse of a lazy add for @ref atl::experimental::meldable_heap_forest_by_binary — `-shift`. */
	struct negate_shift {
		template<typename Shift>
			requires requires(Shift const &shift) {
				{ -shift } -> std::convertible_to<Shift>;
			}
		constexpr Shift operator()(Shift const &shift) const {
			if constexpr (std::is_unsigned_v<Shift>)
				return Shift(Shift{} - shift);
			else
				return Shift(-shift);
		}
	};

	/**
	 * @brief The baseline: one `std::vector` binary heap per id, melded
	 *        smaller-into-larger (each moved element is pushed into the larger
	 *        heap), O(n log^2 n) over any sequence of melds.
	 *
	 * A lazy add is one offset per heap: a heap stores `mapping(invert(offset),
	 * value)` for each element, so unlike the node-pool forests it needs the
	 * shifts to form a group (`Invert`, `-shift` by default) to re-express an
	 * element against another heap's offset, and `top` returns by value.
	 *
	 * @tparam T, Compare, Shift, Mapping, Compose As for @ref atl::meldable_heap_forest.
	 * @tparam IsBulkRebuilding `true` appends the smaller heap and rebuilds
	 *         with `make_heap` when pushing each element could cost more
	 *         (`smaller_size * log2(total) > 2 * total`); `false` always pushes.
	 */
	template<typename T, typename Compare = std::less<>, typename Shift = void, typename Mapping = shift_by_addition,
			 typename Compose = compose_by_addition, bool IsBulkRebuilding = false, typename Invert = negate_shift>
		requires meldable_heap_parameters<T, Compare, Shift, Mapping, Compose>
			  && (std::is_void_v<Shift> || std::regular_invocable<Invert const &, Shift const &>)
	class meldable_heap_forest_by_binary {
		using tag_type = atl::detail::heap_tag_t<Shift>;
		static constexpr bool has_shift = !std::is_void_v<Shift>;

		std::vector<std::vector<T>> heaps_;
		std::vector<tag_type> offsets_; // stays empty without a lazy add
		[[no_unique_address]] Compare comp_;
		[[no_unique_address]] Mapping mapping_;
		[[no_unique_address]] Compose compose_;
		[[no_unique_address]] Invert invert_;
		[[no_unique_address]] tag_type shift_identity_{};

		/** @brief Heap `heap`'s vector. @pre `heap < heap_count()` (asserted) */
		constexpr std::vector<T> &heap_at(std::size_t heap) {
			atl_assert(heap < heaps_.size());
			return heaps_[heap];
		}

		/** @brief Heap `heap`'s vector. @pre `heap < heap_count()` (asserted) */
		constexpr std::vector<T> const &heap_at(std::size_t heap) const {
			atl_assert(heap < heaps_.size());
			return heaps_[heap];
		}

	public:
		using value_type = T;
		using shift_type = Shift;

		/** @brief `heap_count` empty heaps; with a lazy add, identity `Shift{}` and value-initialized callables. */
		constexpr explicit meldable_heap_forest_by_binary(std::size_t heap_count = 0, Compare comp = {})
			requires(!has_shift
					 || (std::default_initializable<Shift> && std::default_initializable<Mapping>
						 && std::default_initializable<Compose> && std::default_initializable<Invert>))
			: heaps_(heap_count), comp_(std::move(comp)) {
			if constexpr (has_shift) offsets_.assign(heap_count, shift_identity_);
		}

		/** @brief `heap_count` empty heaps with the lazy add `(shift_identity, mapping, compose)` and its inverse. */
		constexpr meldable_heap_forest_by_binary(std::size_t heap_count, Compare comp, tag_type shift_identity, Mapping mapping = {},
												 Compose compose = {}, Invert invert = {})
			requires has_shift
			: heaps_(heap_count),
			  comp_(std::move(comp)),
			  mapping_(std::move(mapping)),
			  compose_(std::move(compose)),
			  invert_(std::move(invert)),
			  shift_identity_(std::move(shift_identity)) {
			offsets_.assign(heap_count, shift_identity_);
		}

		/** @brief Nothing to reserve per node: each heap owns its vector. */
		constexpr void reserve(std::size_t) {}

		/** @brief The number of heaps, empty ones included. */
		constexpr std::size_t heap_count() const noexcept { return heaps_.size(); }

		/** @brief A new empty heap; returns its id. */
		constexpr std::size_t add_heap() {
			heaps_.emplace_back();
			if constexpr (has_shift) offsets_.push_back(shift_identity_);
			return heaps_.size() - 1;
		}

		/** @brief The number of elements in heap `heap`. */
		constexpr std::size_t size(std::size_t heap) const { return heap_at(heap).size(); }

		/** @brief Whether heap `heap` holds no element. */
		constexpr bool empty(std::size_t heap) const { return heap_at(heap).empty(); }

		/** @brief The comp-largest element of heap `heap` (by value with a lazy add). @pre not empty (asserted) */
		constexpr decltype(auto) top(std::size_t heap) const {
			std::vector<T> const &h = heap_at(heap);
			atl_assert(!h.empty());
			if constexpr (has_shift)
				return T(mapping_(offsets_[heap], h.front()));
			else
				return static_cast<T const &>(h.front());
		}

		/** @brief Insert `T(args...)` into heap `heap` (no handle: elements move inside the vectors). @complexity O(log n) */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		constexpr void emplace(std::size_t heap, Args &&...args) {
			std::vector<T> &h = heap_at(heap);
			if constexpr (has_shift)
				h.push_back(mapping_(Shift(invert_(offsets_[heap])), T(std::forward<Args>(args)...)));
			else
				h.emplace_back(std::forward<Args>(args)...);
			std::ranges::push_heap(h, comp_);
		}

		/** @brief Insert `value` into heap `heap`. @complexity O(log n) */
		constexpr void push(std::size_t heap, T value) { emplace(heap, std::move(value)); }

		/** @brief Remove and return the top of heap `heap`. @pre not empty (asserted) @complexity O(log n) */
		constexpr T pop(std::size_t heap) {
			std::vector<T> &h = heap_at(heap);
			atl_assert(!h.empty());
			std::ranges::pop_heap(h, comp_);
			T top_value = std::move(h.back());
			h.pop_back();
			if constexpr (has_shift) top_value = mapping_(offsets_[heap], std::move(top_value));
			return top_value;
		}

		/** @brief Move every element of `from` into `into`; `from` is left empty. @complexity O(min(n, m) * log(n + m)) */
		constexpr void meld(std::size_t into, std::size_t from) {
			if (into == from) return;
			std::vector<T> *large = &heap_at(into);
			std::vector<T> *smaller_heap = &heap_at(from);
			if (large->size() < smaller_heap->size()) {
				std::swap(*large, *smaller_heap);
				if constexpr (has_shift) std::swap(offsets_[into], offsets_[from]);
			}
			if constexpr (has_shift) {
				Shift const delta = compose_(Shift(invert_(offsets_[into])), offsets_[from]);
				for (T &value : *smaller_heap) value = mapping_(delta, std::move(value));
				offsets_[from] = shift_identity_;
			}
			std::size_t const total = large->size() + smaller_heap->size();
			bool should_rebuild = false;
			if constexpr (IsBulkRebuilding)
				should_rebuild = uint64(smaller_heap->size()) * std::size_t(std::bit_width(total)) > uint64(2) * total;
			if (should_rebuild) {
				large->insert(large->end(), std::make_move_iterator(smaller_heap->begin()), std::make_move_iterator(smaller_heap->end()));
				std::ranges::make_heap(*large, comp_);
			} else {
				for (T &value : *smaller_heap) {
					large->push_back(std::move(value));
					std::ranges::push_heap(*large, comp_);
				}
			}
			smaller_heap->clear();
		}

		/** @brief `value = mapping(shift, value)` for every element now in heap `heap`. @complexity O(1) */
		constexpr void add_to_all(std::size_t heap, tag_type const &shift)
			requires has_shift
		{
			heap_at(heap);
			offsets_[heap] = compose_(shift, offsets_[heap]);
		}
	};
}
