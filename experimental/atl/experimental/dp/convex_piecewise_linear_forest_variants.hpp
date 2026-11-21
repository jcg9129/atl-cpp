#pragma once

/**
 * @file
 * @brief Contenders for the breakpoint pool of
 *        @ref atl::convex_piecewise_linear_forest (`detail::breakpoint_pool`,
 *        which empties a discarded heap with `meldable_heap_forest::release`).
 *
 * - @ref atl::experimental::breakpoint_pool_by_repeated_pops: the library
 *   pool with `clear` and the discarded right heap of `take_all` emptied by
 *   repeated pops instead of a release.
 * - @ref atl::experimental::breakpoint_pool_by_tunable_pairing_pops: the same
 *   repeated pops on the tunable pairing-heap contender
 *   (`meldable_heap_forest_by_pairing` at its defaults).
 *
 * Race verdict (benchmarks/dp/convex_piecewise_linear_forest.cpp, records of
 * 2026-11-01, Xeon Platinum 8280, GCC 15.2, ms, plain -O2 / -march=native):
 * every pop of a discarded heap pairs the children of its root, while a
 * release walks the links once with no comparison. Clear of one set of 1e6:
 * library 74.1 / 75.3, by repeated pops 1357 / 1348, by tunable pairing pops
 * 1332 / 1340; clear and refill of 1000 sets of 1000, three rounds: 632 /
 * 631, 1724 / 1693, 1666 / 1683; take_all of one set of 1e6 (the left heap
 * still pops to emit the order): 812 / 807, 1463 / 1449, 1447 / 1435. A
 * multimap per set certifies all three in the tests.
 *
 * And a contender for the forest's `insert_segment`:
 *
 * - @ref atl::experimental::insert_segment_by_fewer_entries: walks both ends
 *   of the function at once, one live entry from each in turn, and moves the
 *   side that reaches the crossing first, i.e. the side with fewer entries
 *   to the crossing, where the library moves the side of less weight. On a
 *   bounded leaf-balancing path the light side can hold many small pieces
 *   while the heavy side is one record (a domain wall): there the library
 *   moves `m` pieces per edge with `m` growing along the path, quadratic in
 *   all, and this moves a constant number. Not measured yet
 *   (benchmarks/dp/leaf_path_balance.cpp races it through
 *   @ref atl::experimental::balance_leaf_paths_by_fewer_entries); the library
 *   keeps the weight rule until it is.
 */

#include <atl/dp/convex_piecewise_linear.hpp>
#include <atl/dp/convex_piecewise_linear_forest.hpp>
#include <atl/experimental/ds/meldable_heap_variants.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::experimental {
	/** @brief The scratch of @ref atl::experimental::insert_segment_by_fewer_entries: the pieces walked from each end. */
	template<piecewise_linear_value Position, signed_integral Slope>
	struct fewer_entries_scratch {
		/** @brief The pieces taken from the left end, in walk order (true positions). */
		std::vector<weighted_breakpoint<Position, Slope>> left;
		/** @brief The pieces taken from the right end, in walk order (true positions). */
		std::vector<weighted_breakpoint<Position, Slope>> right;
	};

	/**
	 * @brief The forest's `insert_segment(slope, length)` on the function
	 *        stored as `set` of `pool`, whose end slopes are `leftmost` and
	 *        `rightmost` (unchanged by the call), moving the side with FEWER
	 *        ENTRIES between its end and the crossing instead of the side of
	 *        less weight.
	 *
	 * The two ends are walked in turn, one live entry each: an entry taken
	 * is removed from its heap and its record emptied, so the other walk sees
	 * it as stale and the two never share a record. A walk is done when the
	 * weight it took reaches what its side needs (`slope - leftmost` from the
	 * left, `rightmost - slope` from the right); the remaining live weight
	 * always exceeds what an unfinished walk still needs, so neither runs out.
	 * The finished side is the one moved, exactly as the library moves it
	 * (left: its pieces stay and the rest is translated; right: its pieces
	 * move by `length`), its last piece split at the crossing; the other
	 * walk's pieces go back where they were. Every piece taken is pushed again
	 * as a new record.
	 *
	 * Same contract and result as the library's, except which end of a piece
	 * of slope exactly `slope` is returned: the right end when the right side
	 * is moved, the left end otherwise, now by entry counts (both optimal).
	 *
	 * @pre `length >= 0`, `slope <= rightmost` (asserted); `leftmost` and
	 *      `rightmost` are the function's end slopes (documented)
	 * @complexity O((min(k_left, k_right) + 1) log n) amortized, for
	 *      `k_left` and `k_right` the live entries between each end and the
	 *      crossing; as many new records. Not measured yet.
	 */
	template<piecewise_linear_value Position, signed_integral Slope>
	constexpr std::optional<Position> insert_segment_by_fewer_entries(atl::detail::breakpoint_pool<Position, Slope> &pool,
																	  std::size_t set, std::type_identity_t<Slope> leftmost,
																	  std::type_identity_t<Slope> rightmost,
																	  std::type_identity_t<Slope> slope,
																	  std::type_identity_t<Position> const &segment_length,
																	  fewer_entries_scratch<Position, Slope> &scratch) {
		Position const length = segment_length;
		atl_assert(!(length < Position{}));
		atl_assert(slope <= rightmost);
		if (slope <= leftmost) {
			pool.translate(set, length);
			return std::nullopt;
		}
		if (slope == rightmost) {
			[[maybe_unused]] bool const is_live = pool.template has_live<false>(set);
			atl_assert(is_live);
			return pool.template top<false>(set).position;
		}
		Slope const need_left = Slope(slope - leftmost), need_right = Slope(rightmost - slope);
		Slope passed_left{}, passed_right{};
		scratch.left.clear();
		scratch.right.clear();
		auto const take = [&pool, set]<bool IsFromLeft>(std::vector<weighted_breakpoint<Position, Slope>> &walked,
														Slope &passed) {
			[[maybe_unused]] bool const is_live = pool.template has_live<IsFromLeft>(set);
			atl_assert(is_live);
			auto const e = pool.template pop<IsFromLeft>(set);
			Slope &weight = pool.weight(e.record);
			walked.push_back({e.position, weight});
			passed = Slope(passed + weight);
			weight = Slope{};
		};
		bool is_left_moved = false;
		for (;;) {
			take.template operator()<true>(scratch.left, passed_left);
			if (!(passed_left < need_left)) {
				is_left_moved = true;
				break;
			}
			take.template operator()<false>(scratch.right, passed_right);
			if (!(passed_right < need_right)) break;
		}
		auto const push = [&pool, set](Position const &position, Slope weight) {
			if (weight != 0) pool.push(set, position, weight);
		};
		if (is_left_moved) {
			// the left part stays, the rest moves: the split piece's excess and
			// the right walk's pieces go back moved
			auto const &last = scratch.left.back();
			Slope const excess = Slope(passed_left - need_left);
			pool.translate(set, length);
			for (std::size_t i = 0; i + 1 < scratch.left.size(); ++i) push(scratch.left[i].position, scratch.left[i].weight);
			push(last.position, Slope(last.weight - excess));
			push(Position(last.position + length), excess);
			for (auto const &[position, weight] : scratch.right) push(Position(position + length), weight);
			return last.position;
		}
		auto const &last = scratch.right.back();
		Slope const excess = Slope(passed_right - need_right);
		for (std::size_t i = 0; i + 1 < scratch.right.size(); ++i)
			push(Position(scratch.right[i].position + length), scratch.right[i].weight);
		push(Position(last.position + length), Slope(last.weight - excess));
		push(last.position, excess);
		for (auto const &[position, weight] : scratch.left) push(position, weight);
		return last.position;
	}

	/**
	 * @brief The breakpoint pool's push, translate, meld, clear and take_all on
	 *        @ref atl::experimental::meldable_heap_forest_by_pairing, a
	 *        discarded heap emptied by repeated pops. Lost to the library's
	 *        release: clear of one set of 1e6 took 1332 / 1340 ms against
	 *        74.1 / 75.3 (plain -O2 / -march=native, 2026-11-01; file verdict).
	 * @tparam Position The breakpoint coordinate.
	 * @tparam Slope The weight type.
	 */
	template<piecewise_linear_value Position, signed_integral Slope>
	class breakpoint_pool_by_tunable_pairing_pops {
		/** @brief A heap entry: a position and the record holding its weight. */
		using entry = atl::detail::forest_breakpoint<Position>;

		/** @brief One pool of per-set heaps of entries, ordered by `Compare`, with a lazy translation. */
		template<typename Compare>
			requires std::strict_weak_order<Compare &, entry const &, entry const &>
		using heap_pool =
			meldable_heap_forest_by_pairing<entry, Compare, Position, atl::detail::translate_breakpoint, compose_by_addition>;

		/** @brief Per set, a min-heap of its entries by position (top = leftmost). */
		heap_pool<atl::detail::breakpoint_right_of> leftmost_first_;
		/** @brief Per set, a max-heap of its entries by position (top = rightmost). */
		heap_pool<atl::detail::breakpoint_left_of> rightmost_first_;
		/** @brief Per record, its remaining weight; 0 = eaten, its entries are stale. */
		std::vector<Slope> weights_;

	public:
		/** @brief `set_count` empty sets. */
		constexpr explicit breakpoint_pool_by_tunable_pairing_pops(std::size_t set_count)
			: leftmost_first_(set_count), rightmost_first_(set_count) {}

		/** @brief Room for `elements` records (breakpoints) without reallocating. */
		constexpr void reserve(std::size_t elements) {
			leftmost_first_.reserve(elements);
			rightmost_first_.reserve(elements);
			weights_.reserve(elements);
		}

		/** @brief The number of sets. */
		[[nodiscard]] constexpr std::size_t set_count() const noexcept { return leftmost_first_.heap_count(); }

		/** @brief A new empty set; returns its id. */
		constexpr std::size_t add_set() {
			rightmost_first_.add_heap();
			return leftmost_first_.add_heap();
		}

		/** @brief A new record `(position, weight)` in both heaps of `set`. */
		constexpr void push(std::size_t set, Position const &position, Slope weight) {
			atl_assert(weights_.size() < std::numeric_limits<uint32>::max());
			auto const record = uint32(weights_.size());
			weights_.push_back(weight);
			leftmost_first_.push(set, entry{position, record});
			rightmost_first_.push(set, entry{position, record});
		}

		/** @brief `position += d` for every breakpoint now in `set`. */
		constexpr void translate(std::size_t set, Position const &d) {
			leftmost_first_.add_to_all(set, d);
			rightmost_first_.add_to_all(set, d);
		}

		/** @brief Move every breakpoint of `from` into `into` (a no-op when equal). O(1). */
		constexpr void meld(std::size_t into, std::size_t from) {
			leftmost_first_.meld(into, from);
			rightmost_first_.meld(into, from);
		}

		/** @brief Empty `set` by repeated pops. */
		constexpr void clear(std::size_t set) {
			while (!leftmost_first_.empty(set)) leftmost_first_.pop(set);
			while (!rightmost_first_.empty(set)) rightmost_first_.pop(set);
		}

		/**
		 * @brief Append the breakpoints of `set` by increasing position to `out`,
		 *        equal positions combined (also with `out`'s last when it is at the
		 *        same position), and empty the set.
		 */
		constexpr void take_all(std::size_t set, std::vector<weighted_breakpoint<Position, Slope>> &out) {
			while (!leftmost_first_.empty(set)) {
				entry const top = leftmost_first_.pop(set);
				Slope &weight = weights_[top.record];
				if (weight == 0) continue;
				if (!out.empty() && !(out.back().position < top.position) && !(top.position < out.back().position)) {
					atl_assert(weight <= std::numeric_limits<Slope>::max() - out.back().weight);
					out.back().weight += weight;
				} else {
					out.push_back({top.position, weight});
				}
				weight = 0;
			}
			while (!rightmost_first_.empty(set)) rightmost_first_.pop(set);
		}
	};

	/**
	 * @brief The library breakpoint pool with a discarded heap emptied by
	 *        repeated pops instead of a release. Lost: clear of one set of
	 *        1e6 took 1357 / 1348 ms against 74.1 / 75.3 (plain -O2 /
	 *        -march=native, 2026-11-01; file verdict).
	 * @tparam Position The breakpoint coordinate.
	 * @tparam Slope The weight type.
	 */
	template<piecewise_linear_value Position, signed_integral Slope>
	class breakpoint_pool_by_repeated_pops {
	public:
		/** @brief A heap entry. */
		using entry = atl::detail::forest_breakpoint<Position>;

	private:
		/** @brief One pool of per-set heaps of entries, ordered by `Compare`, with a lazy translation. */
		template<typename Compare>
			requires std::strict_weak_order<Compare &, entry const &, entry const &>
		using heap_pool =
			meldable_heap_forest<entry, Compare, Position, atl::detail::translate_breakpoint, compose_by_addition>;

		/** @brief Per set, a min-heap of its entries by position (top = leftmost). */
		heap_pool<atl::detail::breakpoint_right_of> leftmost_first_;
		/** @brief Per set, a max-heap of its entries by position (top = rightmost). */
		heap_pool<atl::detail::breakpoint_left_of> rightmost_first_;
		/** @brief Per record, its remaining weight; 0 = eaten, its entries are stale. */
		std::vector<Slope> weights_;
		/** @brief Scratch for `locate` and `visit`. */
		std::vector<entry> walked_;

		/** @brief The heaps reached from the left end (`IsFromLeft`: the min-heaps) or the right end. */
		template<bool IsFromLeft>
		constexpr auto &heaps() {
			if constexpr (IsFromLeft)
				return leftmost_first_;
			else
				return rightmost_first_;
		}

	public:
		/** @brief `set` as a heap id of both pools. @pre `set < set_count()` (asserted) */
		[[nodiscard]] constexpr uint32 heap_of(std::size_t set) const {
			atl_assert(set < leftmost_first_.heap_count());
			return uint32(set);
		}

		/**
		 * @brief `set_count` empty sets.
		 * @pre `set_count < 2^32` (asserted by the heap pools)
		 */
		constexpr explicit breakpoint_pool_by_repeated_pops(std::size_t set_count)
			: leftmost_first_(set_count), rightmost_first_(set_count) {}

		/** @brief Room for `elements` records (breakpoints) without reallocating. */
		constexpr void reserve(std::size_t elements) {
			leftmost_first_.reserve(elements);
			rightmost_first_.reserve(elements);
			weights_.reserve(elements);
		}

		/** @brief The number of sets. */
		[[nodiscard]] constexpr std::size_t set_count() const noexcept { return leftmost_first_.heap_count(); }

		/** @brief A new empty set; returns its id. */
		constexpr std::size_t add_set() {
			rightmost_first_.add_heap();
			return leftmost_first_.add_heap();
		}

		/**
		 * @brief A new record `(position, weight)` in both heaps of `set`.
		 * @pre fewer than `2^32 - 1` records (asserted)
		 */
		constexpr void push(std::size_t set, Position const &position, Slope weight) {
			uint32 const heap = heap_of(set);
			atl_assert(weights_.size() < std::numeric_limits<uint32>::max());
			auto const record = uint32(weights_.size());
			weights_.push_back(weight);
			leftmost_first_.push(heap, entry{position, record});
			rightmost_first_.push(heap, entry{position, record});
		}

		/** @brief Drop stale entries off one end of `set`; returns whether a live breakpoint is left. */
		template<bool IsFromLeft>
		constexpr bool has_live(std::size_t set) {
			uint32 const heap = heap_of(set);
			auto &h = heaps<IsFromLeft>();
			while (!h.empty(heap)) {
				if (weights_[h.top(heap).record] != 0) return true;
				h.pop(heap);
			}
			return false;
		}

		/** @brief The end entry of `set`. @pre `has_live<IsFromLeft>(set)` just returned true (documented). */
		template<bool IsFromLeft>
		[[nodiscard]] constexpr entry const &top(std::size_t set) {
			return heaps<IsFromLeft>().top(heap_of(set));
		}

		/** @brief Remove the end entry of `set` from that end's heap only. */
		template<bool IsFromLeft>
		constexpr entry pop(std::size_t set) {
			return heaps<IsFromLeft>().pop(heap_of(set));
		}

		/** @brief The weight of `record`. */
		[[nodiscard]] constexpr Slope &weight(uint32 record) { return weights_[record]; }

		/**
		 * @brief Eat `amount > 0` of weight from the left end (`IsFromLeft`)
		 *        or the right end of `set`, passing every eaten piece to
		 *        `sink(position, weight)`; returns the position of the last
		 *        breakpoint touched.
		 * @pre `amount` is at most the set's weight (asserted when it runs out)
		 */
		template<bool IsFromLeft, breakpoint_sink<Position, Slope> Sink>
		constexpr Position eat(std::size_t set, Slope amount, Sink &&sink) {
			uint32 const heap = heap_of(set);
			auto &h = heaps<IsFromLeft>();
			for (;;) {
				atl_assert(!h.empty(heap));
				entry const &top = h.top(heap);
				Slope &weight = weights_[top.record];
				if (weight == 0) {
					h.pop(heap);
					continue;
				}
				Position const position = top.position;
				Slope const taken = std::min(weight, amount);
				weight -= taken;
				amount -= taken;
				sink(position, taken);
				if (weight == 0) h.pop(heap);
				if (amount == 0) return position;
			}
		}

		/**
		 * @brief Walking from the left end (`IsFromLeft`) or the right end of
		 *        `set`, the positions where the weight passed first reaches
		 *        `need` and first exceeds it; the set is unchanged (the
		 *        walked entries are popped and pushed back).
		 * @pre `0 < need <` the set's weight (asserted when it runs out)
		 * @complexity O((k + 1) log n) amortized for the `k` entries walked
		 */
		template<bool IsFromLeft>
		[[nodiscard]] constexpr std::pair<Position, Position> locate(std::size_t set, Slope need) {
			uint32 const heap = heap_of(set);
			auto &h = heaps<IsFromLeft>();
			walked_.clear();
			Slope passed = 0;
			std::optional<Position> reach;
			for (;;) {
				[[maybe_unused]] bool const is_live = has_live<IsFromLeft>(set);
				atl_assert(is_live);
				entry const e = h.pop(heap);
				walked_.push_back(e);
				passed += weights_[e.record];
				if (!reach && !(passed < need)) reach = e.position;
				if (need < passed) break;
			}
			for (entry const &e : walked_) h.push(heap, e);
			return {*reach, walked_.back().position};
		}

		/** @brief `position += d` for every breakpoint now in `set`. */
		constexpr void translate(std::size_t set, Position const &d) {
			uint32 const heap = heap_of(set);
			leftmost_first_.add_to_all(heap, d);
			rightmost_first_.add_to_all(heap, d);
		}

		/** @brief Move every breakpoint of `from` into `into` (a no-op when equal). O(1). */
		constexpr void meld(std::size_t into, std::size_t from) {
			leftmost_first_.meld(heap_of(into), heap_of(from));
			rightmost_first_.meld(heap_of(into), heap_of(from));
		}

		/**
		 * @brief Empty `set` by repeated pops.
		 * @complexity O((k + 1) log n) amortized for `k` entries
		 */
		constexpr void clear(std::size_t set) {
			uint32 const heap = heap_of(set);
			while (!leftmost_first_.empty(heap)) leftmost_first_.pop(heap);
			while (!rightmost_first_.empty(heap)) rightmost_first_.pop(heap);
		}

		/**
		 * @brief Visit every live breakpoint of `set` as `visit(position,
		 *        weight)`, in increasing position; the set is unchanged
		 *        (popped and pushed back, stale entries dropped).
		 * @complexity O((k + 1) log n) amortized for `k` entries
		 */
		template<breakpoint_sink<Position, Slope> Visit>
		constexpr void visit(std::size_t set, Visit &&visit) {
			uint32 const heap = heap_of(set);
			walked_.clear();
			while (!leftmost_first_.empty(heap)) {
				entry const e = leftmost_first_.pop(heap);
				if (weights_[e.record] == 0) continue;
				visit(e.position, weights_[e.record]);
				walked_.push_back(e);
			}
			for (entry const &e : walked_) leftmost_first_.push(heap, e);
		}

		/**
		 * @brief Append the breakpoints of `set` by increasing position to
		 *        `out`, equal positions combined (also with `out`'s last when
		 *        it is at the same position), and empty the set.
		 * @complexity O((k + 1) log n) amortized for `k` entries
		 */
		constexpr void take_all(std::size_t set, std::vector<weighted_breakpoint<Position, Slope>> &out) {
			uint32 const heap = heap_of(set);
			while (!leftmost_first_.empty(heap)) {
				entry const top = leftmost_first_.pop(heap);
				Slope &weight = weights_[top.record];
				if (weight == 0) continue;
				if (!out.empty() && !(out.back().position < top.position) && !(top.position < out.back().position)) {
					atl_assert(weight <= std::numeric_limits<Slope>::max() - out.back().weight);
					out.back().weight += weight;
				} else {
					out.push_back({top.position, weight});
				}
				weight = 0;
			}
			while (!rightmost_first_.empty(heap)) rightmost_first_.pop(heap);
		}
	};
}
