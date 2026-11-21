#pragma once

/**
 * @file
 * @brief Contenders for @ref atl::convex_piecewise_linear (the slope-trick
 *        function in two heaps), each named for the one thing it does
 *        differently.
 *
 * - @ref convex_piecewise_linear_by_priority_queue — the textbook form on two
 *   `std::priority_queue`s, where adding a hinge that moves the minimum is a
 *   pop and a push on one heap and a push on the other. The library replaces
 *   the top in place with one sift-down instead of the pop and push. A race
 *   contender only: it has the unit update and read interface the race
 *   exercises, not `reserve` or `operator()`.
 * - @ref convex_piecewise_linear_by_breakpoint_pool — the same function on the
 *   engine of @ref atl::convex_piecewise_linear_forest, split at the minimum
 *   into two sets of its breakpoint pool.
 * - @ref convex_piecewise_linear_by_unit_heaps — unit breakpoints only, bare
 *   positions in two vectors without the weighted machinery: the baseline of
 *   what supporting weights costs the library's unit mode.
 * - @ref convex_piecewise_linear_by_configuration, with the aliases
 *   @ref convex_piecewise_linear_by_floyd_replace_top and
 *   @ref convex_piecewise_linear_by_whole_heap_scan — the library class with
 *   the farthest right breakpoint found by a scan of the whole heap and, as
 *   an option, Floyd's bottom-up `replace_top`.
 *
 * Race of 2026-10-15 (benchmarks/dp/convex_piecewise_linear.cpp, Xeon
 * Platinum 8280, ms, plain -O2 / -march=native; n = 1e6 operations, a_i
 * uniform in [0, 1e9); both keep heap capacity when cleared; the library
 * measured with unit breakpoints, as its unit mode stores them):
 *
 * | shape | library (in place) | by priority_queue |
 * |------------------------------------------|--------------------|--------|
 * | prefix minimum + \|x - a\| | 74.0 / 73.5 | 91.4 / 92.4 |
 * | window minimum [-1000, 1000] + \|x - a\| | 122 / 128 | 141 / 141 |
 * | random hinges (both heaps grow to n/2) | 67.9 / 69.9 | 74.7 / 74.3 |
 *
 * The priority_queue form lost by 1.24x on the prefix-minimum shape (a
 * rising hinge that moves the minimum costs a pop and a push there, one
 * sift-down in place) and by 1.06x to 1.15x elsewhere.
 *
 * Race of 2026-10-23 (same benchmark, Xeon Platinum 8280, ms, plain -O2 /
 * -march=native; a_i uniform in [0, 1e9), weights in [1, 1000], moving:
 * weights in [1, 10], add w|x - a| then clip_slopes(-speed, speed);
 * "library" is the weighted mode on the weighted shapes and the default unit
 * mode on the unit shapes):
 *
 * | shape | library | by breakpoint pool | by unit heaps |
 * |----------------------------|-------------|-------------|-------------|
 * | prefix min + \|x - a\|, 1e6 | 69.3 / 65.6 | 238 / 237 | 69.5 / 69.7 |
 * | window + \|x - a\|, 1e6 | 115 / 114 | 403 / 396 | 114 / 120 |
 * | random hinges, 1e6 | 64.3 / 61.8 | 195 / 194 | 64.4 / 66.4 |
 * | weighted prefix min, 1e6 | 109 / 108 | 275 / 271 | |
 * | weighted hinges, 1e6 | 85.3 / 85.1 | 204 / 200 | |
 * | moving speed 100, 2e5 | 28.1 / 29.2 | 58.5 / 59.8 | |
 * | moving speed 1e4, 2e5 | 1410 / 1412 | 42.9 / 42.8 | |
 *
 * The unit heaps are within 1% of the library's unit mode at plain -O2: the
 * weighted machinery costs the unit shapes nothing measurable. The breakpoint
 * pool pushes every breakpoint into two pairing heaps and pops pairing-heap
 * roots where the library sifts a vector heap; it reaches the far ends in
 * amortized O(log n), so it wins only when clips eat the far ends of large
 * sides (33x at speed 1e4) and is 2.1x to 3.5x slower on every other shape.
 * To replace the library it had to be within noise on every shape.
 */

#include <atl/dp/convex_piecewise_linear.hpp>
#include <atl/dp/convex_piecewise_linear_forest.hpp>
#include <atl/dp/detail/wide.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <queue>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::experimental {
	namespace detail {
		/**
		 * @brief `minimum + (p - q)`, formed in the library's wide type
		 *        (`atl::detail::dp_wide_t`) as the library forms it: only the
		 *        result has to fit `Value`.
		 */
		template<piecewise_linear_value Value>
		[[nodiscard]] constexpr Value plus_difference(Value const &minimum, Value const &p, Value const &q) {
			using wide = atl::detail::dp_wide_t<Value>;
			return Value(wide(wide(minimum) + wide(wide(p) - wide(q))));
		}

		/** @brief `minimum + weight * (p - q)`, formed in the library's wide type: only the result has to fit `Value`. */
		template<piecewise_linear_value Value, signed_integral Slope>
		[[nodiscard]] constexpr Value plus_scaled_difference(Value const &minimum, Slope weight, Value const &p, Value const &q) {
			using wide = atl::detail::dp_wide_t<Value>;
			return Value(wide(wide(minimum) + wide(wide(weight) * wide(wide(p) - wide(q)))));
		}

		/** @brief `minimum + slope * length`, formed in the library's wide type: only the result has to fit `Value`. */
		template<piecewise_linear_value Value, signed_integral Slope>
		[[nodiscard]] constexpr Value plus_product(Value const &minimum, Slope slope, Value const &length) {
			using wide = atl::detail::dp_wide_t<Value>;
			return Value(wide(wide(minimum) + wide(wide(slope) * wide(length))));
		}
	}

	/**
	 * @brief @ref atl::convex_piecewise_linear on two `std::priority_queue`s:
	 *        same unit interface and semantics, a hinge that moves the minimum
	 *        costs pop + push + push.
	 * @note Not constexpr (`std::priority_queue` is not constexpr in C++23).
	 */
	template<piecewise_linear_value Value>
	class convex_piecewise_linear_by_priority_queue {
	public:
		/** @brief The zero function. */
		convex_piecewise_linear_by_priority_queue() = default;

		/** @brief `f(x) += c`. */
		void add_constant(Value const &c) { minimum_ += c; }

		/** @brief `f(x) += max(0, x - a)`. */
		void add_rising_hinge(Value const a) { // by value: `a` may be the minimum
			if (!left_.empty() && a < left_.top() + left_offset_) {
				Value const top = left_.top() + left_offset_;
				minimum_ = detail::plus_difference(minimum_, top, a);
				left_.pop();
				left_.push(a - left_offset_);
				right_.push(top - right_offset_);
			} else {
				right_.push(a - right_offset_);
			}
		}

		/** @brief `f(x) += max(0, a - x)`. */
		void add_falling_hinge(Value const a) {
			if (!right_.empty() && right_.top() + right_offset_ < a) {
				Value const top = right_.top() + right_offset_;
				minimum_ = detail::plus_difference(minimum_, a, top);
				right_.pop();
				right_.push(a - right_offset_);
				left_.push(top - left_offset_);
			} else {
				left_.push(a - left_offset_);
			}
		}

		/** @brief `f(x) += |x - a|`. */
		void add_absolute(Value const &a) {
			Value const point = a; // `a` may be the minimum, which the first hinge changes
			add_rising_hinge(point);
			add_falling_hinge(point);
		}

		/** @brief As @ref atl::convex_piecewise_linear::take_prefix_minimum; the emptied side's offset resets, as there. */
		slope_clip_interval<Value> take_prefix_minimum() {
			slope_clip_interval<Value> range;
			if (!right_.empty()) range.upper = right_.top() + right_offset_;
			right_.clear();
			right_offset_ = Value{};
			return range;
		}

		/** @brief As @ref atl::convex_piecewise_linear::take_suffix_minimum; the emptied side's offset resets, as there. */
		slope_clip_interval<Value> take_suffix_minimum() {
			slope_clip_interval<Value> range;
			if (!left_.empty()) range.lower = left_.top() + left_offset_;
			left_.clear();
			left_offset_ = Value{};
			return range;
		}

		/** @brief `f(x) := f(x - d)`. */
		void translate(Value const &d) {
			left_offset_ += d;
			right_offset_ += d;
		}

		/** @brief As @ref atl::convex_piecewise_linear::take_window_minimum; `lo <= hi` asserted. */
		std::optional<Value> take_window_minimum(Value const &lo, Value const &hi) {
			atl_assert(!(hi < lo));
			std::optional<Value> crossing;
			if (!left_.empty()) crossing = left_.top() + left_offset_;
			left_offset_ += lo;
			right_offset_ += hi;
			return crossing;
		}

		/** @brief The minimum value. */
		[[nodiscard]] Value const &minimum() const { return minimum_; }

		/** @brief The minimizer interval. */
		[[nodiscard]] minimizer_interval<Value> minimizers() const {
			minimizer_interval<Value> range;
			if (!left_.empty()) range.lower = left_.top() + left_offset_;
			if (!right_.empty()) range.upper = right_.top() + right_offset_;
			return range;
		}

	private:
		/** @brief A `std::priority_queue` that empties without giving its storage back. */
		template<typename Compare>
			requires std::strict_weak_order<Compare const &, Value const &, Value const &>
		struct heap : std::priority_queue<Value, std::vector<Value>, Compare> {
			void clear() { this->c.clear(); }
		};

		heap<std::less<>> left_;
		heap<std::greater<>> right_;
		Value left_offset_{}, right_offset_{}, minimum_{};
	};

	/**
	 * @brief @ref atl::convex_piecewise_linear on the engine of
	 *        @ref atl::convex_piecewise_linear_forest — the breakpoints left
	 *        and right of the minimum are two sets of one
	 *        `detail::breakpoint_pool` (each breakpoint a weight record in a
	 *        min-heap and a max-heap of pairing heaps), plus the tracked
	 *        minimum. Same interface and semantics; the far ends are reachable,
	 *        so a clip is amortized O(log n) per breakpoint removed, but every
	 *        breakpoint is pushed into two pairing heaps.
	 *
	 * Two sets, not one forest function: a forest function keeps no split at
	 * the minimum, so finding the minimizers (for the minimum value of a
	 * hinge that lands across it) would walk from an end.
	 *
	 * `minimizers()` and `f(x)` are const but drop stale entries or pop and
	 * push back (the pool is `mutable`); `f(x)` is O(n log n) amortized.
	 *
	 * Lost the race of 2026-10-23 (file header): 2.1x to 3.5x slower than
	 * the library on every shape except a clip of large sides every step
	 * (speed 1e4: 43 vs 1410 ms), so it did not replace it.
	 */
	template<piecewise_linear_value Value, signed_integral Slope = int64>
		requires scalable_by<Value, Slope>
	class convex_piecewise_linear_by_breakpoint_pool {
		using pool_type = atl::detail::breakpoint_pool<Value, Slope>;

		static constexpr std::size_t left_set = 0;
		static constexpr std::size_t right_set = 1;

		template<bool IsLeft>
		static constexpr std::size_t set_of = IsLeft ? left_set : right_set;

	public:
		/** @brief The coordinate and value type. */
		using value_type = Value;
		/** @brief The slope and weight type. */
		using slope_type = Slope;

		/** @brief The zero function. */
		constexpr convex_piecewise_linear_by_breakpoint_pool() = default;

		/** @brief Reserve room for `elements` breakpoint records. */
		constexpr void reserve(std::size_t elements) { pool_.reserve(elements); }

		/** @brief `f(x) += c`. */
		constexpr void add_constant(Value const &c) { minimum_ += c; }

		/** @brief `f(x) += weight * max(0, x - a)`; `weight >= 0` asserted. */
		constexpr void add_rising_hinge(Value const &a, Slope weight = Slope{1}) { add_hinge<true>(a, weight); }

		/** @brief `f(x) += weight * max(0, a - x)`; `weight >= 0` asserted. */
		constexpr void add_falling_hinge(Value const &a, Slope weight = Slope{1}) { add_hinge<false>(a, weight); }

		/** @brief `f(x) += weight * |x - a|`; `weight >= 0` asserted. */
		constexpr void add_absolute(Value const &a, Slope weight = Slope{1}) {
			Value const point = a; // `a` may be the minimum, which the first hinge changes
			add_hinge<true>(point, weight);
			add_hinge<false>(point, weight);
		}

		/** @brief `f(x) := f(x - d)`. */
		constexpr void translate(Value const &d) {
			pool_.translate(left_set, d);
			pool_.translate(right_set, d);
		}

		/** @brief As @ref atl::convex_piecewise_linear::clip_slopes; far ends eaten in amortized O(log n) each. */
		constexpr slope_clip_interval<Value> clip_slopes(Slope min_slope, Slope max_slope) {
			atl_assert(min_slope <= 0 && 0 <= max_slope);
			auto const ignore = [](Value const &, Slope const &) {};
			slope_clip_interval<Value> range;
			if (min_slope == 0)
				range.lower = take_suffix_minimum().lower;
			else if (Slope const excess = left_weight_ + min_slope; excess > 0) {
				range.lower = pool_.template eat<true>(left_set, excess, ignore);
				left_weight_ = -min_slope;
			}
			if (max_slope == 0)
				range.upper = take_prefix_minimum().upper;
			else if (Slope const excess = right_weight_ - max_slope; excess > 0) {
				range.upper = pool_.template eat<false>(right_set, excess, ignore);
				right_weight_ = max_slope;
			}
			return range;
		}

		/** @brief As @ref atl::convex_piecewise_linear::take_prefix_minimum. */
		constexpr slope_clip_interval<Value> take_prefix_minimum() {
			slope_clip_interval<Value> range{std::nullopt, nearest<false>()};
			pool_.clear(right_set);
			right_weight_ = 0;
			return range;
		}

		/** @brief As @ref atl::convex_piecewise_linear::take_suffix_minimum. */
		constexpr slope_clip_interval<Value> take_suffix_minimum() {
			slope_clip_interval<Value> range{nearest<true>(), std::nullopt};
			pool_.clear(left_set);
			left_weight_ = 0;
			return range;
		}

		/** @brief As @ref atl::convex_piecewise_linear::take_window_minimum; `lo <= hi` asserted. */
		constexpr std::optional<Value> take_window_minimum(Value const &lo, Value const &hi) {
			atl_assert(!(hi < lo));
			std::optional<Value> const crossing = nearest<true>();
			pool_.translate(left_set, lo);
			pool_.translate(right_set, hi);
			return crossing;
		}

		/**
		 * @brief As @ref atl::convex_piecewise_linear::insert_segment;
		 *        `slope == rightmost_slope()` is amortized O(log n) here.
		 */
		constexpr std::optional<Value> insert_segment(Slope slope, Value const &segment_length) {
			Value const length = segment_length;
			atl_assert(!(length < Value{}));
			atl_assert(slope <= right_weight_);
			if (slope <= -left_weight_) {
				translate(length);
				minimum_ = detail::plus_product(minimum_, slope, length);
				return std::nullopt;
			}
			if (slope == 0) {
				std::optional<Value> const crossing = nearest<true>();
				pool_.translate(right_set, length);
				return crossing;
			}
			if (slope == right_weight_) {
				[[maybe_unused]] bool const is_live = pool_.template has_live<false>(right_set);
				atl_assert(is_live);
				return pool_.template top<false>(right_set).position;
			}
			moved_.clear();
			auto const collect = [this](Value const &at, Slope const &weight) { moved_.push_back({at, weight}); };
			if (slope > 0) {
				Value const crossing = pool_.template eat<true>(right_set, slope, collect);
				pool_.translate(right_set, length);
				for (auto const &[position, weight] : moved_) pool_.push(right_set, position, weight);
				return crossing;
			}
			Value const crossing = pool_.template eat<false>(left_set, -slope, collect);
			for (auto const &[position, weight] : moved_) pool_.push(left_set, position + length, weight);
			pool_.translate(right_set, length);
			minimum_ = detail::plus_product(minimum_, slope, length);
			return crossing;
		}

		/** @brief The slope at minus infinity. */
		[[nodiscard]] constexpr Slope leftmost_slope() const { return -left_weight_; }

		/** @brief The slope at plus infinity. */
		[[nodiscard]] constexpr Slope rightmost_slope() const { return right_weight_; }

		/**
		 * @brief As @ref atl::convex_piecewise_linear::take_breakpoints (the
		 *        combined weight at each position must fit `Slope`, asserted).
		 */
		constexpr std::vector<weighted_breakpoint<Value, Slope>> take_breakpoints() {
			std::vector<weighted_breakpoint<Value, Slope>> breakpoints;
			pool_.take_all(left_set, breakpoints);
			pool_.take_all(right_set, breakpoints);
			left_weight_ = right_weight_ = 0;
			minimum_ = Value{};
			return breakpoints;
		}

		/** @brief The minimum value. */
		[[nodiscard]] constexpr Value const &minimum() const { return minimum_; }

		/** @brief The minimizer interval (drops stale entries off the near tops). */
		[[nodiscard]] constexpr minimizer_interval<Value> minimizers() const { return {nearest<true>(), nearest<false>()}; }

		/**
		 * @brief `f(x)`, O(n log n) amortized (every breakpoint popped and
		 *        pushed back); the terms are summed in the library's wide type.
		 */
		[[nodiscard]] constexpr Value operator()(Value const &x) const {
			using wide = atl::detail::dp_wide_t<Value>;
			wide result = wide(minimum_);
			pool_.visit(left_set, [&](Value const &a, Slope const &weight) {
				if (x < a) result = wide(result + wide(wide(weight) * wide(wide(a) - wide(x))));
			});
			pool_.visit(right_set, [&](Value const &a, Slope const &weight) {
				if (a < x) result = wide(result + wide(wide(weight) * wide(wide(x) - wide(a))));
			});
			return Value(result);
		}

	private:
		/** @brief The breakpoint nearest the minimum on the left (`IsLeft`) or right side, if any. */
		template<bool IsLeft>
		[[nodiscard]] constexpr std::optional<Value> nearest() const {
			// the near end of the left side is its right end
			if (!pool_.template has_live<!IsLeft>(set_of<IsLeft>)) return std::nullopt;
			return pool_.template top<!IsLeft>(set_of<IsLeft>).position;
		}

		/** @brief A new breakpoint on the left (`IsLeft`) or right side, merged into the near top at its position. */
		template<bool IsLeft>
		constexpr void push_near(Value const &position, Slope weight) {
			if (pool_.template has_live<!IsLeft>(set_of<IsLeft>))
				if (auto const &top = pool_.template top<!IsLeft>(set_of<IsLeft>); top.position == position) {
					pool_.weight(top.record) += weight;
					return;
				}
			pool_.push(set_of<IsLeft>, position, weight);
		}

		/** @brief As the library: the hinge's weight crosses from the near top of the side it rises out of. */
		template<bool IsRising>
		constexpr void add_hinge(Value const a, Slope weight) { // by value: `a` may be the minimum
			atl_assert(weight >= 0);
			constexpr std::size_t from = set_of<IsRising>;
			(IsRising ? right_weight_ : left_weight_) += weight;
			Slope rest = weight;
			while (rest != 0 && pool_.template has_live<!IsRising>(from)) {
				auto const top = pool_.template top<!IsRising>(from);
				bool const is_past = IsRising ? a < top.position : top.position < a;
				if (!is_past) break;
				Slope const available = pool_.weight(top.record);
				Slope const moved = available < rest ? available : rest;
				if constexpr (IsRising)
					minimum_ = detail::plus_scaled_difference(minimum_, moved, top.position, a);
				else
					minimum_ = detail::plus_scaled_difference(minimum_, moved, a, top.position);
				pool_.weight(top.record) = available - moved;
				if (available == moved) pool_.template pop<!IsRising>(from);
				push_near<!IsRising>(top.position, moved);
				rest -= moved;
			}
			if (rest != weight) push_near<IsRising>(a, weight - rest);
			if (rest != 0) push_near<!IsRising>(a, rest);
		}

		mutable pool_type pool_{2};
		Slope left_weight_{}, right_weight_{};
		Value minimum_{};
		/** @brief Scratch for the pieces `insert_segment` moves. */
		std::vector<weighted_breakpoint<Value, Slope>> moved_;
	};

	/**
	 * @brief @ref atl::convex_piecewise_linear with unit breakpoints only:
	 *        heaps of bare positions in two vectors, without the template on
	 *        `Slope` and the side weights. The baseline of what the weighted
	 *        machinery costs the library's unit mode (within 1% at plain
	 *        -O2, 2026-10-23 race in the file header); it has only the unit
	 *        update and read interface the race exercises.
	 */
	template<piecewise_linear_value Value>
	class convex_piecewise_linear_by_unit_heaps {
	public:
		/** @brief The zero function. */
		constexpr convex_piecewise_linear_by_unit_heaps() = default;

		/** @brief Reserve room for `elements` breakpoints on each side. */
		constexpr void reserve(std::size_t elements) {
			left_.reserve(elements);
			right_.reserve(elements);
		}

		/** @brief `f(x) += c`. */
		constexpr void add_constant(Value const &c) { minimum_ += c; }

		/** @brief `f(x) += max(0, x - a)`. */
		constexpr void add_rising_hinge(Value const a) { // by value: `a` may be the minimum
			if (!left_.empty()) {
				Value const top = left_.front() + left_offset_;
				if (a < top) {
					minimum_ = detail::plus_difference(minimum_, top, a);
					replace_top(left_, a - left_offset_, std::less<>{});
					push(right_, top - right_offset_, std::greater<>{});
					return;
				}
			}
			push(right_, a - right_offset_, std::greater<>{});
		}

		/** @brief `f(x) += max(0, a - x)`. */
		constexpr void add_falling_hinge(Value const a) {
			if (!right_.empty()) {
				Value const top = right_.front() + right_offset_;
				if (top < a) {
					minimum_ = detail::plus_difference(minimum_, a, top);
					replace_top(right_, a - right_offset_, std::greater<>{});
					push(left_, top - left_offset_, std::less<>{});
					return;
				}
			}
			push(left_, a - left_offset_, std::less<>{});
		}

		/** @brief `f(x) += |x - a|`. */
		constexpr void add_absolute(Value const &a) {
			Value const point = a; // `a` may be the minimum, which the first hinge changes
			add_rising_hinge(point);
			add_falling_hinge(point);
		}

		/** @brief As @ref atl::convex_piecewise_linear::take_prefix_minimum. */
		constexpr slope_clip_interval<Value> take_prefix_minimum() {
			slope_clip_interval<Value> range;
			if (!right_.empty()) range.upper = right_.front() + right_offset_;
			right_.clear();
			right_offset_ = Value{};
			return range;
		}

		/** @brief As @ref atl::convex_piecewise_linear::take_suffix_minimum. */
		constexpr slope_clip_interval<Value> take_suffix_minimum() {
			slope_clip_interval<Value> range;
			if (!left_.empty()) range.lower = left_.front() + left_offset_;
			left_.clear();
			left_offset_ = Value{};
			return range;
		}

		/** @brief `f(x) := f(x - d)`. */
		constexpr void translate(Value const &d) {
			left_offset_ += d;
			right_offset_ += d;
		}

		/** @brief As @ref atl::convex_piecewise_linear::take_window_minimum; `lo <= hi` asserted. */
		constexpr std::optional<Value> take_window_minimum(Value const &lo, Value const &hi) {
			atl_assert(!(hi < lo));
			std::optional<Value> crossing;
			if (!left_.empty()) crossing = left_.front() + left_offset_;
			left_offset_ += lo;
			right_offset_ += hi;
			return crossing;
		}

		/** @brief The minimum value. */
		[[nodiscard]] constexpr Value const &minimum() const { return minimum_; }

		/** @brief The minimizer interval. */
		[[nodiscard]] constexpr minimizer_interval<Value> minimizers() const {
			minimizer_interval<Value> range;
			if (!left_.empty()) range.lower = left_.front() + left_offset_;
			if (!right_.empty()) range.upper = right_.front() + right_offset_;
			return range;
		}

	private:
		template<typename Compare>
			requires std::strict_weak_order<Compare const &, Value const &, Value const &>
		static constexpr void push(std::vector<Value> &heap, Value value, Compare comp) {
			heap.push_back(std::move(value));
			std::ranges::push_heap(heap, comp);
		}

		/** @brief Replace the top of the non-empty heap ordered by `comp` with `value` by one sift-down. */
		template<typename Compare>
			requires std::strict_weak_order<Compare const &, Value const &, Value const &>
		static constexpr void replace_top(std::vector<Value> &heap, Value value, Compare comp) {
			std::size_t const n = heap.size();
			std::size_t hole = 0;
			for (std::size_t child = 1; child < n; child = 2 * hole + 1) {
				if (child + 1 < n && comp(heap[child], heap[child + 1])) ++child;
				if (!comp(value, heap[child])) break;
				heap[hole] = std::move(heap[child]);
				hole = child;
			}
			heap[hole] = std::move(value);
		}

		std::vector<Value> left_, right_;
		Value left_offset_{}, right_offset_{}, minimum_{};
	};

	/**
	 * @brief @ref atl::convex_piecewise_linear with the farthest right
	 *        breakpoint (an `insert_segment` at the rightmost slope) found by
	 *        `std::ranges::max` over the whole heap instead of its leaves, and
	 *        with `UsesFloydReplaceTop` Floyd's bottom-up `replace_top` (to a
	 *        leaf along the larger children, then up) instead of the sift-down.
	 *        Otherwise the library's code; the same interface, so the tests
	 *        certify it against the library and both aliases below.
	 *
	 * Record of 2026-11-01 (benchmarks/dp/convex_piecewise_linear.cpp, the
	 * tuning race; Xeon Platinum 8280, GCC 15.2.0, ms, plain -O2 /
	 * -march=native). The whole-heap scan lost on rightmost-slope segments
	 * (about 1e4 right entries, 2000 lookups): unit 12.2 / 17.7 against the
	 * leaf scan's 7.3 / 9.5, weighted 12.5 / 17.9 against 10.0 / 9.9; the
	 * leaf scan reads half the entries. Floyd's `replace_top` lost on unit
	 * random hinges, n = 1e6: 68.5 / 68.7 against 67.1 / 64.7 (+2% / +6%),
	 * despite wins on the prefix-minimum and window shapes (69.8 against
	 * 76.7 and 117 against 124 at plain -O2); a change that costs a shape
	 * is not taken.
	 */
	template<piecewise_linear_value Value, piecewise_linear_slope Slope = unit_weight, bool UsesFloydReplaceTop = true>
		requires scalable_by<Value, atl::detail::slope_count_t<Slope>>
	class convex_piecewise_linear_by_configuration {
	public:
		/** @brief The coordinate and value type. */
		using value_type = Value;
		/** @brief The type of a position (a breakpoint, an interval end): the same as @ref value_type, spelled as on @ref atl::convex_piecewise_linear_forest. */
		using position_type = Value;
		/** @brief The type slopes and weights are reported in (`int64` in unit mode). */
		using slope_type = atl::detail::slope_count_t<Slope>;
		/** @brief Whether breakpoints are unit (bare positions, hinges without a weight). */
		static constexpr bool is_unit = std::same_as<Slope, unit_weight>;

	private:
		/** @brief The library's wide type for intermediates. */
		using wide = atl::detail::dp_wide_t<Value>;

		/** @brief A stored breakpoint: its position minus the side's offset, and in weighted mode its weight. */
		using entry = std::conditional_t<is_unit, Value, weighted_breakpoint<Value, slope_type>>;

		/** @brief One side of the minimum: a heap of stored breakpoints and the lazy offset. */
		struct side {
			std::vector<entry> heap;
			Value offset{};
			/**
			 * @brief The total weight in weighted mode: minus the slope at minus
			 *        infinity (left), the slope at plus infinity (right). Unit
			 *        mode counts the heap instead.
			 */
			slope_type weight{};
		};

	public:
		/** @brief The zero function. */
		constexpr convex_piecewise_linear_by_configuration() = default;

		/** @brief Reserve room for `elements` stored breakpoints (the unit is breakpoints) on each side. */
		constexpr void reserve(std::size_t elements) {
			left_.heap.reserve(elements);
			right_.heap.reserve(elements);
		}

		/** @brief `f(x) += c`. O(1). */
		constexpr void add_constant(Value const &c) { minimum_ = Value(minimum_ + c); }

		/**
		 * @brief `f(x) += max(0, x - a)` (unit mode): a hinge rising to the right of `a`.
		 * @complexity O(log n) for `n` breakpoints
		 */
		constexpr void add_rising_hinge(Value const &a)
			requires is_unit
		{
			add_unit_hinge<true>(a);
		}

		/**
		 * @brief `f(x) += weight * max(0, x - a)` (weighted mode).
		 * @pre `weight >= 0` (asserted); a zero weight changes nothing
		 * @complexity O((k + 1) log n) for `n` breakpoints and the `k` the hinge
		 *      carries across the minimum (`k <= weight`; 1 for a unit hinge)
		 */
		constexpr void add_rising_hinge(Value const &a, slope_type weight = slope_type{1})
			requires(!is_unit)
		{
			add_weighted_hinge<true>(a, weight);
		}

		/**
		 * @brief `f(x) += max(0, a - x)` (unit mode): a hinge rising to the left of `a`.
		 * @complexity O(log n)
		 */
		constexpr void add_falling_hinge(Value const &a)
			requires is_unit
		{
			add_unit_hinge<false>(a);
		}

		/**
		 * @brief `f(x) += weight * max(0, a - x)` (weighted mode).
		 * @pre `weight >= 0` (asserted); a zero weight changes nothing
		 * @complexity as the weighted @ref add_rising_hinge
		 */
		constexpr void add_falling_hinge(Value const &a, slope_type weight = slope_type{1})
			requires(!is_unit)
		{
			add_weighted_hinge<false>(a, weight);
		}

		/**
		 * @brief `f(x) += |x - a|` (unit mode).
		 * @complexity O(log n)
		 */
		constexpr void add_absolute(Value const &a)
			requires is_unit
		{
			Value const point = a; // `a` may be the minimum, which the first hinge changes
			add_unit_hinge<true>(point);
			add_unit_hinge<false>(point);
		}

		/**
		 * @brief `f(x) += weight * |x - a|` (weighted mode).
		 * @pre `weight >= 0` (asserted); a zero weight changes nothing
		 * @complexity as the weighted @ref add_rising_hinge
		 */
		constexpr void add_absolute(Value const &a, slope_type weight = slope_type{1})
			requires(!is_unit)
		{
			Value const point = a;
			add_weighted_hinge<true>(point, weight);
			add_weighted_hinge<false>(point, weight);
		}

		/** @brief `f(x) := f(x - d)`: the graph moves `d` to the right. O(1). */
		constexpr void translate(Value const &d) {
			left_.offset = Value(left_.offset + d);
			right_.offset = Value(right_.offset + d);
		}

		/**
		 * @brief Clip the slopes of `f` into `[min_slope, max_slope]`: `f :=
		 *        min_y f(y) + k(x - y)` with `k(t) = max_slope * t` for `t >= 0`
		 *        and `min_slope * t` for `t <= 0` (with `-D` and `D`, the
		 *        infimal convolution with `D|t|`). The part of the graph whose
		 *        slopes are inside stays in place, the minimum value too;
		 *        steeper parts are replaced by rays of the limiting slopes.
		 *
		 * Returns where the slopes were unchanged: its `clamp(x)` is an optimal
		 * `y` above. An end is present exactly when that side was clipped
		 * (`min_slope > leftmost_slope()`, `max_slope < rightmost_slope()`);
		 * `clip_slopes(0, 0)` returns the minimizers (and flattens `f`).
		 *
		 * @pre `min_slope <= 0 <= max_slope` (asserted): `f` keeps a minimum
		 * @complexity O(1) for a side clipped to slope 0 or not at all; else,
		 *      for the `m` entries of that side, O(m) per entry eaten for the
		 *      first four, then one O(m log m) sort
		 */
		constexpr slope_clip_interval<Value> clip_slopes(slope_type min_slope, slope_type max_slope) {
			atl_assert(min_slope <= 0 && 0 <= max_slope);
			slope_clip_interval<Value> range;
			if (min_slope == 0)
				range.lower = take_suffix_minimum().lower;
			else if (slope_type const excess = slope_type(total(left_) + min_slope); excess > 0) {
				range.lower = eat_far<true>(left_, excess);
				if constexpr (!is_unit) left_.weight = slope_type(-min_slope);
			}
			if (max_slope == 0)
				range.upper = take_prefix_minimum().upper;
			else if (slope_type const excess = slope_type(total(right_) - max_slope); excess > 0) {
				range.upper = eat_far<false>(right_, excess);
				if constexpr (!is_unit) right_.weight = max_slope;
			}
			return range;
		}

		/**
		 * @brief `f(x) := min over y <= x of f(y)`: every slope right of the
		 *        minimum becomes 0 (`clip_slopes(leftmost_slope(), 0)`).
		 * @return `{none, upper}`: `upper` the largest minimizer before the
		 *         call (none if `f` was already nonincreasing), so `clamp(x)`
		 *         is an optimal `y`.
		 * @complexity O(1)
		 */
		constexpr slope_clip_interval<Value> take_prefix_minimum() {
			slope_clip_interval<Value> range;
			if (!right_.heap.empty()) range.upper = top_position(right_);
			clear(right_);
			return range;
		}

		/**
		 * @brief `f(x) := min over y >= x of f(y)`: every slope left of the
		 *        minimum becomes 0 (`clip_slopes(0, rightmost_slope())`).
		 * @return `{lower, none}`: `lower` the smallest minimizer before the
		 *         call (none if `f` was already nondecreasing).
		 * @complexity O(1)
		 */
		constexpr slope_clip_interval<Value> take_suffix_minimum() {
			slope_clip_interval<Value> range;
			if (!left_.heap.empty()) range.lower = top_position(left_);
			clear(left_);
			return range;
		}

		/**
		 * @brief `f(x) := min over lo <= d <= hi of f(x - d)`, i.e. the minimum
		 *        of `f` over the window `[x - hi, x - lo]`: the decreasing
		 *        part moves right by `lo`, the increasing part by `hi`, and
		 *        the flat bottom widens by `hi - lo` (`insert_segment(0, hi -
		 *        lo)`, then `translate(lo)`).
		 * @return The smallest minimizer `p` of `f` before the call
		 *         (`minimizers().lower`), or none when `f` was nondecreasing
		 *         (then read `p` as minus infinity): an optimal `y` for `x` is
		 *         `clamp(p, x - hi, x - lo)`.
		 * @pre `lo <= hi` (asserted)
		 * @complexity O(1)
		 */
		constexpr std::optional<Value> take_window_minimum(Value const &lo, Value const &hi) {
			atl_assert(!(hi < lo));
			std::optional<Value> crossing;
			if (!left_.heap.empty()) crossing = top_position(left_);
			left_.offset = Value(left_.offset + lo);
			right_.offset = Value(right_.offset + hi);
			return crossing;
		}

		/**
		 * @brief Insert a segment of slope `slope` and length `length` into the
		 *        graph where `f`'s slope rises through `slope`: the part with
		 *        slopes `>= slope` moves `length` to the right, i.e. `f :=
		 *        min_{0 <= t <= length} f(x - t) + slope * t` (the minimum value
		 *        changes by `slope * length` when `slope < 0`).
		 *
		 * Returns a point `p` where the slope of `f` rises through `slope` (`f`
		 * has slopes `<= slope` left of `p`, `>= slope` right of it), or none
		 * when `slope <= leftmost_slope()` (then the whole graph moves; read
		 * `p` as minus infinity): an optimal `t` for `x` is `x - clamp(p, x -
		 * length, x)`. When a stretch of slope exactly `slope` makes several
		 * points qualify, `p` is the one nearest the minimum: the smallest
		 * minimizer for `slope == 0`, else the end of the stretch on the
		 * minimum's side (the slope is `< slope` just left of `p` when
		 * `slope > 0`, `> slope` just right of it when `slope < 0`).
		 *
		 * @pre `length >= 0` and `slope <= rightmost_slope()` (asserted)
		 * @complexity O(1) for `slope == 0` or `slope <= leftmost_slope()`;
		 *      O((k + 1) log n) for the `k` breakpoints between the minimum and
		 *      `p`; O(m) for `slope == rightmost_slope()` (a scan of the `m`
		 *      entries right of the minimum for the farthest)
		 */
		constexpr std::optional<Value> insert_segment(slope_type slope, Value const &segment_length) {
			Value const length = segment_length;
			atl_assert(!(length < Value{}));
			atl_assert(slope <= total(right_));
			if (slope <= -total(left_)) {
				translate(length);
				minimum_ = detail::plus_product(minimum_, slope, length);
				return std::nullopt;
			}
			if (slope == 0) {
				right_.offset = Value(right_.offset + length);
				return top_position(left_);
			}
			if (slope == total(right_)) {
				// nothing is steeper, so the graph is unchanged and `p` is the farthest breakpoint
				auto const &farthest = std::ranges::max(right_.heap, {}, by_position{});
				return Value(position_of(farthest) + right_.offset);
			}
			moved_.clear();
			if (slope > 0) {
				// the pieces up to slope `slope` stay; the offset moves the rest
				Value const crossing = eat_near<false>(right_, slope);
				right_.offset = Value(right_.offset + length);
				for (auto const &[position, weight] : moved_) push<false>(right_, position, weight);
				return crossing;
			}
			Value const crossing = eat_near<true>(left_, slope_type(-slope));
			for (auto const &[position, weight] : moved_) push<true>(left_, Value(position + length), weight);
			right_.offset = Value(right_.offset + length);
			minimum_ = detail::plus_product(minimum_, slope, length);
			return crossing;
		}

		/** @brief The slope of `f` at minus infinity (left of every breakpoint); `<= 0`. */
		[[nodiscard]] constexpr slope_type leftmost_slope() const { return slope_type(-total(left_)); }

		/** @brief The slope of `f` at plus infinity (right of every breakpoint); `>= 0`. */
		[[nodiscard]] constexpr slope_type rightmost_slope() const { return total(right_); }

		/**
		 * @brief The breakpoints of `f` by increasing position, equal positions
		 *        combined (in unit mode too: the weight counts them), and `f`
		 *        reset to the zero function (read the slopes and the minimum
		 *        beforehand).
		 * @pre the combined weight at every position fits `slope_type` (asserted)
		 * @complexity O(n log n)
		 */
		constexpr std::vector<weighted_breakpoint<Value, slope_type>> take_breakpoints() {
			std::vector<weighted_breakpoint<Value, slope_type>> breakpoints;
			breakpoints.reserve(left_.heap.size() + right_.heap.size());
			// every left breakpoint is at most every right one, so the sides concatenate
			for (side *s : {&left_, &right_}) {
				std::ranges::sort(s->heap, {}, by_position{});
				for (entry const &e : s->heap) {
					Value const position = Value(position_of(e) + s->offset);
					if (!breakpoints.empty() && !(breakpoints.back().position < position)) {
						atl_assert(weight_of(e) <= std::numeric_limits<slope_type>::max() - breakpoints.back().weight);
						breakpoints.back().weight = slope_type(breakpoints.back().weight + weight_of(e));
					} else
						breakpoints.push_back({position, weight_of(e)});
				}
				clear(*s);
			}
			minimum_ = Value{};
			return breakpoints;
		}

		/** @brief The minimum value of `f`. O(1). */
		[[nodiscard]] constexpr Value const &minimum() const { return minimum_; }

		/** @brief Where `f` attains its minimum, a closed interval maybe unbounded on either side. O(1). */
		[[nodiscard]] constexpr minimizer_interval<Value> minimizers() const {
			minimizer_interval<Value> range;
			if (!left_.heap.empty()) range.lower = top_position(left_);
			if (!right_.heap.empty()) range.upper = top_position(right_);
			return range;
		}

		/**
		 * @brief `f(x)`, by summing every breakpoint's hinge.
		 * @complexity O(n) for `n` breakpoints
		 */
		[[nodiscard]] constexpr Value operator()(Value const &x) const {
			wide result = wide(minimum_);
			wide const at = wide(x);
			for (entry const &e : left_.heap)
				if (Value const a = Value(position_of(e) + left_.offset); x < a)
					result = wide(result + scaled(e, wide(wide(a) - at)));
			for (entry const &e : right_.heap)
				if (Value const a = Value(position_of(e) + right_.offset); a < x)
					result = wide(result + scaled(e, wide(at - wide(a))));
			return Value(result);
		}

	private:
		/** @brief The heap order of a side: the top is the breakpoint nearest the minimum. */
		template<bool IsLeft>
		using order = std::conditional_t<is_unit, std::conditional_t<IsLeft, std::less<>, std::greater<>>,
										 std::conditional_t<IsLeft, atl::detail::breakpoint_left_of, atl::detail::breakpoint_right_of>>;

		[[nodiscard]] static constexpr Value const &position_of(entry const &e) {
			if constexpr (is_unit)
				return e;
			else
				return e.position;
		}

		/** @brief The projection of the sorts. */
		struct by_position {
			[[nodiscard]] constexpr Value const &operator()(entry const &e) const { return position_of(e); }
		};

		[[nodiscard]] static constexpr slope_type weight_of(entry const &e) {
			if constexpr (is_unit)
				return slope_type{1};
			else
				return e.weight;
		}

		/** @brief The entry's hinge value at `distance > 0` past it, `weight * distance`, in the wide type. */
		[[nodiscard]] static constexpr wide scaled(entry const &e, wide const &distance) {
			if constexpr (is_unit)
				return distance;
			else
				return wide(wide(e.weight) * distance);
		}

		/** @brief The weight of a side: its heap's size in unit mode. */
		[[nodiscard]] static constexpr slope_type total(side const &s) {
			if constexpr (is_unit) {
				atl_assert(std::in_range<slope_type>(s.heap.size()));
				return slope_type(s.heap.size());
			} else {
				return s.weight;
			}
		}

		/** @brief The true position of the top of a non-empty side. */
		[[nodiscard]] static constexpr Value top_position(side const &s) { return Value(position_of(s.heap.front()) + s.offset); }

		/** @brief The entry storing a breakpoint at true `position` on side `s`, of weight `weight`. */
		[[nodiscard]] static constexpr entry make_entry(side const &s, Value const &position, [[maybe_unused]] slope_type weight) {
			if constexpr (is_unit)
				return Value(position - s.offset);
			else
				return entry{Value(position - s.offset), weight};
		}

		/** @brief Whether `x` lies farther from the minimum than `y` on the left (`IsLeft`) or right side. */
		template<bool IsLeft>
		[[nodiscard]] static constexpr bool is_farther(Value const &x, Value const &y) {
			if constexpr (IsLeft)
				return x < y;
			else
				return y < x;
		}

		/** @brief Push the breakpoint `(position, weight)` onto a side (`weight` copies in unit mode). */
		template<bool IsLeft>
		static constexpr void push(side &s, Value const &position, slope_type weight) {
			slope_type copies{1};
			if constexpr (is_unit) copies = weight;
			for (; copies != 0; --copies) {
				s.heap.push_back(make_entry(s, position, weight));
				std::ranges::push_heap(s.heap, order<IsLeft>{});
			}
		}

		/**
		 * @brief @ref push, or add the weight to the top when it sits at
		 *        `position` (weighted mode). Every breakpoint a hinge moves
		 *        across the minimum lands at the top of its new side, so this
		 *        keeps a position split between the sides to one entry per
		 *        side; without it a position crossing back and forth fragments
		 *        into many entries that every later crossing walks (weighted
		 *        random hinges ran superlinearly).
		 */
		template<bool IsLeft>
		static constexpr void push_near(side &s, Value const &position, slope_type weight)
			requires(!is_unit)
		{
			if (!s.heap.empty())
				if (entry &top = s.heap.front(); Value(top.position + s.offset) == position) {
					top.weight = slope_type(top.weight + weight);
					return;
				}
			push<IsLeft>(s, position, weight);
		}

		/** @brief Remove the top of a non-empty side. */
		template<bool IsLeft>
		static constexpr void pop(side &s) {
			std::ranges::pop_heap(s.heap, order<IsLeft>{});
			s.heap.pop_back();
		}

		/**
		 * @brief Replace the top of a non-empty side with `value`: one
		 *        sift-down, or with `UsesFloydReplaceTop` a descent to a leaf
		 *        along the larger children without comparing `value`, then a
		 *        sift-up (Floyd's bottom-up variant).
		 */
		template<bool IsLeft>
		static constexpr void replace_top(side &s, entry value) {
			std::vector<entry> &heap = s.heap;
			order<IsLeft> const comp;
			std::size_t const n = heap.size();
			std::size_t hole = 0;
			if constexpr (UsesFloydReplaceTop) {
				for (std::size_t child = 1; child < n; child = 2 * hole + 1) {
					if (child + 1 < n && comp(heap[child], heap[child + 1])) ++child;
					heap[hole] = std::move(heap[child]);
					hole = child;
				}
				while (hole != 0) {
					std::size_t const parent = (hole - 1) / 2;
					if (!comp(heap[parent], value)) break;
					heap[hole] = std::move(heap[parent]);
					hole = parent;
				}
			} else {
				for (std::size_t child = 1; child < n; child = 2 * hole + 1) {
					if (child + 1 < n && comp(heap[child], heap[child + 1])) ++child;
					if (!comp(value, heap[child])) break;
					heap[hole] = std::move(heap[child]);
					hole = child;
				}
			}
			heap[hole] = std::move(value);
		}

		/** @brief Empty a side: its slope at infinity becomes 0. */
		static constexpr void clear(side &s) {
			s.heap.clear();
			s.offset = Value{};
			s.weight = slope_type{};
		}

		/**
		 * @brief Add `max(0, x - a)` (`IsRising`) or `max(0, a - x)` in unit
		 *        mode: if the near top of the side the hinge rises out of (the
		 *        left for a rising hinge) lies past `a`, that unit crosses to
		 *        the other side, the minimum rises by the distance, and `a`
		 *        takes the top's place by one sift-down.
		 */
		template<bool IsRising>
		constexpr void add_unit_hinge(Value const a) // by value: the caller's `a` may be the minimum
			requires is_unit
		{
			side &from = IsRising ? left_ : right_;
			side &to = IsRising ? right_ : left_;
			if (!from.heap.empty()) {
				Value const top = Value(from.heap.front() + from.offset);
				if (is_farther<IsRising>(a, top)) {
					if constexpr (IsRising)
						minimum_ = detail::plus_difference(minimum_, top, a);
					else
						minimum_ = detail::plus_difference(minimum_, a, top);
					replace_top<IsRising>(from, Value(a - from.offset));
					to.heap.push_back(Value(top - to.offset));
					std::ranges::push_heap(to.heap, order<!IsRising>{});
					return;
				}
			}
			to.heap.push_back(Value(a - to.offset));
			std::ranges::push_heap(to.heap, order<!IsRising>{});
		}

		/**
		 * @brief Add `weight * max(0, x - a)` (`IsRising`) or `weight * max(0,
		 *        a - x)` in weighted mode: the hinge's weight crosses from the
		 *        near top of the side it rises out of while that top lies past
		 *        `a`, each crossed unit at `p` raising the minimum by `|p - a|`;
		 *        the crossed weight lands at `a` on that side, the rest of the
		 *        hinge at `a` on the other.
		 */
		template<bool IsRising>
		constexpr void add_weighted_hinge(Value const a, slope_type weight) // by value: `a` may be the minimum
			requires(!is_unit)
		{
			atl_assert(weight >= 0);
			side &from = IsRising ? left_ : right_;
			side &to = IsRising ? right_ : left_;
			to.weight = slope_type(to.weight + weight);
			slope_type rest = weight;
			while (rest != 0 && !from.heap.empty()) {
				entry &top = from.heap.front();
				Value const p = Value(top.position + from.offset);
				if (!is_farther<IsRising>(a, p)) break;
				slope_type const moved = top.weight < rest ? top.weight : rest;
				if constexpr (IsRising)
					minimum_ = detail::plus_scaled_difference(minimum_, moved, p, a);
				else
					minimum_ = detail::plus_scaled_difference(minimum_, moved, a, p);
				push_near<!IsRising>(to, p, moved);
				rest = slope_type(rest - moved);
				if (top.weight != moved) {
					top.weight = slope_type(top.weight - moved);
					break;
				}
				if (rest == 0) {
					// the crossed weight takes the emptied top's place
					replace_top<IsRising>(from, make_entry(from, a, weight));
					return;
				}
				pop<IsRising>(from);
			}
			if (rest != weight) push_near<IsRising>(from, a, slope_type(weight - rest));
			if (rest != 0) push_near<!IsRising>(to, a, rest);
		}

		/**
		 * @brief Remove `amount > 0` of weight from the near end of a side
		 *        (the last breakpoint touched may be split), appending the
		 *        removed pieces with true positions to `moved_`; returns the
		 *        position of the last breakpoint touched.
		 * @pre `amount` is less than the side's weight (the callers' bounds)
		 */
		template<bool IsLeft>
		constexpr Value eat_near(side &s, slope_type amount) {
			for (;;) {
				entry &top = s.heap.front();
				Value const position = Value(position_of(top) + s.offset);
				if constexpr (!is_unit)
					if (amount < top.weight) {
						top.weight = slope_type(top.weight - amount);
						moved_.push_back({position, amount});
						return position;
					}
				moved_.push_back({position, weight_of(top)});
				amount = slope_type(amount - weight_of(top));
				pop<IsLeft>(s);
				if (amount == 0) return position;
			}
		}

		/** @brief How many far entries @ref eat_far removes by leaf scans before it sorts the side. */
		static constexpr int far_scan_limit = 4;

		/**
		 * @brief Remove `amount > 0` of weight from the far end of a side
		 *        (the last breakpoint touched may be split); returns its
		 *        position. Up to @ref far_scan_limit entries are found by a scan
		 *        of the heap's leaves (O(m) each); past that the side is sorted
		 *        nearest first (a sorted array is a valid heap) and eaten from
		 *        the back.
		 * @pre `amount` is less than the side's weight (the callers' bounds)
		 */
		template<bool IsLeft>
		static constexpr Value eat_far(side &s, slope_type amount) {
			std::vector<entry> &heap = s.heap;
			order<IsLeft> const is_farther_entry;
			// a clip after a hinge or two eats one or two entries: the farthest
			// is a leaf, so scan the leaves and remove it by one sift-up
			for (int scan = 0; scan < far_scan_limit; ++scan) {
				std::size_t const n = heap.size();
				std::size_t farthest = n / 2;
				for (std::size_t i = farthest + 1; i < n; ++i)
					if (is_farther_entry(heap[i], heap[farthest])) farthest = i;
				entry &eaten = heap[farthest];
				Value const position = Value(position_of(eaten) + s.offset);
				if constexpr (!is_unit)
					if (amount < eaten.weight) {
						eaten.weight = slope_type(eaten.weight - amount);
						return position;
					}
				amount = slope_type(amount - weight_of(eaten));
				// the last entry fills the leaf slot, still a leaf in the smaller heap
				eaten = std::move(heap.back());
				heap.pop_back();
				if (farthest < heap.size())
					std::ranges::push_heap(heap.begin(), heap.begin() + std::ptrdiff_t(farthest + 1), is_farther_entry);
				if (amount == 0) return position;
			}
			if constexpr (IsLeft)
				std::ranges::sort(heap, std::greater<>{}, by_position{});
			else
				std::ranges::sort(heap, std::less<>{}, by_position{});
			for (;;) {
				entry &back = heap.back();
				Value const position = Value(position_of(back) + s.offset);
				if constexpr (!is_unit)
					if (amount < back.weight) {
						back.weight = slope_type(back.weight - amount);
						return position;
					}
				amount = slope_type(amount - weight_of(back));
				heap.pop_back();
				if (amount == 0) return position;
			}
		}

		side left_, right_;
		Value minimum_{};
		/** @brief Scratch for the pieces @ref insert_segment moves. */
		std::vector<weighted_breakpoint<Value, slope_type>> moved_;
	};

	/**
	 * @brief @ref convex_piecewise_linear_by_configuration with Floyd's
	 *        bottom-up `replace_top`: the contender of the hinge shapes, which
	 *        never reach the whole-heap scan.
	 */
	template<piecewise_linear_value Value, piecewise_linear_slope Slope = unit_weight>
		requires scalable_by<Value, atl::detail::slope_count_t<Slope>>
	using convex_piecewise_linear_by_floyd_replace_top = convex_piecewise_linear_by_configuration<Value, Slope, true>;

	/**
	 * @brief @ref convex_piecewise_linear_by_configuration with the library's
	 *        sift-down `replace_top`: differs from the library only in the
	 *        whole-heap scan for the farthest right breakpoint.
	 */
	template<piecewise_linear_value Value, piecewise_linear_slope Slope = unit_weight>
		requires scalable_by<Value, atl::detail::slope_count_t<Slope>>
	using convex_piecewise_linear_by_whole_heap_scan = convex_piecewise_linear_by_configuration<Value, Slope, false>;
}
