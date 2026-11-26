#pragma once

/**
 * @file
 * @brief @ref atl::convex_piecewise_linear, a convex piecewise-linear function
 *        of one variable kept as its weighted breakpoints in two heaps around
 *        the minimum (the "slope trick" representation of a DP row that is
 *        convex in its index), and the vocabulary types it shares with
 *        @ref atl::convex_piecewise_linear_forest.
 */

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
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief A coordinate-and-value type for @ref atl::convex_piecewise_linear
	 *        and @ref atl::convex_piecewise_linear_forest — totally ordered,
	 *        closed under `+` and `-`, with `Value{}` its zero, and not an
	 *        unsigned integer (offsets and values go negative). Any signed
	 *        integer type (`atl::int128` included) or floating-point type.
	 */
	template<typename Value>
	concept piecewise_linear_value = std::semiregular<Value> && std::totally_ordered<Value> && !unsigned_integral<Value>
								  && requires(Value const a, Value &b) {
										 { a + a } -> std::convertible_to<Value>;
										 { a - a } -> std::convertible_to<Value>;
										 b += a;
										 b -= a;
									 };

	/**
	 * @brief A @ref atl::piecewise_linear_value that a slope of type `Slope`
	 *        can scale: `Value(slope) * value` is a value (a function's value
	 *        changes by slope times distance).
	 */
	template<typename Value, typename Slope>
	concept scalable_by = piecewise_linear_value<Value> && signed_integral<Slope>
					   && std::constructible_from<Value, Slope const &> && requires(Value const a) {
							  { a * a } -> std::convertible_to<Value>;
						  };

	/**
	 * @brief A closed interval `[lower, upper]` of positions, an absent end
	 *        meaning the interval is unbounded on that side: where a
	 *        `clip_slopes` call left the slopes unchanged, or (as
	 *        @ref atl::minimizer_interval) where a function attains its minimum.
	 *
	 * Clipping `f` to slopes in `[a, b]` is the infimal convolution `h(x) =
	 * min_y f(y) + k(x - y)` with `k(t) = b t` for `t >= 0` and `a t` for
	 * `t <= 0`; `clamp(x)` is an optimal `y` for that `x`, which is what a
	 * top-down reconstruction needs. On a minimizer interval, `clamp(x)` is the
	 * minimizer nearest `x`.
	 */
	template<piecewise_linear_value Position>
	struct slope_clip_interval {
		/** @brief The lower end, or none when the interval is unbounded below (nothing clipped on the left). */
		std::optional<Position> lower;
		/** @brief The upper end, or none when the interval is unbounded above (nothing clipped on the right). */
		std::optional<Position> upper;

		/** @brief `x` moved into `[lower, upper]`. */
		[[nodiscard]] constexpr Position clamp(Position x) const {
			if (lower && x < *lower) return *lower;
			if (upper && *upper < x) return *upper;
			return x;
		}
	};

	/** @brief Where a convex piecewise-linear function attains its minimum (the same interval type). */
	template<piecewise_linear_value Position>
	using minimizer_interval = slope_clip_interval<Position>;

	/** @brief A breakpoint where a convex piecewise-linear function's slope rises by `weight`. */
	template<piecewise_linear_value Position, signed_integral Slope>
	struct weighted_breakpoint {
		/** @brief Where the slope changes. */
		Position position;
		/** @brief By how much it rises there (positive). */
		Slope weight;
	};

	namespace detail {
		/** @brief A breakpoint record: a `position` member of a @ref atl::piecewise_linear_value type. */
		template<typename Breakpoint>
		concept positioned_breakpoint = requires(Breakpoint const &b) {
			requires piecewise_linear_value<std::remove_cvref_t<decltype(b.position)>>;
		};

		/** @brief Orders breakpoints by position: the max-heap order (top = rightmost). */
		struct breakpoint_left_of {
			template<positioned_breakpoint Breakpoint>
			[[nodiscard]] constexpr bool operator()(Breakpoint const &a, Breakpoint const &b) const {
				return a.position < b.position;
			}
		};

		/** @brief Orders breakpoints by reversed position: the min-heap order (top = leftmost). */
		struct breakpoint_right_of {
			template<positioned_breakpoint Breakpoint>
			[[nodiscard]] constexpr bool operator()(Breakpoint const &a, Breakpoint const &b) const {
				return b.position < a.position;
			}
		};
	}

	/**
	 * @brief The `Slope` argument of @ref atl::convex_piecewise_linear that
	 *        selects unit breakpoints (the default): every hinge has weight 1
	 *        and a breakpoint is a bare position, the fastest storage for the
	 *        unit slope trick.
	 */
	struct unit_weight {};

	/**
	 * @brief A `Slope` argument of @ref atl::convex_piecewise_linear —
	 *        @ref atl::unit_weight (unit breakpoints) or a signed integer type
	 *        (weighted breakpoints of that weight type).
	 */
	template<typename Slope>
	concept piecewise_linear_slope = std::same_as<Slope, unit_weight> || signed_integral<Slope>;

	namespace detail {
		/** @brief The slope type a `Slope` argument reports slopes in: `int64` for @ref atl::unit_weight. */
		template<piecewise_linear_slope Slope>
		using slope_count_t = std::conditional_t<std::same_as<Slope, unit_weight>, int64, Slope>;
	}

	/**
	 * @brief A convex piecewise-linear function `f` of one variable with
	 *        integer slopes, starting from `f = 0`, under adding hinges
	 *        `w max(0, x - a)`, `w max(0, a - x)`, `w |x - a|` and constants,
	 *        translation, slope clipping (prefix / suffix minimum as special
	 *        cases), sliding-window minimum and segment insertion (the "slope
	 *        trick"), with its minimum value tracked.
	 *
	 * `f` is stored as its minimum value and its breakpoints, the slope
	 * rising at each. The breakpoints left of the minimum sit in a max-heap,
	 * those right of it in a min-heap, each with a lazy offset, so the
	 * minimum is between the two tops. A hinge that lands across the minimum
	 * moves weight from one top to the other; the window and the prefix /
	 * suffix minimum touch only offsets or drop a heap.
	 *
	 * Two storage modes, by `Slope`: @ref atl::unit_weight (the default) keeps
	 * bare positions, a breakpoint of multiplicity `m` stored `m` times, and
	 * the hinges take no weight; a signed integer `Slope` keeps `(position,
	 * weight)` entries and the hinges take a weight (default 1). Unit storage
	 * is the faster of the two on unit hinges (half the entry size, no weight
	 * bookkeeping).
	 *
	 * The same vocabulary (names, arguments, return values) as a function of
	 * @ref atl::convex_piecewise_linear_forest; this class also tracks the
	 * value (@ref minimum, @ref add_constant, `f(x)`), and so keeps
	 * `leftmost_slope() <= 0 <= rightmost_slope()`. Clipping the far ends of a
	 * heap (a `clip_slopes` that is not a prefix or suffix minimum) cannot
	 * reach them through the heap order: it scans that heap's leaves for each
	 * of the first four entries it eats, O(m) each for the side's `m` entries,
	 * then sorts the side, O(m log m). A DP that clips large sides on every
	 * step belongs on the forest, where it is amortized O(log n).
	 *
	 * Typical use: a DP over `i` whose row `dp_i(x)` is convex in `x` and
	 * moves by "add `|x - a_i|`, then take a prefix minimum" (make a sequence
	 * nondecreasing at the least total change) or "take the minimum over a
	 * window of predecessors" (`dp_i(x) = min_{x - hi <= y <= x - lo}
	 * dp_{i-1}(y) + cost_i(x)`).
	 *
	 * Overflow: for an integral `Value` up to 64 bits, the distances,
	 * products and sums that only feed the tracked values -- `p - a` and
	 * `weight * |p - a|` of a hinge crossing the minimum, `slope * length`
	 * of a segment, the hinge terms summed by `f(x)` -- are formed in a
	 * wider integer (`int128` for `int64`, `int32` below 32 bits) and only
	 * the minimum or `f(x)` is narrowed back, so they need not fit `Value`
	 * (`add_constant(INT64_MIN)`, `add_rising_hinge(INT64_MIN)`, then `f(0)
	 * == 0` is exact). An `int128` or floating `Value` has no wider type:
	 * there they must fit `Value` too. Every coordinate argument is read
	 * before the function changes, so an argument referring to the
	 * function's own state (`add_rising_hinge(f.minimum())`) is the value
	 * at the call.
	 *
	 * @tparam Value The coordinate and value type (`int64`, `int128`,
	 *         `double`); a breakpoint, an offset, a stored breakpoint (its
	 *         position minus its side's offset: the translations since the
	 *         side was last emptied) and the minimum each must fit, and so
	 *         must `f(x)` for every evaluated `x` (documented); intermediates
	 *         as under Overflow.
	 * @tparam Slope @ref atl::unit_weight, or the slope and weight type (a
	 *         signed integer); slopes are reported as `slope_type` (`int64` in
	 *         unit mode), and the sum of the weights must fit it.
	 */
	template<piecewise_linear_value Value, piecewise_linear_slope Slope = unit_weight>
		requires scalable_by<Value, detail::slope_count_t<Slope>>
	class convex_piecewise_linear {
	public:
		/** @brief The coordinate and value type. */
		using value_type = Value;
		/** @brief The type of a position (a breakpoint, an interval end): the same as @ref value_type, spelled as on @ref atl::convex_piecewise_linear_forest. */
		using position_type = Value;
		/** @brief The type slopes and weights are reported in (`int64` in unit mode). */
		using slope_type = detail::slope_count_t<Slope>;
		/** @brief Whether breakpoints are unit (bare positions, hinges without a weight). */
		static constexpr bool is_unit = std::same_as<Slope, unit_weight>;

	private:
		/** @brief The type intermediate distances, products and sums are formed in (see Overflow). */
		using wide = detail::dp_wide_t<Value>;

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
		constexpr convex_piecewise_linear() = default;

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
			// `a` may refer to the minimum, which the first hinge changes
			Value const point = a;
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
		 *      `p`; O(m) for `slope == rightmost_slope()` (a scan of the leaves
		 *      of the `m` entries right of the minimum for the farthest)
		 */
		constexpr std::optional<Value> insert_segment(slope_type slope, Value const &segment_length) {
			Value const length = segment_length;
			atl_assert(!(length < Value{}));
			atl_assert(slope <= total(right_));
			if (slope <= -total(left_)) {
				translate(length);
				add_to_minimum(slope, wide(length));
				return std::nullopt;
			}
			if (slope == 0) {
				right_.offset = Value(right_.offset + length);
				return top_position(left_);
			}
			if (slope == total(right_)) {
				// nothing is steeper, so the graph is unchanged and `p` is the
				// farthest breakpoint: a min-heap's maximum is a leaf, and the
				// leaves start at size / 2; keeping the farthest position rather
				// than its index avoids a dependent reload on every comparison
				std::size_t const first = right_.heap.size() / 2;
				Value farthest = position_of(right_.heap[first]);
				for (std::size_t i = first + 1; i < right_.heap.size(); ++i)
					if (farthest < position_of(right_.heap[i])) farthest = position_of(right_.heap[i]);
				return Value(farthest + right_.offset);
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
			add_to_minimum(slope, wide(length));
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
		 * @pre the combined weight at every position fits `slope_type`
		 *      (asserted). Each side's weight fits by the class contract, but
		 *      a position at the minimum can carry weight from both sides:
		 *      `add_absolute(0, 127)` on `Slope = int8` makes a breakpoint of
		 *      weight 254. Unit mode counts in `int64` and always fits.
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
		 *
		 * The terms are nonnegative, so every partial sum lies between the
		 * minimum and `f(x)`: in the wide type (see Overflow) nothing
		 * overflows when `f(x)` fits.
		 *
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
										 std::conditional_t<IsLeft, detail::breakpoint_left_of, detail::breakpoint_right_of>>;

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

		/**
		 * @brief `minimum += slope * distance`, formed in the wide type: the
		 *        product is the change of the minimum, so it fits there
		 *        whenever the old and new minimum fit `Value`.
		 */
		constexpr void add_to_minimum(slope_type slope, wide const &distance) {
			minimum_ = Value(wide(wide(minimum_) + wide(wide(slope) * distance)));
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
		 * @brief Replace the top of a non-empty side with `value` by one
		 *        sift-down, half the work of a pop and a push.
		 */
		template<bool IsLeft>
		static constexpr void replace_top(side &s, entry value) {
			std::vector<entry> &heap = s.heap;
			order<IsLeft> const comp;
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
						minimum_ = Value(wide(wide(minimum_) + wide(wide(top) - wide(a))));
					else
						minimum_ = Value(wide(wide(minimum_) + wide(wide(a) - wide(top))));
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
				// every step raises the minimum toward its final value, so each
				// partial minimum fits and only the product needs the wide type
				if constexpr (IsRising)
					add_to_minimum(moved, wide(wide(p) - wide(a)));
				else
					add_to_minimum(moved, wide(wide(a) - wide(p)));
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
}
