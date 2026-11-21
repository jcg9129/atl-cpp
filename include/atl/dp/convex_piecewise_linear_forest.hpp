#pragma once

/**
 * @file
 * @brief @ref atl::convex_piecewise_linear_forest: many convex
 *        piecewise-linear functions over one shared breakpoint pool, with
 *        weighted breakpoints, an O(1) sum of two functions and slope clipping
 *        (the infimal convolution with `D|t|`): the slope trick of tree DPs.
 */

#include <atl/dp/convex_piecewise_linear.hpp>
#include <atl/ds/meldable_heap_forest.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <compare>
#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief A callback taking the pieces of a breakpoint multiset: `sink(position,
	 *        weight)` for a `Position` and a `Slope` weight, both by const reference
	 *        (the eaten pieces of a clip, the visited breakpoints of a walk).
	 */
	template<typename Sink, typename Position, typename Slope>
	concept breakpoint_sink = std::invocable<Sink &, Position const &, Slope const &>;

	namespace detail {
		/** @brief A heap entry of `breakpoint_pool`: a position and the record holding its weight. */
		template<piecewise_linear_value Position>
		struct forest_breakpoint {
			Position position;
			uint32 record;
		};

		/** @brief The lazy translation of a forest breakpoint: moves its position, keeps its record. */
		struct translate_breakpoint {
			template<piecewise_linear_value Position>
			constexpr forest_breakpoint<Position> operator()(Position const &shift, forest_breakpoint<Position> b) const {
				b.position += shift;
				return b;
			}
		};

		/**
		 * @brief Sets ("sets", ids `0, 1, ...`) of weighted breakpoints on one
		 *        pool, each reachable from both ends: every breakpoint is one
		 *        weight record entered in a min-heap and a max-heap by position
		 *        of its set (two @ref atl::meldable_heap_forest pools with a lazy
		 *        translation). Eating weight from one end decrements the shared
		 *        record; an entry whose record the other end emptied is stale
		 *        and dropped when it reaches its heap's top. The engine of
		 *        @ref atl::convex_piecewise_linear_forest.
		 */
		template<piecewise_linear_value Position, signed_integral Slope>
		class breakpoint_pool {
		public:
			using entry = forest_breakpoint<Position>;

		private:
			/** @brief One pool of per-set heaps of entries, ordered by `Compare`, with a lazy translation. */
			template<typename Compare>
				requires std::strict_weak_order<Compare &, entry const &, entry const &>
			using heap_pool =
				meldable_heap_forest<entry, Compare, Position, translate_breakpoint, compose_by_addition>;

			/** @brief Per set, a min-heap of its entries by position (top = leftmost). */
			heap_pool<breakpoint_right_of> leftmost_first_;
			/** @brief Per set, a max-heap of its entries by position (top = rightmost). */
			heap_pool<breakpoint_left_of> rightmost_first_;
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
			constexpr explicit breakpoint_pool(std::size_t set_count)
				: leftmost_first_(set_count), rightmost_first_(set_count) {}

			/** @brief Room for `elements` records (breakpoints) without reallocating. */
			constexpr void reserve(std::size_t elements) {
				leftmost_first_.reserve(elements);
				rightmost_first_.reserve(elements);
				weights_.reserve(elements);
			}

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

			/** @brief `position += d` for every breakpoint now in `set`; O(1). */
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
			 * @brief Empty `set`, returning its nodes to the pools in bulk
			 *        (no pairing, unlike repeated pops).
			 * @complexity O(k) for `k` entries
			 */
			constexpr void clear(std::size_t set) {
				uint32 const heap = heap_of(set);
				leftmost_first_.release(heap);
				rightmost_first_.release(heap);
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
			 *        `out`, equal positions combined (also with `out`'s last
			 *        when it is at the same position: equivalent in both
			 *        directions, whatever `out` held before), and empty the
			 *        set; the right heap is released in bulk.
			 * @pre a combined weight fits `Slope` (asserted)
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
				rightmost_first_.release(heap);
			}
		};
	}

	/**
	 * @brief Many convex piecewise-linear functions with integer slopes over
	 *        one shared pool of weighted breakpoints: add weighted hinges,
	 *        translate, sum two functions in O(1), clip the slopes into `[a, b]`
	 *        (the infimal convolution with a two-sided linear kernel, e.g.
	 *        `D|t|`) and insert a linear segment at a given slope -- the
	 *        operations of the "merge the children, pass the edge up" tree DPs
	 *        of the slope trick.
	 *
	 * Functions are named by @ref atl::convex_piecewise_linear_forest::function_id
	 * (the constructor's `n` functions are `id(0) .. id(n - 1)`, `add_function`
	 * returns more); `forest[f]` is a
	 * @ref atl::convex_piecewise_linear_forest::function_reference with the
	 * slope-trick vocabulary of @ref atl::convex_piecewise_linear (same names,
	 * arguments and return values), and `merge(into, from)` sums two functions.
	 *
	 * A function is kept as its slope at minus infinity (`leftmost_slope`),
	 * its slope at plus infinity (`rightmost_slope`) and the multiset of its
	 * breakpoints `(position, weight)`, the slope rising by `weight` at
	 * `position` (so the weights sum to `rightmost - leftmost`). Each
	 * breakpoint is one record (its weight, shared) entered in a min-heap and
	 * a max-heap by position, both with a lazy translation. Clipping eats
	 * weight from the two ends of the multiset -- the leftmost breakpoints
	 * raise the slope at minus infinity, the rightmost lower the slope at plus
	 * infinity -- splitting the last breakpoint it touches; an entry whose
	 * record was eaten from the other end is dropped when it reaches its
	 * heap's top. So a breakpoint is pushed once per heap when added, popped at
	 * most once per heap by clipping, and pushed again each time
	 * `insert_segment` moves it; `minimizers` and `take_breakpoints` walk by
	 * popping and pushing back. A sum is two melds. No split at the minimum is
	 * kept: what needs the minimum (`minimizers`, the window minimum) walks to
	 * it from the lighter end. Nothing recurses: every walk is a loop over
	 * the pairing heaps, so the depth of a tree DP does not reach the stack.
	 *
	 * Function VALUES are not kept, only the shape: tracking one (say the
	 * left asymptote's intercept) overflows int64 long before the answers of
	 * a tree DP do (a slope of 2e11 times a position of 2e11). The cost of an
	 * optimum comes from its reconstruction instead (see
	 * docs/convex-piecewise-linear-forest.md and experimental/atl/experimental/dp/leaf_path_balance.hpp).
	 *
	 * @tparam Position The coordinate type (`int64`, `int128`, `double`, a
	 *         narrow `int16`); every breakpoint position and translation must
	 *         fit (documented).
	 * @tparam Slope The slope and weight type, a signed integer (`int128`
	 *         included); the sum of the absolute slopes of all functions must
	 *         fit (documented).
	 */
	template<piecewise_linear_value Position = int64, signed_integral Slope = int64>
	class convex_piecewise_linear_forest {
		/** @brief The slopes at minus and plus infinity of one function. */
		struct end_slopes {
			Slope leftmost{}, rightmost{};
		};

		/** @brief The breakpoints of every function. */
		detail::breakpoint_pool<Position, Slope> pool_;
		/** @brief Per function, its end slopes. */
		std::vector<end_slopes> slopes_;
		/** @brief Scratch for the pieces `insert_segment` moves. */
		std::vector<weighted_breakpoint<Position, Slope>> moved_;

		/** @brief The minimizers of function `index`; the walk pops breakpoints and pushes them back. */
		[[nodiscard]] constexpr minimizer_interval<Position> minimizers_of(uint32 index) {
			end_slopes const &s = slopes_[index];
			atl_assert(s.leftmost <= 0 && 0 <= s.rightmost);
			auto &records = pool_;
			minimizer_interval<Position> range;
			if (s.leftmost == 0) {
				if (s.rightmost != 0 && records.template has_live<true>(index))
					range.upper = records.template top<true>(index).position;
			} else if (s.rightmost == 0) {
				if (records.template has_live<false>(index)) range.lower = records.template top<false>(index).position;
			} else if (-s.leftmost <= s.rightmost) {
				std::tie(range.lower, range.upper) = records.template locate<true>(index, Slope(-s.leftmost));
			} else {
				std::tie(range.upper, range.lower) = records.template locate<false>(index, s.rightmost);
			}
			return range;
		}

	public:
		/** @brief The coordinate type. */
		using position_type = Position;
		/** @brief The slope and weight type. */
		using slope_type = Slope;

		class function_reference;
		class const_function_reference;

		/**
		 * @brief The name of one function of a forest: from the constructor
		 *        (via `id`), `add_function` or `function_reference::id`.
		 *        Default-constructed it names function 0.
		 */
		class function_id {
			uint32 index_ = 0;

			constexpr explicit function_id(uint32 index) noexcept : index_(index) {}

			friend class convex_piecewise_linear_forest;
			friend class function_reference;
			friend class const_function_reference;

		public:
			/** @brief The id of function 0. */
			constexpr function_id() noexcept = default;

			/** @brief The function's index: `id(i).index() == i`. */
			[[nodiscard]] constexpr std::size_t index() const noexcept { return index_; }

			/** @brief Ids compare as their indices. */
			friend constexpr bool operator==(function_id const &, function_id const &) noexcept = default;
			/** @brief Ids order as their indices. */
			friend constexpr std::strong_ordering operator<=>(function_id const &, function_id const &) noexcept =
				default;
		};

		/**
		 * @brief One function of the forest, by reference (a forest pointer and
		 *        an index: cheap to copy, valid across `add_function` while the
		 *        forest lives and stays in place), with the slope-trick
		 *        vocabulary of @ref atl::convex_piecewise_linear.
		 */
		class function_reference {
			convex_piecewise_linear_forest *forest_;
			uint32 index_;

			constexpr function_reference(convex_piecewise_linear_forest &forest, uint32 index) noexcept
				: forest_(&forest), index_(index) {}

			friend class convex_piecewise_linear_forest;

			constexpr end_slopes &slopes() const { return forest_->slopes_[index_]; }

			constexpr detail::breakpoint_pool<Position, Slope> &pool() const { return forest_->pool_; }

			constexpr void push(Position const &position, Slope weight) const {
				if (weight != 0) pool().push(index_, position, weight);
			}

		public:
			/** @brief The function's id. */
			[[nodiscard]] constexpr function_id id() const noexcept { return function_id(index_); }

			/**
			 * @brief `f(x) += weight * max(0, x - a)`.
			 * @pre `weight >= 0` (asserted); a zero weight changes nothing
			 * @complexity O(1) amortized
			 */
			constexpr void add_rising_hinge(Position const &a, Slope weight = Slope{1}) const {
				atl_assert(weight >= 0);
				slopes().rightmost += weight;
				push(a, weight);
			}

			/**
			 * @brief `f(x) += weight * max(0, a - x)`.
			 * @pre `weight >= 0` (asserted); a zero weight changes nothing
			 * @complexity O(1) amortized
			 */
			constexpr void add_falling_hinge(Position const &a, Slope weight = Slope{1}) const {
				atl_assert(weight >= 0);
				slopes().leftmost -= weight;
				push(a, weight);
			}

			/**
			 * @brief `f(x) += weight * |x - a|`, one breakpoint of weight `2 * weight`.
			 * @pre `0 <= weight <= max(Slope) / 2` (asserted); a zero weight changes nothing
			 * @complexity O(1) amortized
			 */
			constexpr void add_absolute(Position const &a, Slope weight = Slope{1}) const {
				atl_assert(weight >= 0 && weight <= std::numeric_limits<Slope>::max() / 2);
				end_slopes &s = slopes();
				s.leftmost -= weight;
				s.rightmost += weight;
				push(a, Slope(weight + weight));
			}

			/**
			 * @brief `f(x) := f(x - d)`: the graph moves `d` to the right.
			 * @complexity O(1)
			 */
			constexpr void translate(Position const &d) const { pool().translate(index_, d); }

			/**
			 * @brief Clip the slopes of `f` into `[min_slope, max_slope]`: `f :=
			 *        min_y f(y) + k(x - y)` with `k(t) = max_slope * t` for `t >= 0`
			 *        and `min_slope * t` for `t <= 0` (with `min_slope = -D`,
			 *        `max_slope = D`, the infimal convolution with `D|t|`). The
			 *        part of the graph whose slopes are inside stays in place;
			 *        steeper parts are replaced by rays of the limiting slopes.
			 *
			 * Returns where the slopes were unchanged: its `clamp(x)` is an optimal
			 * `y` above, and `clip_slopes(0, 0)` returns the minimizers of `f`
			 * (flattening it).
			 *
			 * @pre `min_slope <= max_slope`, `leftmost_slope() <= max_slope` and
			 *      `min_slope <= rightmost_slope()` (else the infimum is minus
			 *      infinity) (asserted)
			 * @complexity O((k + 1) log n) amortized for `k` breakpoints removed
			 */
			constexpr slope_clip_interval<Position> clip_slopes(Slope min_slope, Slope max_slope) const {
				end_slopes &s = slopes();
				atl_assert(min_slope <= max_slope);
				atl_assert(s.leftmost <= max_slope && min_slope <= s.rightmost);
				auto const ignore = [](Position const &, Slope const &) {};
				slope_clip_interval<Position> range;
				if (s.leftmost < min_slope) {
					range.lower = pool().template eat<true>(index_, Slope(min_slope - s.leftmost), ignore);
					s.leftmost = min_slope;
				}
				if (max_slope < s.rightmost) {
					range.upper = pool().template eat<false>(index_, Slope(s.rightmost - max_slope), ignore);
					s.rightmost = max_slope;
				}
				return range;
			}

			/**
			 * @brief `f(x) := min over y <= x of f(y)`: `clip_slopes(leftmost_slope(), 0)`.
			 * @return `{none, upper}`, `upper` the largest minimizer before the
			 *         call (none if `f` was nonincreasing)
			 * @pre `leftmost_slope() <= 0` (asserted)
			 * @complexity O((k + 1) log n) amortized for `k` breakpoints removed
			 */
			constexpr slope_clip_interval<Position> take_prefix_minimum() const {
				return clip_slopes(slopes().leftmost, 0);
			}

			/**
			 * @brief `f(x) := min over y >= x of f(y)`: `clip_slopes(0, rightmost_slope())`.
			 * @return `{lower, none}`, `lower` the smallest minimizer before the
			 *         call (none if `f` was nondecreasing)
			 * @pre `0 <= rightmost_slope()` (asserted)
			 * @complexity O((k + 1) log n) amortized for `k` breakpoints removed
			 */
			constexpr slope_clip_interval<Position> take_suffix_minimum() const {
				return clip_slopes(0, slopes().rightmost);
			}

			/**
			 * @brief `f(x) := min over lo <= d <= hi of f(x - d)`:
			 *        `insert_segment(0, hi - lo)`, then `translate(lo)`.
			 * @return A point `p` where `f`'s slope rose through 0 before the
			 *         call, or none when `leftmost_slope() >= 0` (read minus
			 *         infinity): an optimal `y` for `x` is `clamp(p, x - hi, x - lo)`.
			 * @pre `lo <= hi` and `0 <= rightmost_slope()` (asserted)
			 * @complexity as `insert_segment`: NOT O(1) as on
			 *      @ref atl::convex_piecewise_linear, the forest keeps no split at
			 *      the minimum
			 */
			constexpr std::optional<Position> take_window_minimum(Position const &lo, Position const &hi) const {
				atl_assert(!(hi < lo));
				std::optional<Position> const crossing = insert_segment(Slope{}, Position(hi - lo));
				translate(lo);
				return crossing;
			}

			/**
			 * @brief Insert a segment of slope `slope` and length `length` into the
			 *        graph where `f`'s slope rises through `slope`: the part with
			 *        slopes `>= slope` moves `length` to the right, i.e. `f :=
			 *        min_{0 <= t <= length} f(x - t) + slope * t`, up to the constant
			 *        `slope * length` the forest does not track when `slope` is below
			 *        every slope of `f`.
			 *
			 * Returns a point `p` where the slope of `f` rises through `slope` (`f`
			 * has slopes `<= slope` left of `p`, `>= slope` right of it), or none
			 * when `slope <= leftmost_slope()` (then the whole graph moves; read
			 * minus infinity): an optimal `t` for `x` is `x - clamp(p, x - length,
			 * x)`. Moves whichever side carries less weight (the other is moved by
			 * a lazy translation), so the cost is in the smaller of
			 * `rightmost_slope() - slope` and `slope - leftmost_slope()`. Where
			 * `f` has a piece of slope exactly `slope`, `p` is its right end when
			 * the right side is moved (`rightmost_slope() - slope <= slope -
			 * leftmost_slope()`) and its left end otherwise; both are optimal.
			 *
			 * With `clip_slopes` it builds a DP edge kernel with a lower bound
			 * on its new length: `D|t - C|` over `t >= 0` is a
			 * clip to slopes `<= D` and a segment of slope `-D` and length `C`.
			 *
			 * @pre `length >= 0` and `slope <= rightmost_slope()` (asserted)
			 * @complexity O((k + 1) log n) amortized for the `k` breakpoints on the
			 *      side moved, at most that side's weight; each is pushed again
			 */
			constexpr std::optional<Position> insert_segment(Slope slope, Position const &length) const {
				end_slopes const &s = slopes();
				atl_assert(!(length < Position{}));
				atl_assert(slope <= s.rightmost);
				if (slope <= s.leftmost) {
					translate(length);
					return std::nullopt;
				}
				auto &records = pool();
				if (slope == s.rightmost) {
					[[maybe_unused]] bool const is_live = records.template has_live<false>(index_);
					atl_assert(is_live);
					return records.template top<false>(index_).position;
				}
				auto &moved = forest_->moved_;
				moved.clear();
				Position crossing{};
				if (s.rightmost - slope <= slope - s.leftmost) {
					crossing = records.template eat<false>(index_, Slope(s.rightmost - slope),
														   [&](Position const &at, Slope const &weight) {
															   moved.push_back({Position(at + length), weight});
														   });
				} else {
					// the light left part stays: take it out, move the rest, put it back
					crossing = records.template eat<true>(index_, Slope(slope - s.leftmost),
														  [&](Position const &at, Slope const &weight) {
															  moved.push_back({at, weight});
														  });
					translate(length);
				}
				for (auto const &[position, weight] : moved) records.push(index_, position, weight);
				return crossing;
			}

			/** @brief The slope of `f` at minus infinity (left of every breakpoint). */
			[[nodiscard]] constexpr Slope leftmost_slope() const { return slopes().leftmost; }

			/** @brief The slope of `f` at plus infinity (right of every breakpoint). */
			[[nodiscard]] constexpr Slope rightmost_slope() const { return slopes().rightmost; }

			/**
			 * @brief Where `f` attains its minimum; `f` is unchanged. Walks from
			 *        the end carrying less weight to where the slope crosses 0
			 *        (popping and pushing back), so not O(1).
			 * @pre `leftmost_slope() <= 0 <= rightmost_slope()` (asserted)
			 * @complexity O((k + 1) log n) amortized for the `k` breakpoints
			 *      between the lighter end and the minimum, `k <=
			 *      min(-leftmost_slope(), rightmost_slope())`
			 */
			[[nodiscard]] constexpr minimizer_interval<Position> minimizers() const { return forest_->minimizers_of(index_); }

			/**
			 * @brief The breakpoints of `f` by increasing position, equal positions
			 *        combined, and `f` reset to the zero function (read
			 *        `leftmost_slope()` and `rightmost_slope()` beforehand).
			 * @complexity O((k + 1) log n) amortized for `k` entries
			 */
			constexpr std::vector<weighted_breakpoint<Position, Slope>> take_breakpoints() const {
				std::vector<weighted_breakpoint<Position, Slope>> breakpoints;
				pool().take_all(index_, breakpoints);
				slopes() = {};
				return breakpoints;
			}
		};

		/**
		 * @brief One function of a const forest, by reference: the O(1)
		 *        queries of @ref atl::convex_piecewise_linear_forest::function_reference
		 *        (the same names and results), from `operator[]` on a const
		 *        forest. `minimizers` is not among them: its walk pops and
		 *        pushes back breakpoints, so it needs a non-const forest.
		 *        Concurrent const queries are safe. Cheap to copy; valid while
		 *        the forest lives and stays in place.
		 */
		class const_function_reference {
			convex_piecewise_linear_forest const *forest_;
			uint32 index_;

			constexpr const_function_reference(convex_piecewise_linear_forest const &forest, uint32 index) noexcept
				: forest_(&forest), index_(index) {}

			friend class convex_piecewise_linear_forest;

		public:
			/** @brief The function's id. */
			[[nodiscard]] constexpr function_id id() const noexcept { return function_id(index_); }

			/** @brief The slope of `f` at minus infinity (left of every breakpoint). */
			[[nodiscard]] constexpr Slope leftmost_slope() const { return forest_->slopes_[index_].leftmost; }

			/** @brief The slope of `f` at plus infinity (right of every breakpoint). */
			[[nodiscard]] constexpr Slope rightmost_slope() const { return forest_->slopes_[index_].rightmost; }
		};

		/**
		 * @brief `function_count` zero functions, `id(0) .. id(function_count - 1)`.
		 * @pre `function_count < 2^32` (asserted)
		 * @complexity O(function_count)
		 */
		constexpr explicit convex_piecewise_linear_forest(std::size_t function_count = 0)
			: pool_(function_count), slopes_(function_count) {
			atl_assert(function_count < std::numeric_limits<uint32>::max());
		}

		/** @brief Room for `elements` breakpoints (records) without reallocating. */
		constexpr void reserve(std::size_t elements) { pool_.reserve(elements); }

		/** @brief The number of functions. */
		[[nodiscard]] constexpr std::size_t function_count() const noexcept { return slopes_.size(); }

		/**
		 * @brief The id of function `index`.
		 * @pre `index < function_count()` (asserted)
		 */
		[[nodiscard]] constexpr function_id id(std::size_t index) const {
			atl_assert(index < slopes_.size());
			return function_id(uint32(index));
		}

		/**
		 * @brief A new zero function; returns its id, `id(function_count() - 1)`.
		 * @pre fewer than `2^32 - 2` functions before the call (asserted)
		 * @complexity O(1) amortized
		 */
		constexpr function_id add_function() {
			atl_assert(slopes_.size() + 1 < std::numeric_limits<uint32>::max());
			pool_.add_set();
			slopes_.emplace_back();
			return function_id(uint32(slopes_.size() - 1));
		}

		/**
		 * @brief The function `f`.
		 * @pre `f` names a function of this forest (asserted: its index is in range)
		 */
		[[nodiscard]] constexpr function_reference operator[](function_id f) {
			atl_assert(f.index_ < slopes_.size());
			return function_reference(*this, f.index_);
		}

		/**
		 * @brief The function `f` of a const forest, read-only.
		 * @pre `f` names a function of this forest (asserted: its index is in range)
		 */
		[[nodiscard]] constexpr const_function_reference operator[](function_id f) const {
			atl_assert(f.index_ < slopes_.size());
			return const_function_reference(*this, f.index_);
		}

		/**
		 * @brief `into := into + from` and `from := 0`; a no-op when `into == from`.
		 * @pre both name functions of this forest (asserted)
		 * @complexity O(1)
		 */
		constexpr void merge(function_id into, function_id from) {
			atl_assert(into.index_ < slopes_.size() && from.index_ < slopes_.size());
			if (into == from) return;
			end_slopes &target = slopes_[into.index_];
			end_slopes &source = slopes_[from.index_];
			target.leftmost += source.leftmost;
			target.rightmost += source.rightmost;
			source = {};
			pool_.meld(into.index_, from.index_);
		}
	};
}
