#pragma once

/**
 * @file
 * @brief @ref atl::li_chao_tree, the lower (or upper) envelope of lines and line
 *        segments over a fixed set of integer query points, with line and
 *        segment insertion and point queries.
 */

#include <atl/ds/linear_function.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Lines `y = slope * x + intercept` -- or members of any function
	 *        family in which two members cross at most once -- and segments
	 *        (a function on `[l, r)` only), over a domain of query points
	 *        fixed up front: every integer of `[lo, hi)`, or a sorted list of
	 *        coordinates. Keeps the best value at each point (the Li Chao
	 *        tree).
	 *
	 * A perfect binary tree over the domain's positions; each node holds at
	 * most one function (a used bit per node lives in a separate bitset), the
	 * best one at the node's midpoint among those that reached it, and a query
	 * at a point takes the best of the functions on its leaf-to-root path.
	 * Inserting a function descends one root-to-leaf path, pushing the loser
	 * at each midpoint into the one half where it can still win -- which is
	 * where the crossing-at-most-once property is used; a function that wins
	 * (or loses) at both ends of a node's range wins (or loses) on all of
	 * it, so the loser is dropped there. A segment is split over the
	 * O(log n) canonical nodes covering its positions (a segment reaching
	 * the last point also covers the padding positions past it), each
	 * insertion then descending from there.
	 *
	 * @tparam Coordinate The domain's point type (`int64`; any totally
	 *         ordered type for the coordinate-list domain, `double` say).
	 * @tparam Compare `comp(a, b)` reads "a is better than b" on the values:
	 *         std::less<> for the minimum (lower envelope), std::greater<>
	 *         for the maximum.
	 * @tparam Function The family's member type (@ref atl::envelope_function),
	 *         `linear_function<Coordinate>` by default, which also gets the
	 *         `insert(slope, intercept)` shorthands and an exact overflow
	 *         check. The value type is what `f(x)` returns.
	 * @pre any two inserted functions cross at most once over the domain:
	 *      `f(x)` better than `g(x)` on a prefix or a suffix of the points
	 *      (not checkable; documented, as for every non-line family). Every
	 *      evaluation `f(x)` at a covered domain point is valid, including all intermediates (for a line
	 *      over integers this is asserted at insertion).
	 */
	template<typename Coordinate, typename Compare = std::less<>, typename Function = linear_function<Coordinate>>
		requires std::totally_ordered<Coordinate> && std::copyable<Coordinate>
			  && envelope_function<Function, Coordinate, Compare>
	class li_chao_tree {
	public:
		/** @brief Domain coordinate type. */
		using coordinate_type = Coordinate;
		/** @brief Stored function type. */
		using function_type = Function;
		/** @brief Function evaluation result type. */
		using value_type = envelope_value_t<Function, Coordinate>;

		/** @brief An empty domain, a placeholder to assign a real tree to later. */
		constexpr li_chao_tree() = default;

		/**
		 * @brief Every integer `x` in `[lo, hi)`, no functions yet.
		 * @param lo Inclusive domain bound.
		 * @param hi Exclusive domain bound.
		 * @param comp Strict weak order defining the better value.
		 * @pre `lo <= hi` and `hi - lo` representable in Coordinate and less than `2^31` (asserted); memory is about
		 *      `2 * bit_ceil(hi - lo)` slots of `sizeof(Function)` bytes and
		 *      one bit.
		 * @complexity O(hi - lo)
		 */
		constexpr li_chao_tree(Coordinate lo, Coordinate hi, Compare comp = {})
			requires integral<Coordinate>
			: comp_(std::move(comp)), lo_(lo) {
			allocate(atl::detail::li_chao_width(lo, hi));
		}

		/**
		 * @brief The points of `coordinates`, the only ones a query may ask
		 *        and a segment's ends are matched against; no functions yet.
		 * @pre `coordinates` strictly increasing (asserted) and fewer than
		 *      `2^31`; the padded slot count must fit size_t (asserted).
		 * @complexity O(n)
		 */
		constexpr explicit li_chao_tree(std::vector<Coordinate> coordinates, Compare comp = {})
			: comp_(std::move(comp)), coordinates_(std::move(coordinates)), has_coordinates_(true) {
			atl_assert(coordinates_.size() <= std::size_t((std::numeric_limits<int32>::max)()));
			atl_assert(std::ranges::adjacent_find(coordinates_, std::ranges::greater_equal{}) == coordinates_.end());
			allocate(coordinates_.size());
		}

		/** @brief The number of query points. */
		constexpr std::size_t size() const { return size_; }

		/** @brief Whether the domain has no points. */
		constexpr bool empty() const { return size_ == 0; }

		/**
		 * @brief Add `f` over the whole domain.
		 * @param f Function whose evaluations satisfy the class preconditions.
		 * @complexity O(log n)
		 */
		constexpr void insert(Function f) {
			if (size_ == 0) return;
			atl_assert(detail::evaluates_on(f, coordinate(0), coordinate(size_ - 1)));
			descend(1, 0, leaves_, std::move(f));
		}

		/**
		 * @brief Add `f` on the domain points `x` with `l <= x < r` only (a
		 *        segment); an empty range adds nothing.
		 * @param f Function to add.
		 * @param l Inclusive segment bound.
		 * @param r Exclusive segment bound.
		 * @complexity O(log^2 n)
		 */
		constexpr void insert(Function const &f, Coordinate const &l, Coordinate const &r) {
			uint32 const begin = position_at_least(l), end = position_at_least(r);
			if (begin >= end) return;
			atl_assert(detail::evaluates_on(f, coordinate(begin), coordinate(end - 1)));
			// a range reaching the last point also takes the padding positions:
			// they repeat the last point and no query asks there, and the range
			// then splits into one node per level on its left side only (a
			// whole-domain range into the root alone)
			uint64 first = uint64(begin) + leaves_, last = uint64(end == size_ ? leaves_ : end) + leaves_;
			// at the level of `width`-leaf subtrees a node covers the positions
			// from (node - leaves / width) * width
			uint64 width = 1;
			auto const settle = [&](uint64 node) {
				descend(uint32(node), uint32((node - leaves_ / width) * width), uint32(width), f);
			};
			for (; first < last; first >>= 1, last >>= 1, width <<= 1) {
				if (first & 1) settle(first++);
				if (last & 1) settle(--last);
			}
		}

		/**
		 * @brief Add the line `slope * x + intercept` over the whole domain.
		 * @pre its product and sum at every domain point fit (asserted at the two
		 *      ends for an integral coordinate)
		 * @complexity O(log n)
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept)
			requires detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)});
		}

		/**
		 * @brief Add the line `slope * x + intercept` on `l <= x < r` only.
		 * @complexity O(log^2 n)
		 */
		constexpr void insert(Coordinate slope, Coordinate intercept, Coordinate const &l, Coordinate const &r)
			requires detail::is_linear_function_of<Function, Coordinate>
		{
			insert(Function{std::move(slope), std::move(intercept)}, l, r);
		}

		/**
		 * @brief The best value at `x` over the functions and segments
		 *        covering it, or none when nothing covers it.
		 * @param x Domain coordinate to evaluate.
		 * @return Best covering value, or nullopt if no function covers x.
		 * @pre `x` is a domain point (asserted)
		 * @complexity O(log n) (plus a binary search over the coordinates)
		 */
		[[nodiscard]] constexpr std::optional<value_type> query(Coordinate const &x) const {
			uint32 const position = position_of(x);
			std::optional<value_type> best;
			for (uint32 node = position + leaves_; node > 0; node >>= 1) {
				if (!is_used(node)) continue;
				value_type value = functions_[node](x);
				if (!best || comp_(value, *best)) best = std::move(value);
			}
			return best;
		}

	private:
		[[nodiscard]] constexpr bool is_used(uint32 node) const { return used_[node >> 6] >> (node & 63) & 1; }

		/**
		 * @brief Size the tree for `points` positions, all slots empty.
		 * @pre `points < 2^31` (asserted by both constructors), so the
		 *      positions and the `2 * bit_ceil(points)` slot ids fit `uint32`.
		 */
		constexpr void allocate(std::size_t points) {
			atl_assert(points <= std::size_t((std::numeric_limits<int32>::max)()));
			size_ = uint32(points);
			leaves_ = std::bit_ceil((std::max<uint32>)(size_, 1));
			atl_assert(std::size_t(leaves_) <= (std::numeric_limits<std::size_t>::max)() / 2);
			functions_.assign(2 * std::size_t(leaves_), Function{});
			used_.assign((2 * std::size_t(leaves_)) / 64 + ((2 * std::size_t(leaves_)) % 64 != 0), 0);
		}

		/** @brief The coordinate of position `i`; padding positions repeat the last point. */
		[[nodiscard]] constexpr Coordinate coordinate(uint32 i) const {
			i = (std::min)(i, size_ - 1);
			if constexpr (integral<Coordinate>)
				return has_coordinates_ ? coordinates_[i] : Coordinate(lo_ + Coordinate(i));
			else
				return coordinates_[i];
		}

		/** @brief The position of the domain point `x` (asserted to be one). */
		[[nodiscard]] constexpr uint32 position_of(Coordinate const &x) const {
			if constexpr (integral<Coordinate>) {
				if (!has_coordinates_) {
					atl_assert(!(x < lo_) && x < Coordinate(lo_ + Coordinate(size_)));
					return uint32(x - lo_);
				}
			}
			auto const it = std::ranges::lower_bound(coordinates_, x);
			atl_assert(it != coordinates_.end() && !(x < *it));
			return uint32(it - coordinates_.begin());
		}

		/** @brief The first position whose coordinate is at least `x` (`size()` if none). */
		[[nodiscard]] constexpr uint32 position_at_least(Coordinate const &x) const {
			if constexpr (integral<Coordinate>) {
				if (!has_coordinates_) {
					if (!(lo_ < x)) return 0;
					// lo + size <= hi fits, x - lo may not
					if (!(x < Coordinate(lo_ + Coordinate(size_)))) return size_;
					return uint32(x - lo_);
				}
			}
			return uint32(std::ranges::lower_bound(coordinates_, x) - coordinates_.begin());
		}

		/**
		 * @brief Settle `added` into the subtree of `node`, which covers the
		 *        `width` positions from `first`: keep the better function at
		 *        each midpoint and send the other into the half where it can
		 *        still win.
		 */
		constexpr void descend(uint32 node, uint32 first, uint32 width, Function added) {
			while (true) {
				Function &here = functions_[node];
				if (!is_used(node)) {
					here = std::move(added);
					used_[node >> 6] |= uint64(1) << (node & 63);
					return;
				}
				uint32 const middle = first + width / 2;
				Coordinate const x_middle = coordinate(middle), x_first = coordinate(first);
				bool const wins_middle = comp_(added(x_middle), here(x_middle));
				bool const wins_first = comp_(added(x_first), here(x_first));
				// crossing at most once, a function that wins (or loses) at both
				// ends of the node's range wins (or loses) on all of it: the
				// loser is dropped instead of walked down the tree
				if (wins_first == wins_middle && width > 1) {
					Coordinate const x_last = coordinate(first + width - 1);
					if (comp_(added(x_last), here(x_last)) == wins_middle) {
						if (wins_middle) here = std::move(added);
						return;
					}
				}
				if (wins_middle) std::swap(here, added);
				if (width == 1) return;
				width /= 2;
				// the function now in `added` loses at the midpoint; crossing at
				// most once, it can win only left of it if it wins at the first
				// point, else only right of it
				if (wins_first != wins_middle) {
					node = 2 * node;
				} else {
					node = 2 * node + 1;
					first = middle;
				}
			}
		}

		[[no_unique_address]] Compare comp_{};
		std::vector<Coordinate> coordinates_;
		std::vector<Function> functions_;
		// one bit per slot, kept apart from the functions so a query skips the
		// empty slots on its path without touching their (uncached) memory
		std::vector<uint64> used_;
		Coordinate lo_{};
		uint32 size_ = 0, leaves_ = 0;
		bool has_coordinates_ = false;
	};
}
