#pragma once
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <ranges>
#include <set>
#include <tuple>
#include <vector>

namespace atl::geometry {
	namespace detail {
		/** @brief A point paired with its index in the caller's input range. */
		template<coordinate T>
		struct closest_entry {
			point<T> position;
			vertex_index index;
		};

		/** @brief The closest pair found so far: its exact squared distance and its two indices. */
		template<coordinate T>
		struct closest_state {
			difference_product_t<T> best;
			vertex_index first, second;

			/**
			 * @brief Replaces the pair by `(x, y)` when it is STRICTLY closer,
			 *        so among equally close pairs the first one found stays.
			 * @pre For `int64` coordinates `|x|, |y| < 2^62` (asserted by the
			 *      squared distance).
			 */
			constexpr void consider(closest_entry<T> const &x, closest_entry<T> const &y) {
				auto const d = squared_distance(x.position, y.position);
				if (d < best) {
					best = d;
					first = x.index;
					second = y.index;
				}
			}
		};

		/** @brief `(x - y)^2` in the widened types (`x - y` can overflow `T`). */
		template<coordinate T>
		constexpr difference_product_t<T> squared_gap(T const x, T const y) {
			auto const d = difference_product_t<T>(widened_difference<product_t<T>>(x, y));
			return d * d;
		}
	}

	/**
	 * @brief Indices into @p points of a closest pair: an x-sweep over a
	 *        window of active points kept in a set ordered by y.
	 *
	 * All decisive comparisons are exact squared distances in the widened
	 * types, so for integral coordinates the pair is EXACTLY a closest one:
	 * the y window is an integer radius at least the square root of the
	 * best squared distance, compared in the widened type, so it never drops
	 * a true candidate. For floating coordinates the squared distances are
	 * rounded (differences, squares and their sum can each round in `double`,
	 * including for `float` inputs) and the window is the rounded-up root, which
	 * by monotone rounding also never drops a candidate.
	 *
	 * Equal points give distance 0 and end the sweep at once. Which of
	 * several equally close pairs is returned, and in which order its two
	 * indices come, is unspecified (it follows the order `std::ranges::sort`
	 * leaves equal points in).
	 * @pre `2 <= size(points) < 2^32` (asserted); for `int64` coordinates
	 *      `|x|, |y| < 2^62` (asserted for every point); floating coordinate/offset squares and pairwise products finite
	 *      and normal unless a factor is zero (coordinate squares asserted),
	 *      and finite sums.
	 * @note Not constexpr (`std::set`, `std::sqrt`). The constexpr-capable
	 *       divide and conquer in `atl/experimental/geometry/closest_pair_variants.hpp`
	 *       is slower on every measured shape but points on one vertical line.
	 * @complexity O(n log n) time, O(n) memory
	 */
	template<vertex_range R>
	[[nodiscard]] std::array<vertex_index, 2> closest_pair_indices(R &&points) {
		using T = vertex_coordinate_t<R>;
		atl_assert(std::ranges::size(points) >= 2);
		atl_assert(std::ranges::size(points) < (uint64(1) << 32));
		auto const first = std::ranges::begin(points);
		uint32 const n = uint32(std::ranges::size(points));
		std::vector<detail::closest_entry<T>> order;
		order.reserve(n);
		for (uint32 i = 0; i < n; ++i) {
			detail::check_difference_bounds(first[i]);
			detail::check_floating_magnitude(point<T>(first[i])); // the x gaps of every point are squared
			order.push_back({first[i], i});
		}
		std::ranges::sort(order, [](auto const &l, auto const &r) {
			return l.position.x != r.position.x ? l.position.x < r.position.x : l.position.y < r.position.y;
		});
		detail::closest_state<T> state{squared_distance(order[0].position, order[1].position), order[0].index,
									   order[1].index};
		// a y further than this from the current point is beyond best. For
		// integral coordinates an INTEGER radius with radius^2 >= best,
		// compared in the widened type: a float64 window would round away up
		// to 512 at |y| ~ 2^61 and drop true candidates
		using D = product_t<T>;
		auto const radius_bound = [&]() -> D {
			float64 const root = std::nextafter(std::sqrt(float64(state.best)), 1e308);
			if constexpr (std::floating_point<T>)
				return root;
			else
				return D(D(std::ceil(root * (1 + 0x1p-50))) + 1); // covers the rounding of the root
		};
		D window = radius_bound();
		std::set<std::tuple<T, T, uint32>> active; // (y, x, index)
		std::size_t left = 0;
		for (std::size_t at = 0; at < n; ++at) {
			if (state.best == difference_product_t<T>{}) break; // distance 0 cannot improve
			auto const &e = order[at];
			while (left < at && detail::squared_gap(e.position.x, order[left].position.x) >= state.best) {
				active.erase({order[left].position.y, order[left].position.x, order[left].index});
				++left;
			}
			// y in [e.y - window, e.y + window]; the gaps are not monotone
			// from below, so only the upper bound ends the scan
			D const y_low = D(D(e.position.y) - window);
			auto it = active.begin();
			if (y_low > D(std::numeric_limits<T>::lowest())) {
				T key = T(y_low);					  // exact for integral T
				if constexpr (std::floating_point<T>) // a float may have rounded up
					if (D(key) > y_low) key = std::nextafter(key, std::numeric_limits<T>::lowest());
				it = active.lower_bound({key, std::numeric_limits<T>::lowest(), 0});
			}
			for (; it != active.end() && D(std::get<0>(*it)) - D(e.position.y) <= window; ++it) {
				auto const incumbent = state.best;
				state.consider({{std::get<1>(*it), std::get<0>(*it)}, std::get<2>(*it)}, e);
				if (state.best < incumbent) window = radius_bound();
			}
			active.insert({e.position.y, e.position.x, e.index});
		}
		return {state.first, state.second};
	}

	/**
	 * @brief The two points of @ref atl::geometry::closest_pair_indices, in the same order.
	 * @pre As @ref atl::geometry::closest_pair_indices.
	 * @note Not constexpr (as @ref atl::geometry::closest_pair_indices).
	 * @complexity O(n log n)
	 */
	template<vertex_range R>
	[[nodiscard]] std::array<point<vertex_coordinate_t<R>>, 2> closest_pair(R &&points) {
		auto const [i, j] = closest_pair_indices(points);
		auto const first = std::ranges::begin(points);
		return {first[i], first[j]};
	}
}
