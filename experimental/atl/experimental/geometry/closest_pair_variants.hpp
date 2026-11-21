#pragma once
#include <atl/ds/flat_hash_map.hpp>
#include <atl/geometry/closest_pair.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <ranges>
#include <vector>

/**
 * @file
 * @brief The contenders raced against @ref atl::geometry::closest_pair_indices,
 *        the library's x-sweep over a y-ordered window: divide and conquer and
 *        a randomized grid. Both decide with exact widened squared distances,
 *        like the library; there is no point-returning twin, since
 *        `closest_pair` only looks the indices up.
 */
namespace atl::experimental {
	namespace detail {
		/** @brief The `(y, x)` order the divide and conquer merges by. */
		template<geometry::coordinate T>
		constexpr bool closest_by_y_less(geometry::detail::closest_entry<T> const &l,
										 geometry::detail::closest_entry<T> const &r) {
			return l.position.y != r.position.y ? l.position.y < r.position.y : l.position.x < r.position.x;
		}

		/**
		 * @brief Whether two coordinates can still be within the best
		 *        distance: for integral @p T an integer radius `r` with
		 *        `r^2 >= best` compared against the exact widened gap (no
		 *        multiply; a gap above `r` has a square above best), for
		 *        floating @p T, and in constant evaluation (where `std::sqrt`
		 *        is unavailable), the squared gap against best.
		 */
		template<geometry::coordinate T>
		struct closest_window {
			geometry::difference_product_t<T> best;
			geometry::product_t<T> radius;

			constexpr explicit closest_window(geometry::difference_product_t<T> const best_distance)
				: best(best_distance), radius{} {
				if !consteval {
					if constexpr (!std::floating_point<T>) {
						using D = geometry::product_t<T>;
						float64 const root = std::sqrt(float64(best));
						radius = D(D(std::ceil(root * (1 + 0x1p-50))) + 1); // covers the rounding of the root
					}
				}
			}

			constexpr bool reaches(T const a, T const b) const {
				if consteval {
					return geometry::detail::squared_gap(a, b) < best;
				} else {
					if constexpr (std::floating_point<T>)
						return geometry::detail::squared_gap(a, b) < best;
					else {
						using D = geometry::product_t<T>;
						D const gap = geometry::detail::widened_difference<D>(a, b);
						return (gap < 0 ? D(-gap) : gap) <= radius;
					}
				}
			}
		};

		/**
		 * @brief Divide and conquer over `by_x[lo, hi)` (sorted by (x, y));
		 *        leaves `out[lo, hi)` sorted by (y, x), using `scratch[lo, hi)`
		 *        as the children's output, so no level copies back.
		 *
		 * Leaves of up to 16 points are brute-forced and insertion-sorted,
		 * the strip filter compares an integer radius instead of squaring
		 * each gap, and every level returns at once when the best distance
		 * is 0 (the parent then skips its merge too). The halving keeps the
		 * recursion depth at O(log n).
		 * @pre `out` and `scratch` have at least `hi` entries; `state` is
		 *      seeded with a real pair
		 * @complexity O((hi - lo) log(hi - lo))
		 */
		template<geometry::coordinate T>
		constexpr void closest_pair_recurse(std::vector<geometry::detail::closest_entry<T>> const &by_x,
											std::vector<geometry::detail::closest_entry<T>> &out,
											std::vector<geometry::detail::closest_entry<T>> &scratch,
											std::vector<uint32> &strip, std::size_t const lo,
											std::size_t const hi,
											geometry::detail::closest_state<T> &state) {
			using P = geometry::difference_product_t<T>;
			if (state.best == P{}) return;
			if (hi - lo <= 16) {
				for (std::size_t i = lo; i < hi; ++i)
					for (std::size_t j = i + 1;
						 j < hi && geometry::detail::squared_gap(by_x[j].position.x, by_x[i].position.x) < state.best; ++j)
						state.consider(by_x[i], by_x[j]);
				for (std::size_t i = lo; i < hi; ++i) {
					auto const e = by_x[i];
					std::size_t k = i;
					for (; k > lo && closest_by_y_less(e, out[k - 1]); --k) out[k] = out[k - 1];
					out[k] = e;
				}
				return;
			}
			std::size_t const mid = lo + (hi - lo) / 2;
			T const mid_x = by_x[mid].position.x;
			closest_pair_recurse(by_x, scratch, out, strip, lo, mid, state);
			closest_pair_recurse(by_x, scratch, out, strip, mid, hi, state);
			if (state.best == P{}) return;
			std::ranges::merge(scratch.begin() + std::ptrdiff_t(lo), scratch.begin() + std::ptrdiff_t(mid),
							   scratch.begin() + std::ptrdiff_t(mid), scratch.begin() + std::ptrdiff_t(hi),
							   out.begin() + std::ptrdiff_t(lo), closest_by_y_less<T>);
			// the strip: candidates within the current best of the split
			// line, in y order; each needs only the next few (<= 7) above it.
			// The window may go stale as best shrinks: it only over-admits
			closest_window<T> const window(state.best);
			strip.clear();
			for (std::size_t k = lo; k < hi; ++k)
				if (window.reaches(out[k].position.x, mid_x)) strip.push_back(uint32(k));
			for (std::size_t k = 0; k < strip.size(); ++k)
				for (std::size_t m = k + 1;
					 m < strip.size() && window.reaches(out[strip[m]].position.y, out[strip[k]].position.y); ++m)
					state.consider(out[strip[k]], out[strip[m]]);
		}
	}

	/**
	 * @brief @ref atl::geometry::closest_pair_indices by divide and conquer
	 *        instead of a sweep: a recursion on the x-sorted points with the
	 *        mergesort by y riding along (ping-pong buffers, 16-point brute
	 *        force leaves, an integer strip radius, a return at distance 0),
	 *        and constexpr-capable.
	 *
	 * Lost the race (Xeon Platinum 8280, GCC 15.2, 2026-10-15, 1e6 int64
	 * points, plain `-O2` / `-march=native`): 290 / 275 ms against 162 / 163
	 * for the library on uniform points, 314 / 301 against 161 / 160 on a
	 * Gaussian cluster without duplicates, 317 / 311 against 283 / 280 on a
	 * thin strip. It wins only with every point on one vertical line, 261 /
	 * 264 against 519 / 516, where the sweep's window holds every point: a
	 * shape not worth a slower common case.
	 * @pre `2 <= size(points) < 2^32` (asserted); for `int64` coordinates
	 *      `|x|, |y| < 2^62` (asserted). Floating magnitudes as for
	 *      @ref atl::geometry::closest_pair_indices.
	 * @complexity O(n log n) time, O(n) memory, recursion depth O(log n)
	 */
	template<geometry::vertex_range R>
	[[nodiscard]] constexpr std::array<geometry::vertex_index, 2> closest_pair_indices_by_divide_and_conquer(R &&points) {
		using T = geometry::vertex_coordinate_t<R>;
		atl_assert(std::ranges::size(points) >= 2);
		atl_assert(std::ranges::size(points) < (uint64(1) << 32));
		auto const first = std::ranges::begin(points);
		uint32 const n = uint32(std::ranges::size(points));
		std::vector<geometry::detail::closest_entry<T>> by_x;
		by_x.reserve(n);
		for (uint32 i = 0; i < n; ++i) {
			geometry::detail::check_difference_bounds(first[i]);
			geometry::detail::check_floating_magnitude(geometry::point<T>(first[i])); // the x gaps of every point are squared
			by_x.push_back({first[i], i});
		}
		std::ranges::sort(by_x, [](auto const &l, auto const &r) {
			return l.position.x != r.position.x ? l.position.x < r.position.x : l.position.y < r.position.y;
		});
		std::vector<geometry::detail::closest_entry<T>> by_y(n), scratch(n);
		std::vector<uint32> strip;
		geometry::detail::closest_state<T> state{squared_distance(by_x[0].position, by_x[1].position),
												 by_x[0].index, by_x[1].index};
		detail::closest_pair_recurse(by_x, by_y, scratch, strip, 0, n, state);
		return {state.first, state.second};
	}

	/**
	 * @brief @ref atl::geometry::closest_pair_indices by a Rabin-style
	 *        randomized grid instead of a sweep: points inserted in a seeded
	 *        random order into a hash grid whose power-of-two cell side covers the
	 *        running best distance, so each insertion
	 *        probes a bounded neighborhood; every improvement rebuilds the
	 *        grid at the new scale.
	 *
	 * Lost the race (Xeon Platinum 8280, GCC 15.2, 2026-10-15, 1e6 int64
	 * points, plain `-O2` / `-march=native`): 1201 / 1126 ms against 162 /
	 * 163 for the library on uniform points, the rebuilds dominating. On a
	 * Gaussian cluster with duplicates it can stop at the first distance 0,
	 * 144 / 52.8 ms against 110 / 111, but with a run spread of 1.74x /
	 * 4.20x: the shuffle decides how soon.
	 *
	 * Cells are computed in `float64`, so integral coordinates must convert
	 * exactly: beyond 2^53 the conversion rounds, two points closer than the
	 * running best can land in cells that are not adjacent, and the true
	 * pair is missed (a test near 2^60 found it). The library has no such
	 * limit.
	 * @pre `2 <= size(points) < 2^32` (asserted); for `int64` coordinates
	 *      `|x|, |y| <= 2^53` (asserted). Floating magnitudes as for
	 *      @ref atl::geometry::closest_pair_indices.
	 * @note Not constexpr (`std::floor`, `std::sqrt`, the hash map).
	 * @complexity Expected O(n) only when the minimum cell side and saturated
	 *             indices do not merge unboundedly many nearby points; O(n^2)
	 *             worst case, O(n) memory. `atl::set_hash_seed` pins the shuffle.
	 */
	template<geometry::vertex_range R>
	[[nodiscard]] std::array<geometry::vertex_index, 2> closest_pair_indices_by_randomized_grid(R &&points) {
		using T = geometry::vertex_coordinate_t<R>;
		atl_assert(std::ranges::size(points) >= 2);
		atl_assert(std::ranges::size(points) < (uint64(1) << 32));
		if constexpr (std::integral<T> && sizeof(T) == 8) { // narrower integers convert exactly
			constexpr T limit = T(1) << 53;
			[[maybe_unused]] auto const is_exact = [&](geometry::point<T> const p) {
				return -limit <= p.x && p.x <= limit && -limit <= p.y && p.y <= limit;
			};
			atl_assert(std::ranges::all_of(points, is_exact));
		}
		auto const first = std::ranges::begin(points);
		uint32 const n = uint32(std::ranges::size(points));
		std::vector<geometry::detail::closest_entry<T>> order;
		order.reserve(n);
		for (uint32 i = 0; i < n; ++i) {
			geometry::detail::check_floating_magnitude(geometry::point<T>(first[i]));
			order.push_back({first[i], i});
		}
		std::mt19937_64 rng(atl::detail::hash_seed() ^ 0xc105e57ULL);
		std::ranges::shuffle(order, rng);
		geometry::detail::closest_state<T> state{
			squared_distance(order[0].position, order[1].position), order[0].index, order[1].index};

		float64 side = 1;
		auto const fold = [](float64 const v) { // saturating floor-to-cell index
			float64 const clamped = v < -9e18 ? -9e18 : v > 9e18 ? 9e18
																 : v;
			return int64(std::floor(clamped));
		};
		auto const cell_of = [&](geometry::point<T> const p) {
			return atl::detail::splitmix64(uint64(fold(float64(p.x) / side)))
				 ^ (atl::detail::splitmix64(uint64(fold(float64(p.y) / side)) + 0x9e3779b97f4a7c15ULL));
		};
		// hash-key collisions between distinct cells are harmless: chained
		// entries are only ever candidates for EXACT distance checks. Values
		// are 1-based order indices; 0 ends a chain
		flat_hash_map<uint64, uint32> head;
		std::vector<uint32> chain(n, 0);
		auto const push = [&](std::size_t const k) {
			uint32 &slot = head[cell_of(order[k].position)];
			chain[k] = slot;
			slot = uint32(k) + 1;
		};
		auto const rebuild_upto = [&](std::size_t const count) {
			head.clear();
			int exponent = 0;
			(void)std::frexp(std::sqrt(float64(state.best)), &exponent);
			// Scaling by a power of two is exact before floor (unless tiny
			// values underflow near cell zero, where the two-cell margin covers
			// them). The side covers the root, including its rounding error.
			side = (std::max)(std::ldexp(1.0, exponent), 1e-30);
			// The minimum side must also preserve exact power-of-two division.
			(void)std::frexp(side, &exponent);
			side = std::ldexp(1.0, exponent);
			for (std::size_t k = 0; k < count; ++k) push(k);
		};
		rebuild_upto(2);
		for (std::size_t at = 2; at < n; ++at) {
			if (state.best == geometry::difference_product_t<T>{}) break; // distance 0 cannot improve
			auto const &e = order[at];
			auto const incumbent = state.best;
			int64 const cx = fold(float64(e.position.x) / side);
			int64 const cy = fold(float64(e.position.y) / side);
			for (int dx = -2; dx <= 2; ++dx)
				for (int dy = -2; dy <= 2; ++dy) {
					uint64 const key =
						atl::detail::splitmix64(uint64(cx + dx))
						^ (atl::detail::splitmix64(uint64(cy + dy) + 0x9e3779b97f4a7c15ULL));
					for (uint32 walk = head.at_or(key, 0u); walk != 0; walk = chain[walk - 1])
						state.consider(order[walk - 1], e);
				}
			if (state.best < incumbent)
				rebuild_upto(at + 1);
			else
				push(at);
		}
		return {state.first, state.second};
	}
}
