#pragma once
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>
#include <vector>

namespace atl::geometry {
	/**
	 * @brief A closed axis-aligned box given by its min and max corners
	 * (`lo.x <= hi.x`, `lo.y <= hi.y`; degenerate segments and points are
	 * legal). Aggregate; `box{lo, hi}` deduces `T`.
	 */
	template<coordinate T>
	struct box {
		/** @brief The coordinate type @p T. */
		using coordinate_type = T;

		point<T> lo{}; ///< The corner with the least coordinates.
		point<T> hi{}; ///< The corner with the greatest coordinates.

		/** @brief Memberwise equality of the two corners. */
		friend constexpr bool operator==(box const &, box const &) = default;
	};

	/** @brief A sized random-access range whose value type is some `box<T>`. */
	template<typename R>
	concept box_range =
		std::ranges::random_access_range<R> && std::ranges::sized_range<R>
		&& requires {
			   typename std::ranges::range_value_t<R>::coordinate_type;
			   requires std::same_as<
				   std::ranges::range_value_t<R>,
				   box<typename std::ranges::range_value_t<R>::coordinate_type>>;
		   };

	namespace detail {
		/**
		 * @brief A covered length: the unsigned difference type for integral
		 * coordinates (a difference of two coordinates always fits it), double
		 * for both floating coordinate types.
		 */
		template<coordinate T>
		using klee_length_t =
			typename std::conditional_t<std::floating_point<T>, std::type_identity<float64>, std::make_unsigned<T>>::type;

		/** @brief One Klee tree node: the covered length and the cover count. */
		template<coordinate T>
		struct klee_node {
			klee_length_t<T> covered;
			int cover;
		};

		/**
		 * @brief The x-sweep input of @ref atl::geometry::union_area — the sorted distinct y
		 *        values and, per box with area, an entry and an exit event
		 *        over its half-open strip range, sorted by x.
		 */
		template<coordinate T>
		struct klee_events {
			/** @brief One box edge: its x, +1 entering / -1 leaving, the strips. */
			struct event {
				T x;
				int delta;
				uint32 y_lo, y_hi; // half-open strip range
			};

			std::vector<T> ys;
			std::vector<event> events;
		};

		/**
		 * @brief Builds the `klee_events` of @p boxes, asserting the
		 *        @ref atl::geometry::union_area preconditions.
		 * @pre `0 < size(boxes) < 2^29`
		 */
		template<box_range R>
		constexpr klee_events<typename std::ranges::range_value_t<R>::coordinate_type> klee_events_of(R &&boxes) {
			using T = typename std::ranges::range_value_t<R>::coordinate_type;
			auto const n = std::ranges::size(boxes);
			atl_assert(uint64(n) < (uint64(1) << 29));
			auto const first = std::ranges::begin(boxes);
			klee_events<T> out;
			auto &ys = out.ys;
			ys.reserve(2 * n);
			for (std::size_t i = 0; i < n; ++i) {
				auto const b = box<T>(first[i]);
				atl_assert(b.lo.x <= b.hi.x && b.lo.y <= b.hi.y);
				if constexpr (std::signed_integral<T> && sizeof(T) == 8) {
					atl_assert(-(T(1) << 62) < b.lo.x && b.hi.x < T(1) << 62);
					atl_assert(-(T(1) << 62) < b.lo.y && b.hi.y < T(1) << 62);
				}
				ys.push_back(b.lo.y);
				ys.push_back(b.hi.y);
			}
			std::ranges::sort(ys);
			auto const dup = std::ranges::unique(ys);
			ys.erase(dup.begin(), dup.end());
			auto const slot_of = [&](T const y) { return uint32(std::ranges::lower_bound(ys, y) - ys.begin()); };
			out.events.reserve(2 * n);
			for (std::size_t i = 0; i < n; ++i) {
				auto const b = box<T>(first[i]);
				if (b.lo.x == b.hi.x || b.lo.y == b.hi.y) continue; // no area
				out.events.push_back({b.lo.x, +1, slot_of(b.lo.y), slot_of(b.hi.y)});
				out.events.push_back({b.hi.x, -1, slot_of(b.lo.y), slot_of(b.hi.y)});
			}
			std::ranges::sort(out.events, {}, &klee_events<T>::event::x);
			return out;
		}
	}

	/**
	 * @brief The area covered by AT LEAST ONE of the boxes (Klee's measure in
	 *        the plane): an x-sweep over box edges with a count-and-covered-
	 *        length segment tree on the compressed y-intervals, updated
	 *        iteratively bottom-up. Exact in @ref atl::geometry::difference_product_t for
	 *        integral coordinates; floating coordinates give a rounded
	 *        `float64` construction (lengths and area accumulate in `double`) with no exactness guarantee. Boxes of zero area contribute
	 *        nothing; an empty range gives 0.
	 * @pre Each box ordered (`lo <= hi` per axis, asserted). For `int64`
	 *      coordinates `|x|, |y| < 2^62` (asserted), so every covered
	 *      length and its product with an x-gap fit `int128`. Fewer than
	 *      2^29 boxes (asserted): the at most 2n compressed y values index a
	 *      tree of at most 8n nodes with `uint32`.
	 * @complexity O(n log n) time, O(n) memory
	 * @note The recursive top-down tree it was raced against took 27-40%
	 *       longer on every measured shape (Xeon Platinum 8358, GCC 15.2,
	 *       `-O2` and `-march=native`; tables on the guide page).
	 */
	template<box_range R>
	[[nodiscard]] constexpr difference_product_t<
		typename std::ranges::range_value_t<R>::coordinate_type>
	union_area(R &&boxes) {
		using T = typename std::ranges::range_value_t<R>::coordinate_type;
		using W = difference_product_t<T>;
		auto const n = std::ranges::size(boxes);
		if (n == 0) return {};
		auto const [ys, events] = detail::klee_events_of(boxes);
		if (events.empty()) return {};
		uint32 const slots = uint32(ys.size()) - 1; // y strips between values

		// per node the cover count and the covered length (children consulted
		// only when uncovered); lengths stay in the unsigned difference type
		// and are widened only at the area multiply. An update adds its delta
		// on the canonical nodes, then re-pulls the two boundary leaf-to-root
		// paths
		using L = detail::klee_length_t<T>;
		uint32 size = 1;
		while (size < slots) size *= 2;
		std::vector<L> length(2 * std::size_t(size), L{});
		// unsigned wraparound gives the exact difference of the sorted values
		for (uint32 k = 0; k < slots; ++k) length[size + k] = L(L(ys[k + 1]) - L(ys[k]));
		for (uint32 v = size - 1; v > 0; --v) length[v] = L(length[2 * v] + length[2 * v + 1]);
		std::vector<detail::klee_node<T>> tree(2 * std::size_t(size), {L{}, 0});
		auto const pull = [&](uint32 const v) {
			tree[v].covered = tree[v].cover > 0 ? length[v]
							: v >= size			? L{}
												: L(tree[2 * v].covered + tree[2 * v + 1].covered);
		};
		W area = {};
		T prev_x = events.front().x;
		for (auto const &e : events) {
			area += W(tree[1].covered) * (W(e.x) - W(prev_x));
			prev_x = e.x;
			uint32 const left = e.y_lo + size, right = e.y_hi - 1 + size;
			for (uint32 l = left, r = e.y_hi + size; l < r; l >>= 1, r >>= 1) {
				if (l & 1) {
					tree[l].cover += e.delta;
					pull(l++);
				}
				if (r & 1) {
					tree[--r].cover += e.delta;
					pull(r);
				}
			}
			for (uint32 v = left >> 1; v > 0; v >>= 1) pull(v);
			for (uint32 v = right >> 1; v > 0; v >>= 1) pull(v);
		}
		return area;
	}
}
