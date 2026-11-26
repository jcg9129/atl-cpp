#pragma once

/** @file
 * @brief Shared convex hull implementation without adaptive floating predicates.
 */
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon_core.hpp>
#include <atl/geometry/predicates_core.hpp>
#include <atl/geometry/segment_core.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::geometry {
	/**
	 * @brief Tag selecting the hull variant that KEEPS collinear boundary
	 *        points (the default is the strict hull: extreme vertices only).
	 */
	struct keep_collinear_t {
		/** @brief Creates the boundary-retention tag. */
		explicit keep_collinear_t() = default;
	};

	/** @brief The @ref atl::geometry::keep_collinear_t tag value. */
	inline constexpr keep_collinear_t keep_collinear{};

	namespace detail {
		template<coordinate T>
		struct hull_entry {
			point<T> position;
			uint32 index;
		};

		/**
		 * @brief Monotone-chain core over lexicographically sorted, deduplicated
		 * entries. Returns ORIGINAL indices, CCW from the smallest point.
		 * @pre @p d sorted lexicographically by position and deduplicated;
		 *      `size(d) < 2^32` (asserted).
		 */
		template<bool IsStrict, coordinate T, sign_policy<T> Policy>
		[[nodiscard]] constexpr std::vector<vertex_index> chain_hull(std::vector<hull_entry<T>> const &d, Policy const p) {
			atl_assert(uint64(d.size()) < (uint64(1) << 32));
			uint32 const m = uint32(d.size());
			std::vector<vertex_index> hull;
			if (m <= 2) {
				for (auto const &e : d) hull.push_back(e.index);
				return hull;
			}
			bool is_collinear = true;
			for (uint32 i = 1; is_collinear && i + 1 < m; ++i)
				is_collinear = p.orientation(d[0].position, d[m - 1].position, d[i].position) == 0;
			if (is_collinear) { // one line: endpoints, or every distinct point
				if constexpr (IsStrict) return {d[0].index, d[m - 1].index};
				for (auto const &e : d) hull.push_back(e.index);
				return hull;
			}
			// both chains in one buffer of positions into d (the upper chain
			// stacks on the lower one's last point); strict pops non-left
			// turns, keep-collinear pops right turns only
			if (d.size() > hull.max_size() / 2) throw std::length_error("convex hull is too large");
			hull.resize(2 * std::size_t(m));
			auto const pops = [&](uint32 const a, uint32 const b, uint32 const c) {
				int const o = p.orientation(d[a].position, d[b].position, d[c].position);
				return IsStrict ? o <= 0 : o < 0;
			};
			std::size_t k = 0;
			for (uint32 at = 0; at < m; ++at) {
				while (k >= 2 && pops(hull[k - 2], hull[k - 1], at)) --k;
				hull[k++] = at;
			}
			std::size_t const lower_end = k + 1; // the upper chain keeps >= 2
			for (uint32 at = m - 1; at-- > 0;) {
				while (k >= lower_end && pops(hull[k - 2], hull[k - 1], at)) --k;
				hull[k++] = at;
			}
			hull.resize(k - 1); // its last is the first point again
			for (auto &at : hull) at = d[at].index;
			return hull;
		}

		/**
		 * @brief Akl–Toussaint pruning: points strictly inside the convex polygon
		 * of the 8 directional extremes cannot be on the hull (nor on its
		 * boundary), so they are dropped before the sort. A rectangle inscribed
		 * in the four diagonal extremes accepts most interior points with
		 * comparisons only, and a ~1k-point sample decides first, on its own
		 * octagon, whether the full pass pays (x86-64 Xeon 8280 -O2, 1e6 int64, record
		 * 2026-10-15: uniform 12.1 ms vs 136 unpruned, circle 132 vs 133).
		 * @pre @p entries non-empty (indexes `entries[0]`).
		 */
		template<coordinate T, sign_policy<T> Policy>
		constexpr void prune_octagon(std::vector<hull_entry<T>> &entries, Policy const p) {
			// x + y and x - y stay within int64 under the |c| < 2^62
			// precondition — no int128 needed for the extremes scan
			using K = std::conditional_t<std::floating_point<T>, float64, int64>;
			auto const keys = [](point<T> const q) {
				return std::array<K, 4>{K(q.x), K(q.y), K(q.x) + K(q.y), K(q.x) - K(q.y)};
			};
			uint32 const n = uint32(entries.size());
			auto const extremes_of = [&](uint32 const stride) {
				std::array<uint32, 8> extreme{};
				extreme.fill(0);
				auto best = keys(entries[0].position);
				std::array<K, 4> worst = best;
				for (uint64 e = stride; e < n; e += stride) {
					auto const k = keys(entries[std::size_t(e)].position);
					for (std::size_t axis = 0; axis < 4; ++axis) {
						if (k[axis] < best[axis]) best[axis] = k[axis], extreme[2 * axis] = uint32(e);
						if (k[axis] > worst[axis]) worst[axis] = k[axis], extreme[2 * axis + 1] = uint32(e);
					}
				}
				return extreme;
			};

			// any polygon of input points is a valid pruning polygon: its
			// edges, closed (edge[size] == edge[0]), CCW, plus a rectangle
			// strictly inside it that accepts without a multiplication
			struct pruning_polygon {
				std::array<point<T>, 9> corner{};
				uint32 size = 0;
				T x_low{}, x_high{}, y_low{}, y_high{};
			};

			auto const polygon_of = [&](std::array<uint32, 8> const &extreme) {
				pruning_polygon poly;
				std::vector<hull_entry<T>> corners;
				for (uint32 const e : extreme) corners.push_back({entries[e].position, 0});
				std::ranges::sort(corners, {}, &hull_entry<T>::position);
				auto const [dead0, dead1] = std::ranges::unique(corners, {}, &hull_entry<T>::position);
				corners.erase(dead0, dead1);
				for (uint32 k = 0; k < uint32(corners.size()); ++k) corners[k].index = k;
				auto const octagon = chain_hull<true>(corners, p);
				poly.size = uint32(octagon.size());
				for (uint32 k = 0; k < poly.size; ++k) poly.corner[k] = corners[octagon[k]].position;
				poly.corner[poly.size] = poly.corner[0];
				// the rectangle between the four diagonal extremes NE (max
				// x + y), NW (min x - y), SW (min x + y), SE (max x - y): its
				// open interior lies strictly inside their quadrilateral, so
				// strictly inside the octagon; empty when they degenerate
				point<T> const sw = entries[extreme[4]].position, ne = entries[extreme[5]].position;
				point<T> const nw = entries[extreme[6]].position, se = entries[extreme[7]].position;
				poly.x_low = (std::max)(sw.x, nw.x);
				poly.x_high = (std::min)(ne.x, se.x);
				poly.y_low = (std::max)(sw.y, se.y);
				poly.y_high = (std::min)(ne.y, nw.y);
				return poly;
			};
			auto const inside = [&](pruning_polygon const &poly, point<T> const q) {
				if (poly.x_low < q.x && q.x < poly.x_high && poly.y_low < q.y && q.y < poly.y_high) return true;
				for (uint32 e = 0; e < poly.size; ++e)
					if (p.orientation(poly.corner[e], poly.corner[e + 1], q) <= 0) return false;
				return true;
			};
			// adaptive gate: on hull-heavy inputs (points on a circle) the
			// full pass would discard nothing and only pay its own cost, so
			// probe a ~1k sample with the sample's own octagon first and bail
			// out, before the full extremes scan, unless it discards half
			uint32 const stride = std::max<uint32>(1, n / 1024);
			auto const probe = polygon_of(extremes_of(stride));
			if (probe.size < 3) return;
			uint32 sampled = 0, discarded = 0;
			for (uint64 e = 0; e < n; e += stride) {
				++sampled;
				discarded += inside(probe, entries[std::size_t(e)].position);
			}
			if (2 * discarded < sampled) return;
			auto const octagon = stride == 1 ? probe : polygon_of(extremes_of(1));
			std::erase_if(entries, [&](hull_entry<T> const &e) { return inside(octagon, e.position); });
		}

		/**
		 * @brief Shared body of the hull entry points: pairs each point with
		 *        its index, optionally prunes (@p should_prune, and only from
		 *        256 points on), sorts and deduplicates exactly, then runs
		 *        `chain_hull` — strict when @p IsStrict, keep-collinear
		 *        otherwise.
		 * @pre `size(points) < 2^32` (asserted).
		 */
		template<bool IsStrict, vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy>
		[[nodiscard]] constexpr std::vector<vertex_index> hull_indices(R &&points, Policy const p, bool const should_prune) {
			using T = vertex_coordinate_t<R>;
			atl_assert(std::ranges::size(points) < (uint64(1) << 32));
			uint32 const n = uint32(std::ranges::size(points));
			auto const first = std::ranges::begin(points);
			std::vector<hull_entry<T>> entries;
			entries.reserve(n);
			for (uint32 i = 0; i < n; ++i) {
				point<T> const q = first[i];
				check_difference_bounds(q);
				entries.push_back({q, i});
			}
			if (should_prune && n >= 256) prune_octagon(entries, p);
			// exact lexicographic sort (raw comparisons are exact for both
			// integral and floating coordinates), index as tiebreak so
			// duplicates keep their smallest original index
			std::ranges::sort(entries, [](hull_entry<T> const &a, hull_entry<T> const &b) {
				return a.position != b.position ? a.position < b.position : a.index < b.index;
			});
			// raw == matches the exact sort; a policy's equal may not be
			// transitive (approx), and then would merge only lex-adjacent pairs
			auto const [dead0, dead1] = std::ranges::unique(entries, {}, &hull_entry<T>::position);
			entries.erase(dead0, dead1);
			return chain_hull<IsStrict>(entries, p);
		}
	}

	/**
	 * @brief Indices of the STRICT convex hull of @p points (extreme
	 *        vertices only, no collinear boundary points), CCW, starting at
	 *        the lexicographically smallest point. Duplicates (exact `==`,
	 *        whatever the policy) collapse to
	 *        their smallest index; degenerate inputs return what exists
	 *        (empty / single / pair; all-collinear input returns the two
	 *        endpoints). Andrew's monotone chain over an exact
	 *        lexicographic sort, preceded by Akl–Toussaint octagon pruning
	 *        (a measured win, see docs).
	 * @tparam R Sized random-access point range.
	 * @tparam Policy Sign policy for the range coordinate type.
	 * @param points Input cloud, copied without modification.
	 * @param p Predicate policy; duplicate equality remains raw ==.
	 * @return Original input indices in hull traversal order.
	 * @pre Fewer than 2^32 input points (asserted); for int64 coordinates, `|x|, |y| < 2^62` (asserted).
	 * @pre Floating coordinates satisfy the finite, normal product bounds of exact.
	 * @complexity O(n log n) time, O(n) extra memory; no per-point recursion.
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] constexpr std::vector<vertex_index> convex_hull_indices(R &&points, Policy const p = {}) {
		return detail::hull_indices<true>(points, p, true);
	}

	/**
	 * @brief The variant that KEEPS collinear boundary points: every distinct
	 *        input point lying on the hull boundary appears, in CCW order
	 *        (all-collinear input returns all distinct points along the line).
	 * @param points Input cloud; not modified.
	 * @param p Predicate policy.
	 * @return Indices CCW from lexicographic minimum, smallest index per duplicate position.
	 * @pre Same size and coordinate bounds as the strict overload.
	 * @complexity O(n log n) time, O(n) extra memory.
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] constexpr std::vector<vertex_index> convex_hull_indices(R &&points, keep_collinear_t,
																		  Policy const p = {}) {
		return detail::hull_indices<false>(points, p, true);
	}

	/** @brief Strict hull as input points, with the ordering and bounds of convex_hull_indices.
	 * @param points Input cloud; not modified.
	 * @param p Predicate policy.
	 * @return Extreme vertices CCW from the lexicographic minimum.
	 * @complexity O(n log n) time, O(n) extra memory.
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] constexpr std::vector<point<vertex_coordinate_t<R>>> convex_hull(R &&points, Policy const p = {}) {
		auto const first = std::ranges::begin(points);
		std::vector<point<vertex_coordinate_t<R>>> hull;
		for (vertex_index const i : convex_hull_indices(points, p)) hull.push_back(first[i]);
		return hull;
	}

	/** @brief Hull boundary as distinct input points, including straight vertices.
	 * @param points Input cloud; not modified.
	 * @param p Predicate policy.
	 * @return Boundary vertices with the ordering and bounds of convex_hull_indices with keep_collinear.
	 * @complexity O(n log n) time, O(n) extra memory.
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] constexpr std::vector<point<vertex_coordinate_t<R>>> convex_hull(R &&points, keep_collinear_t,
																				   Policy const p = {}) {
		auto const first = std::ranges::begin(points);
		std::vector<point<vertex_coordinate_t<R>>> hull;
		for (vertex_index const i : convex_hull_indices(points, keep_collinear, p)) hull.push_back(first[i]);
		return hull;
	}

	/**
	 * @brief Locates @p q relative to a STRICTLY CONVEX polygon in CCW
	 *        order (what @ref atl::geometry::convex_hull produces) in O(log n): a wedge
	 *        binary search on the fan from vertex 0, then one edge sign.
	 *        Exact for integral coordinates and adaptive-exact for
	 *        `double`, boundary included — vertices and edge points report
	 *        `containment::boundary`. Single points and 2-vertex segment
	 *        hulls are legal (`boundary`/`outside` only). The general
	 *        `contains(poly, q)` stays the O(n) any-simple-polygon test.
	 * @pre Strictly convex, counterclockwise (NOT asserted — verifying
	 *      costs O(n) and would defeat the query bound; feed `convex_hull`
	 *      output or `normalized_polygon` of a convex polygon).
	 *      `size >= 1` (asserted); for `int64`: `|x|, |y| < 2^62`.
	 * @param hull Nonempty strict CCW polygon.
	 * @param q Query point.
	 * @param p Predicate policy.
	 * @return Position relative to the closed hull region.
	 * @complexity O(log n) time and O(1) memory.
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] constexpr containment convex_contains(R &&hull, point<vertex_coordinate_t<R>> const q,
														Policy const p = {}) {
		using T = vertex_coordinate_t<R>;
		auto const n = std::ranges::size(hull);
		atl_assert(n >= 1);
		auto const first = std::ranges::begin(hull);
		auto const at = [&](std::size_t const i) -> point<T> { return first[i]; };
		auto const on_edge = [&](point<T> const a, point<T> const b) {
			return contains(segment<T>{a, b}, q, p) ? containment::boundary
													: containment::outside;
		};
		if (n == 1) return p.equal(at(0), q) ? containment::boundary : containment::outside;
		if (n == 2) return on_edge(at(0), at(1));
		int const versus_first = p.orientation(at(0), at(1), q);
		if (versus_first == 0) return on_edge(at(0), at(1));
		int const versus_last = p.orientation(at(0), at(n - 1), q);
		if (versus_last == 0) return on_edge(at(0), at(n - 1));
		if (versus_first < 0 || versus_last > 0) return containment::outside;
		// strictly inside the fan wedge: the last diagonal with q on its left
		std::size_t low = 1, high = n - 2;
		while (low < high) {
			std::size_t const mid = low + (high - low + 1) / 2;
			if (p.orientation(at(0), at(mid), q) > 0)
				low = mid;
			else
				high = mid - 1;
		}
		return containment(p.orientation(at(low), at(low + 1), q));
	}

	namespace detail {
		/**
		 * @brief A random-access iterator over `point<T>` values, `T` totally
		 *        ordered: what the merge's chain machinery walks.
		 */
		template<typename It>
		concept point_iterator =
			std::random_access_iterator<It> && is_point<std::remove_cvref_t<std::iter_value_t<It>>>::value
			&& std::totally_ordered<std::remove_cvref_t<std::iter_value_t<It>>>;

		/**
		 * @brief Whether @p hull is in the form @ref atl::geometry::convex_hull returns: distinct
		 *        vertices, CCW with every turn strictly left, starting at the
		 *        lexicographically smallest vertex, lexicographically increasing
		 *        up to the largest and decreasing after it (which, with the left
		 *        turns, rules out a polygon winding more than once); `int64`
		 *        coordinates within `|c| < 2^62`. The O(h) check behind the
		 *        @ref atl::geometry::merge_convex_hulls precondition assert.
		 */
		template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy>
		[[nodiscard]] constexpr bool is_lexicographic_strict_hull(R &&hull, Policy const p) {
			using T = vertex_coordinate_t<R>;
			std::size_t const n = std::ranges::size(hull);
			auto const first = std::ranges::begin(hull);
			auto const at = [&](std::size_t const i) -> point<T> { return first[i]; };
			if constexpr (std::signed_integral<T> && sizeof(T) == 8) {
				constexpr T bound = T(1) << 62;
				if (!std::ranges::all_of(hull, [](point<T> const q) {
						return -bound < q.x && q.x < bound && -bound < q.y && q.y < bound;
					}))
					return false;
			}
			if (n <= 1) return true;
			std::size_t top = 0;
			while (top + 1 < n && at(top) < at(top + 1)) ++top;
			for (std::size_t i = top; i + 1 < n; ++i)
				if (!(at(i + 1) < at(i))) return false;
			if (!(at(0) < at(n - 1))) return false;
			if (n == 2) return true;
			// chain_hull never tests the turns where its chains join (vertex 0
			// and top), so a non-exact policy's own output may fail them there
			for (std::size_t i = 0; i < n; ++i) {
				std::size_t const middle = (i + 1) % n;
				if constexpr (!std::same_as<Policy, exact>) {
					if (middle == 0 || middle == top) continue;
				}
				if (p.orientation(at(i), at(middle), at((i + 2) % n)) <= 0) return false;
			}
			return true;
		}

		/**
		 * @brief One monotone chain of a strict hull, in sweep order: the
		 *        positions `[begin, end)` of the hull's vertex range, then, when
		 *        `is_closed`, the hull's first vertex again (the upper chain
		 *        runs from the lexicographic maximum back to the minimum), so
		 *        `operator[]` maps position `end - begin` to vertex 0.
		 * @pre `begin < end`, both within the range at `first`.
		 */
		template<point_iterator It>
		struct hull_chain {
			It first;
			std::size_t begin, end;
			bool is_closed;

			constexpr std::size_t size() const { return end - begin + is_closed; }

			constexpr std::iter_value_t<It> operator[](std::size_t const k) const { return first[begin + k == end ? 0 : begin + k]; }
		};

		/**
		 * @brief Merges two chains that are monotone in the same direction
		 *        (lexicographically increasing when @p IsLower, decreasing
		 *        otherwise) and runs the strict monotone-chain stack over the
		 *        merged stream, appending to @p out; pops stop at size
		 *        @p floor. Equal points across the chains are consumed as one.
		 *
		 * Two shortcuts skip most stack work. A point lexicographically between
		 * two consecutive vertices of the OTHER chain, and on or left of the
		 * edge joining them, cannot be a strict vertex of this side of the
		 * union's hull (the hull's chain runs on or right of that edge across
		 * the point), so one orientation drops it without touching the stack.
		 * And consecutive vertices of one chain still adjacent on the stack
		 * turn strictly left by the input's strictness, so the next vertex of
		 * such a run is pushed with no orientation at all.
		 * @pre `floor >= 2`; @p out empty, or (upper pass) holding the lower
		 *      chain without its last vertex, `floor` its size plus 2.
		 * @complexity O(|a| + |b|)
		 */
		template<bool IsLower, coordinate T, point_iterator ItA, point_iterator ItB, sign_policy<T> Policy>
		constexpr void merge_hull_chains(hull_chain<ItA> const a, hull_chain<ItB> const b, std::size_t const floor,
										 std::vector<point<T>> &out, Policy const p) {
			auto const precedes = [](point<T> const l, point<T> const r) { return IsLower ? l < r : r < l; };
			std::size_t const na = a.size(), nb = b.size();
			std::size_t i = 0, j = 0;
			// the chain the top of the stack came from (2: a shared vertex, -1:
			// none, or a point of its chain was dropped since), and whether the
			// vertex under it is its chain predecessor
			int top_chain = -1;
			bool is_run = false;
			while (i < na || j < nb) {
				point<T> q{};
				int chain = 0;
				if (j == nb)
					q = a[i++], chain = 0;
				else if (i == na)
					q = b[j++], chain = 1;
				else {
					point<T> const qa = a[i], qb = b[j];
					if (precedes(qa, qb)) {
						++i;
						if (j > 0 && p.orientation(b[j - 1], qb, qa) >= 0) {
							if (top_chain == 0) top_chain = -1;
							continue;
						}
						q = qa, chain = 0;
					} else if (precedes(qb, qa)) {
						++j;
						if (i > 0 && p.orientation(a[i - 1], qa, qb) >= 0) {
							if (top_chain == 1) top_chain = -1;
							continue;
						}
						q = qb, chain = 1;
					} else
						q = qa, ++i, ++j, chain = 2;
				}
				bool const continues = chain == top_chain && chain != 2;
				if (continues && is_run) {
					out.push_back(q);
					continue;
				}
				bool has_popped = false;
				while (out.size() >= floor && p.orientation(out[out.size() - 2], out.back(), q) <= 0) {
					out.pop_back();
					has_popped = true;
				}
				is_run = continues && !has_popped;
				top_chain = chain;
				out.push_back(q);
			}
		}

		/**
		 * @brief Position of the lexicographically largest vertex of a strict
		 *        hull from its smallest: the vertices increase up to it and
		 *        decrease after, so a binary search finds it.
		 * @pre `n >= 1`, the hull in @ref atl::geometry::convex_hull form.
		 * @complexity O(log n)
		 */
		template<point_iterator It>
		[[nodiscard]] constexpr std::size_t lexicographic_top(It const first, std::size_t const n) {
			std::size_t low = 0, high = n - 1;
			while (low < high) {
				std::size_t const mid = low + (high - low) / 2;
				if (std::iter_value_t<It>(first[mid]) < std::iter_value_t<It>(first[mid + 1]))
					low = mid + 1;
				else
					high = mid;
			}
			return low;
		}
	}

	namespace detail {
		template<point_iterator It>
		using iterator_coordinate_t = typename std::remove_cvref_t<std::iter_value_t<It>>::coordinate_type;

		/**
		 * @brief The last position of the strictly convex chain @p x that the
		 *        monotone-chain stack keeps when @p q, which comes after every
		 *        vertex of @p x in sweep order, is pushed onto it: the edges
		 *        @p q pops (`orientation(x[k], x[k + 1], q) <= 0`) form a suffix
		 *        of the chain, so a binary search finds where they start.
		 * @pre `x.size() >= 1`; @p q after every vertex of @p x.
		 * @complexity O(log |x|)
		 */
		template<point_iterator It, coordinate T, sign_policy<T> Policy>
			requires std::same_as<T, iterator_coordinate_t<It>>
		[[nodiscard]] constexpr std::size_t chain_tangent_from_after(hull_chain<It> const x,
																	 point<T> const q, Policy const p) {
			atl_assert(x.size() >= 1);
			std::size_t low = 0, high = x.size() - 1;
			while (low < high) {
				std::size_t const mid = low + (high - low) / 2;
				if (p.orientation(x[mid], x[mid + 1], q) <= 0)
					high = mid;
				else
					low = mid + 1;
			}
			return low;
		}

		/**
		 * @brief The bridge between two strictly convex chains monotone in the
		 *        same direction, every vertex of @p x before every vertex of
		 *        @p y: `{t, s}` such that the strict monotone chain of the
		 *        stream x, y is `x[0..t]` followed by `y[s..]`. `y[k]` is dropped
		 *        exactly when it does not turn strictly left from its tangent
		 *        point on @p x to `y[k + 1]`, and the dropped vertices are a
		 *        prefix of @p y, so a binary search over @p y with a nested
		 *        tangent search finds @p s. Collinear bridge vertices resolve as
		 *        the stack does: the earliest of @p x, the latest of @p y.
		 * @pre `x.size(), y.size() >= 1`; @p x entirely before @p y.
		 * @complexity O(log |x| log |y|)
		 */
		template<point_iterator ItX, point_iterator ItY,
				 sign_policy<iterator_coordinate_t<ItX>> Policy>
			requires std::same_as<iterator_coordinate_t<ItX>, iterator_coordinate_t<ItY>>
		[[nodiscard]] constexpr std::pair<std::size_t, std::size_t>
		chain_bridge(hull_chain<ItX> const x, hull_chain<ItY> const y,
					 Policy const p) {
			atl_assert(x.size() >= 1 && y.size() >= 1);
			std::size_t low = 0, high = y.size() - 1;
			while (low < high) {
				std::size_t const mid = low + (high - low) / 2;
				if (p.orientation(x[chain_tangent_from_after(x, y[mid], p)], y[mid], y[mid + 1]) <= 0)
					low = mid + 1;
				else
					high = mid;
			}
			return {chain_tangent_from_after(x, y[low], p), low};
		}

		/**
		 * @brief The strict hull of two strict hulls in @ref
		 *        atl::geometry::convex_hull form, every vertex of @p x
		 *        lexicographically before every vertex of @p y: the lower
		 *        bridge joins the lower chains, the upper bridge the upper
		 *        chains (swept from @p y's maximum back to @p x's minimum), and
		 *        the four kept runs are bulk-copied.
		 * @pre `nx, ny >= 1`; @p top_x / @p top_y the positions of each hull's
		 *      lexicographic maximum.
		 * @complexity O(log h_x log h_y) searches plus the O(h) output copy
		 */
		template<point_iterator ItX, point_iterator ItY,
				 sign_policy<iterator_coordinate_t<ItX>> Policy>
			requires std::same_as<iterator_coordinate_t<ItX>, iterator_coordinate_t<ItY>>
		[[nodiscard]] constexpr std::vector<point<iterator_coordinate_t<ItX>>>
		merge_separated_hulls(ItX const first_x, std::size_t const nx, std::size_t const top_x, ItY const first_y,
							  std::size_t const ny, std::size_t const top_y, Policy const p) {
			atl_assert(nx >= 1 && ny >= 1 && top_x < nx && top_y < ny);
			atl_assert(std::iter_value_t<ItX>(first_x[top_x]) < std::iter_value_t<ItY>(first_y[0]));
			auto const [lower_end, lower_begin] = chain_bridge(hull_chain<ItX>{first_x, 0, top_x + 1, false},
															   hull_chain<ItY>{first_y, 0, top_y + 1, false}, p);
			auto const [upper_end, upper_begin] = chain_bridge(hull_chain<ItY>{first_y, top_y, ny, ny > 1},
															   hull_chain<ItX>{first_x, top_x, nx, nx > 1}, p);
			// upper_end is a position on y's upper chain, whose last entry
			// (position ny) is y's vertex 0 again
			std::size_t const upper_y_last = top_y + upper_end;
			std::size_t const upper_x_first = nx > 1 ? top_x + upper_begin : nx;
			std::vector<point<iterator_coordinate_t<ItX>>> out;
			out.reserve(lower_end + 1 + (top_y + 1 - lower_begin) + upper_end + (nx - upper_x_first));
			out.insert(out.end(), first_x, first_x + (lower_end + 1));
			out.insert(out.end(), first_y + lower_begin, first_y + (top_y + 1));
			out.insert(out.end(), first_y + (top_y + 1), first_y + ((std::ranges::min)(upper_y_last, ny - 1) + 1));
			if (upper_y_last == ny) out.push_back(first_y[0]);
			out.insert(out.end(), first_x + upper_x_first, first_x + nx);
			return out;
		}
	}

	/**
	 * @brief The strict convex hull of the union of two strict convex hulls,
	 *        in the same form, without sorting: each hull splits at its
	 *        lexicographic maximum into an increasing lower and a decreasing
	 *        upper chain, the lower (upper) hull of the union is the lower
	 *        (upper) hull of the two merged lower (upper) chains, and one
	 *        monotone-chain pass over each merged stream builds it. Exact
	 *        integral hulls with at least 1024 vertices in total and strictly
	 *        separated lexicographic extents use binary-searched bridges and bulk
	 *        copies instead (about 4 log h_a log h_b turns against h_a + h_b,
	 *        so tiny hulls keep the stream; the dispatch-size race covers totals 256 through 300000).
	 * @pre @p a and @p b each in the form @ref atl::geometry::convex_hull returns: strict
	 *      (distinct vertices, no three consecutive collinear), CCW, starting
	 *      at the lexicographically smallest vertex; either may be empty, a
	 *      single point or a 2-point segment. Checked in O(h) (asserted), as
	 *      is `|x|, |y| < 2^62` for `int64` coordinates. Under a non-exact
	 *      policy the turns at the first and the lexicographically largest
	 *      vertex are not checked: @ref atl::geometry::convex_hull joins its two chains there
	 *      without testing them, and its output under the same policy is
	 *      accepted. Under the exact policy every turn is checked and the
	 *      merged hull is again in @ref atl::geometry::convex_hull form (the round trip).
	 * @param a First finished hull.
	 * @param b Second finished hull.
	 * @param p Predicate policy.
	 * @return Strict hull of the union in convex_hull form.
	 * @complexity O(h_a + h_b); separated fast path O(log h_a log h_b + h_out),
	 *             excluding the O(h_a + h_b) assertion checks
	 */
	template<vertex_range RA, vertex_range RB, sign_policy<vertex_coordinate_t<RA>> Policy = exact>
		requires std::same_as<vertex_coordinate_t<RA>, vertex_coordinate_t<RB>>
	[[nodiscard]] constexpr std::vector<point<vertex_coordinate_t<RA>>> merge_convex_hulls(RA &&a, RB &&b, Policy const p = {}) {
		using T = vertex_coordinate_t<RA>;
		atl_assert(detail::is_lexicographic_strict_hull(a, p));
		atl_assert(detail::is_lexicographic_strict_hull(b, p));
		std::size_t const na = std::ranges::size(a), nb = std::ranges::size(b);
		auto const first_a = std::ranges::begin(a);
		auto const first_b = std::ranges::begin(b);
		if (na == 0 || nb == 0) return na == 0 ? std::vector<point<T>>(first_b, first_b + nb)
											   : std::vector<point<T>>(first_a, first_a + na);
		std::size_t const top_a = detail::lexicographic_top(first_a, na);
		std::size_t const top_b = detail::lexicographic_top(first_b, nb);
		if constexpr (std::integral<T> && std::same_as<Policy, exact>) {
			if (na >= 1024 || nb >= 1024 || na + nb >= 1024) {
				if (point<T>(first_a[top_a]) < point<T>(first_b[0])) return detail::merge_separated_hulls(first_a, na, top_a, first_b, nb, top_b, p);
				if (point<T>(first_b[top_b]) < point<T>(first_a[0])) return detail::merge_separated_hulls(first_b, nb, top_b, first_a, na, top_a, p);
			}
		}
		std::vector<point<T>> out;
		if (na > out.max_size() || nb > out.max_size() - na) throw std::length_error("convex hull is too large");
		out.reserve(na + nb);
		detail::merge_hull_chains<true>(detail::hull_chain{first_a, 0, top_a + 1, false},
										detail::hull_chain{first_b, 0, top_b + 1, false}, 2, out, p);
		std::size_t const lower = out.size();
		if (lower == 1) return out;
		// the upper pass restarts at the union's maximum, rewriting it in
		// place, and never pops into the lower chain
		out.pop_back();
		detail::merge_hull_chains<false>(detail::hull_chain{first_a, top_a, na, na > 1},
										 detail::hull_chain{first_b, top_b, nb, nb > 1}, lower + 1, out, p);
		out.pop_back();
		return out;
	}

	/**
	 * @brief Minkowski sum of two convex polygons: the classic angular
	 *        merge of the two edge sequences, emitted as a STRICT convex
	 *        counterclockwise polygon — no duplicate vertices and no three
	 *        collinear, because parallel edges (within or across the
	 *        inputs) are summed into one step. Anti-parallel steps can
	 *        never become adjacent for nonzero-area inputs (edges summing
	 *        to zero cannot all sit in a closed half-plane), so strictness
	 *        holds without a cleanup pass.
	 *
	 *        Inputs may be in either winding and may carry duplicate or
	 *        collinear vertices (each is passed through
	 *        @ref atl::geometry::normalized_polygon first); points and segments are legal
	 *        summands, and degenerate results come back with 1 or 2
	 *        vertices. The output starts at its (y, x)-minimal vertex.
	 * @note For floating coordinates, edge differences, sums and accumulated
	 * vertices are rounded in T; strictness and exact geometric topology are
	 * guaranteed only for representable integral arithmetic. Near floating ties
	 * this is a construction with no exact geometric guarantee.
	 * @pre Each input a nonempty point/segment/convex polygon (nonzero
	 *      area once 3 distinct vertices remain — normalization asserts);
	 *      coordinate sums, edge differences and accumulated steps must fit T;
	 *      for `int64` coordinates: `|x|, |y| < 2^61` (sums double the
	 *      magnitudes; asserted).
	 * @param a_points First convex summand.
	 * @param b_points Second convex summand.
	 * @param p Predicate policy.
	 * @return Strict CCW sum boundary from its (y, x)-minimum.
	 * @complexity O(n + m) time and memory.
	 */
	template<vertex_range RA, vertex_range RB,
			 sign_policy<vertex_coordinate_t<RA>> Policy = exact>
		requires std::same_as<vertex_coordinate_t<RA>, vertex_coordinate_t<RB>>
	[[nodiscard]] constexpr std::vector<point<vertex_coordinate_t<RA>>>
	minkowski_sum(RA &&a_points, RB &&b_points, Policy const p = {}) {
		using T = vertex_coordinate_t<RA>;
		auto a = normalized_polygon(a_points, p);
		auto b = normalized_polygon(b_points, p);
		atl_assert(!a.empty() && !b.empty());
		if constexpr (std::signed_integral<T> && sizeof(T) == 8) {
			[[maybe_unused]] auto const fits = [](point<T> const q) {
				constexpr T bound = T(1) << 61;
				return -bound < q.x && q.x < bound && -bound < q.y && q.y < bound;
			};
			atl_assert(std::ranges::all_of(a, fits) && std::ranges::all_of(b, fits));
		}
		// start both at the (y, x)-minimal vertex: the CCW edge sequences
		// are then sorted in the same [0, 2pi) sweep order the merge uses
		auto const rotate_to_low = [](std::vector<point<T>> &v) {
			auto const it = std::ranges::min_element(v, [](point<T> const l, point<T> const r) {
				return l.y != r.y ? l.y < r.y : l.x < r.x;
			});
			std::ranges::rotate(v, it);
		};
		rotate_to_low(a);
		rotate_to_low(b);
		auto const edge = [](std::vector<point<T>> const &v, std::size_t const k) {
			return v[k + 1 == v.size() ? 0 : k + 1] - v[k];
		};
		std::size_t const a_edges = a.size() < 2 ? 0 : a.size(), b_edges = b.size() < 2 ? 0 : b.size();
		std::vector<point<T>> out;
		out.reserve(a_edges + b_edges + 1);
		out.push_back(a[0] + b[0]);
		point<T> const origin{};
		point<T> at = out[0];
		std::size_t i = 0, j = 0;
		while (i < a_edges || j < b_edges) {
			point<T> step{};
			if (j == b_edges)
				step = edge(a, i++);
			else if (i == a_edges)
				step = edge(b, j++);
			else { // both live: one three-way angular compare; parallel edges
				   // advance BOTH, so one merged step keeps the output strict
				point<T> const ea = edge(a, i), eb = edge(b, j);
				int const ga = detail::sweep_group(origin, ea), gb = detail::sweep_group(origin, eb);
				int const order = ga != gb ? (ga < gb ? 1 : -1) : p.cross_sign(origin, ea, origin, eb);
				if (order >= 0) step = step + ea, ++i;
				if (order <= 0) step = step + eb, ++j;
			}
			if (i < a_edges || j < b_edges) { // the final step closes the cycle
				at = at + step;
				out.push_back(at);
			}
		}
		return out;
	}
}
