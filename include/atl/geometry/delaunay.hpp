#pragma once
#include <atl/ds/dsu.hpp>
#include <atl/geometry/circle.hpp>
#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/convex_polygon_operations.hpp>
#include <atl/geometry/detail/compare_products.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <ranges>
#include <utility>
#include <vector>

namespace atl::geometry {
	/**
	 * @defgroup delaunay_geometry Delaunay and Voronoi geometry
	 * @brief Exact Delaunay topology; rounded Voronoi, distances and circles.
	 * @pre `size(points) < 2^32`; `int32` uses its full range, `int64` needs
	 *      `|c| < 2^61`. Differences below `2^62` give quadratic factors below
	 *      `2^125` and an incircle determinant below `2^252`, fitting int256.
	 * @pre Allocated directed edges, including retained deleted edges, number
	 *      fewer than `2^32`; triangles number at most `2^32 - 2` (asserted
	 *      before narrowing). These storage limits apply to the whole family.
	 * @pre Floating expansion products must be finite and normal unless zero;
	 *      round-to-nearest with no unsafe reassociation is required.
	 * @complexity Delaunay construction O(n log n) time and allocated storage
	 *      (deleted edge ids are retained), O(log n) recursion depth. Large
	 *      arrays use heap or thread-local storage; no recursion per element.
	 * @{
	 */
	namespace detail {
		/**
		 * @brief Stage-A static filter bound for the adaptive incircle filter, in
		 *        units of the determinant's permanent.
		 */
		inline constexpr float64 incircle_bound_a = (10.0 + 96.0 * epsilon) * epsilon;
		/**
		 * @brief Stage-B bound for the adaptive incircle filter (the exact
		 *        determinant of the rounded differences), in units of the
		 *        permanent.
		 */
		inline constexpr float64 incircle_bound_b = (4.0 + 48.0 * epsilon) * epsilon;

		/** @brief Sign of a non-overlapping increasing-magnitude expansion. */
		constexpr int expansion_sign(int const len, float64 const *h) {
			float64 const top = h[len - 1];
			return top > 0 ? 1 : top < 0 ? -1
										 : 0;
		}

		/**
		 * @brief EXACT incircle sign for `double` points, `pd` the query —
		 *        the staged evaluation described at the incircle sign evaluation.
		 * @pre Finite coordinates with no overflow/underflow in the products
		 *      (the expansion-arithmetic precondition).
		 */
		inline int adaptive_incircle(point<float64> const pa, point<float64> const pb,
									 point<float64> const pc, point<float64> const pd) {
			check_floating_magnitude(pa);
			check_floating_magnitude(pb);
			check_floating_magnitude(pc);
			check_floating_magnitude(pd);
			float64 const adx = rounded_binary64(pa.x - pd.x), ady = rounded_binary64(pa.y - pd.y);
			float64 const bdx = rounded_binary64(pb.x - pd.x), bdy = rounded_binary64(pb.y - pd.y);
			float64 const cdx = rounded_binary64(pc.x - pd.x), cdy = rounded_binary64(pc.y - pd.y);
			float64 const alift = adx * adx + ady * ady;
			float64 const blift = bdx * bdx + bdy * bdy;
			float64 const clift = cdx * cdx + cdy * cdy;
			float64 const bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
			float64 const cdxady = cdx * ady, adxcdy = adx * cdy;
			float64 const adxbdy = adx * bdy, bdxady = bdx * ady;
			float64 const det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy)
							  + clift * (adxbdy - bdxady);
			float64 const permanent = (std::abs(bdxcdy) + std::abs(cdxbdy)) * alift
									+ (std::abs(cdxady) + std::abs(adxcdy)) * blift
									+ (std::abs(adxbdy) + std::abs(bdxady)) * clift;
			atl_assert(is_finite(permanent));
			atl_assert(normal_product(alift, std::abs(bdxcdy) + std::abs(cdxbdy)));
			atl_assert(normal_product(blift, std::abs(cdxady) + std::abs(adxcdy)));
			atl_assert(normal_product(clift, std::abs(adxbdy) + std::abs(bdxady)));
			float64 bound = incircle_bound_a * permanent;
			if (det > bound) return 1;
			if (-det > bound) return -1;

			// stage B: the exact determinant of the ROUNDED differences
			auto const cross_expansion = [](float64 const x1, float64 const y2,
											float64 const x2, float64 const y1,
											float64 *out4) {
				float64 l0 = 0, r0 = 0;
				float64 const l1 = two_product(x1, y2, l0);
				float64 const r1 = two_product(x2, y1, r0);
				two_two_diff(l1, l0, r1, r0, out4[3], out4[2], out4[1], out4[0]);
			};
			auto const lift_term = [](float64 const *cross4, float64 const dx,
									  float64 const dy, float64 *out32) {
				float64 sx[8]{}, sxx[16]{}, sy[8]{}, syy[16]{};
				int const sxl = scale_expansion(4, cross4, dx, sx);
				int const sxxl = scale_expansion(sxl, sx, dx, sxx);
				int const syl = scale_expansion(4, cross4, dy, sy);
				int const syyl = scale_expansion(syl, sy, dy, syy);
				return expansion_sum(sxxl, sxx, syyl, syy, out32);
			};
			float64 bc4[4]{}, ca4[4]{}, ab4[4]{};
			cross_expansion(bdx, cdy, cdx, bdy, bc4);
			cross_expansion(cdx, ady, adx, cdy, ca4);
			cross_expansion(adx, bdy, bdx, ady, ab4);
			float64 adet[32]{}, bdet[32]{}, cdet[32]{}, abdet[64]{}, fin[96]{};
			int const alen = lift_term(bc4, adx, ady, adet);
			int const blen = lift_term(ca4, bdx, bdy, bdet);
			int const clen = lift_term(ab4, cdx, cdy, cdet);
			int const ablen = expansion_sum(alen, adet, blen, bdet, abdet);
			int const finlen = expansion_sum(ablen, abdet, clen, cdet, fin);
			float64 estimate = 0;
			for (int k = 0; k < finlen; ++k) estimate += fin[k];
			// non-strict on purpose, so estimate 0 under bound 0 (a repeated
			// point: every product vanishes, the determinant is exactly zero)
			// certifies sign 0 instead of falling through
			bound = incircle_bound_b * permanent;
			if (estimate >= bound || -estimate >= bound)
				return estimate > 0 ? 1 : estimate < 0 ? -1
													   : 0;

			float64 const adxt = two_diff_tail(pa.x, pd.x, adx);
			float64 const adyt = two_diff_tail(pa.y, pd.y, ady);
			float64 const bdxt = two_diff_tail(pb.x, pd.x, bdx);
			float64 const bdyt = two_diff_tail(pb.y, pd.y, bdy);
			float64 const cdxt = two_diff_tail(pc.x, pd.x, cdx);
			float64 const cdyt = two_diff_tail(pc.y, pd.y, cdy);
			if (adxt == 0 && adyt == 0 && bdxt == 0 && bdyt == 0 && cdxt == 0 && cdyt == 0)
				return expansion_sign(finlen, fin); // differences exact: done

			// fully exact: generic expansion arithmetic over the exact
			// two-component differences (tail first: increasing magnitude)
			float64 const ax[2]{adxt, adx}, ay[2]{adyt, ady};
			float64 const bx[2]{bdxt, bdx}, by[2]{bdyt, bdy};
			float64 const cx[2]{cdxt, cdx}, cy[2]{cdyt, cdy};
			float64 scratch[1600]{}, scaled[64]{};
			auto const mul22 = [&](float64 const *e, float64 const *f, float64 *out8) {
				return expansion_product(2, e, 2, f, out8, scratch, scaled);
			};
			auto const cross_exact = [&](float64 const *x1, float64 const *y2,
										 float64 const *x2, float64 const *y1,
										 float64 *out16) {
				float64 left[8]{}, right[8]{};
				int const ll = mul22(x1, y2, left);
				int const rl = mul22(x2, y1, right);
				for (int k = 0; k < rl; ++k) right[k] = -right[k];
				return expansion_sum(ll, left, rl, right, out16);
			};
			auto const lift_exact = [&](float64 const *dx, float64 const *dy,
										float64 *out16) {
				float64 xx[8]{}, yy[8]{};
				int const xl = mul22(dx, dx, xx);
				int const yl = mul22(dy, dy, yy);
				return expansion_sum(xl, xx, yl, yy, out16);
			};
			float64 bce[16]{}, cae[16]{}, abe[16]{}, alifte[16]{}, blifte[16]{}, clifte[16]{};
			int const bcel = cross_exact(bx, cy, cx, by, bce);
			int const cael = cross_exact(cx, ay, ax, cy, cae);
			int const abel = cross_exact(ax, by, bx, ay, abe);
			int const aliftl = lift_exact(ax, ay, alifte);
			int const bliftl = lift_exact(bx, by, blifte);
			int const cliftl = lift_exact(cx, cy, clifte);
			static thread_local float64 term_a[600], term_b[600], term_c[600],
				sum_ab[1200], total[1800];
			int const tal = expansion_product(aliftl, alifte, bcel, bce, term_a, scratch,
											  scaled);
			int const tbl = expansion_product(bliftl, blifte, cael, cae, term_b, scratch,
											  scaled);
			int const tcl = expansion_product(cliftl, clifte, abel, abe, term_c, scratch,
											  scaled);
			int const sl = expansion_sum(tal, term_a, tbl, term_b, sum_ab);
			int const tl = expansion_sum(sl, sum_ab, tcl, term_c, total);
			return expansion_sign(tl, total);
		}

		/**
		 * @brief The incircle sign for integral coordinates in int256, exact
		 *        within `|x|, |y| < 2^61` (three products of int128-fitting
		 *        factors): the fallback of the incircle sign evaluation at or above 2^32 and
		 *        in constant evaluation.
		 */
		template<coordinate T>
			requires std::signed_integral<T>
		constexpr int incircle_sign_int256(point<T> const a, point<T> const b, point<T> const c,
										   point<T> const q) {
			if constexpr (sizeof(T) == 8) {
				[[maybe_unused]] auto const is_bounded = [](point<T> const v) {
					constexpr T bound = T(1) << 61;
					return -bound < v.x && v.x < bound && -bound < v.y && v.y < bound;
				};
				atl_assert(is_bounded(a) && is_bounded(b) && is_bounded(c) && is_bounded(q));
			}
			using W = int128;
			W const adx = W(a.x) - W(q.x), ady = W(a.y) - W(q.y);
			W const bdx = W(b.x) - W(q.x), bdy = W(b.y) - W(q.y);
			W const cdx = W(c.x) - W(q.x), cdy = W(c.y) - W(q.y);
			W const ad2 = adx * adx + ady * ady;
			W const bd2 = bdx * bdx + bdy * bdy;
			W const cd2 = cdx * cdx + cdy * cdy;
			return sum_three_wide_products_sign(bdx * cdy - cdx * bdy, ad2,
												cdx * ady - adx * cdy, bd2, adx * bdy - bdx * ady, cd2);
		}

		/**
		 * @brief Sign of the incircle determinant: positive when @p q lies
		 *        strictly INSIDE the circumcircle of the CCW triangle
		 *        (a, b, c), negative outside, zero on it. Integral
		 *        coordinates: EXACT within `|x|, |y| < 2^61` — below 2^32 through the adaptive double
		 *        evaluation below, which is exact on exactly converted
		 *        integers and faster than int256 (x86-64 Xeon 8280 -O2, record
		 *        2026-10-15: 26.2 vs 64.6 ms per 1e6 random int64 calls),
		 *        above it and in constant evaluation via int256. Floating
		 *        coordinates: EXACT too — Shewchuk's staged adaptivity: the
		 *        A filter certifies generic inputs from one evaluation, the
		 *        B stage computes the exact determinant of the ROUNDED
		 *        differences by expansion arithmetic (which IS the exact
		 *        answer whenever the differences are exact — integer coordinates
		 *        of magnitude below 2^52 included), and the rare remainder
		 *        falls through to a fully exact GENERIC expansion evaluation
		 *        over the two-component difference expansions (in place of
		 *        Shewchuk's hand-unrolled C/D stages: same answer, simpler
		 *        code, only slower in the near-degenerate sliver of cases).
		 */
		template<coordinate T>
		constexpr int incircle_sign(point<T> const a, point<T> const b, point<T> const c,
									point<T> const q) {
			if constexpr (std::floating_point<T>) {
				return adaptive_incircle(to_float64(a),
										 to_float64(b),
										 to_float64(c),
										 to_float64(q));
			} else {
				if !consteval {
					// Preserve the adaptive fast path for coordinates below 2^32.
					if constexpr (sizeof(T) < 8) {
						return adaptive_incircle(to_float64(a), to_float64(b), to_float64(c), to_float64(q));
					} else {
						constexpr int64 limit = int64(1) << 32;
						auto const fits = [](point<T> const v) {
							return -limit < v.x && v.x < limit && -limit < v.y && v.y < limit;
						};
						if (fits(a) && fits(b) && fits(c) && fits(q))
							return adaptive_incircle(to_float64(a), to_float64(b), to_float64(c), to_float64(q));
					}
				}
				return incircle_sign_int256(a, b, c, q);
			}
		}
	}

	/**
	 * @brief Delaunay triangulation of a point set: CCW index triples into
	 *        the input plus across-edge adjacency — `neighbors[t][e]` is the
	 *        triangle sharing edge `(triangles[t][e], triangles[t][(e+1)%3])`,
	 *        or `no_neighbor` on the hull boundary. Duplicate points collapse
	 *        to their smallest input index; fewer than 3 distinct points or
	 *        an all-collinear set yield no triangles. Cocircular groups get
	 *        one valid tie-break triangulation (the empty-circumcircle
	 *        property holds non-strictly: no point STRICTLY inside any
	 *        circumcircle).
	 */
	struct delaunay_result {
		static constexpr triangle_index no_neighbor = ~triangle_index(0); ///< Hull-edge sentinel.

		std::vector<std::array<vertex_index, 3>> triangles;	  ///< CCW input index triples.
		std::vector<std::array<triangle_index, 3>> neighbors; ///< Across-edge triangle ids or `no_neighbor`.
	};

	namespace detail {
		/**
		 * @brief The shared entry checks of the Delaunay family: fewer than
		 *        2^32 points, and for 64-bit integral coordinates
		 *        `|x|, |y| < 2^61` (both asserted). Returns whether int256 is needed.
		 */
		template<vertex_range R>
		constexpr bool check_delaunay_input([[maybe_unused]] R &&points) {
			using T = vertex_coordinate_t<R>;
			atl_assert(uint64(std::ranges::size(points)) < (uint64(1) << 32));
			bool uses_wide = false;
			if constexpr (std::signed_integral<T> && sizeof(T) == 8) {
				[[maybe_unused]] constexpr T bound = T(1) << 61;
				constexpr T fast_bound = T(1) << 32;
				for (point<T> const q : points) {
					atl_assert(-bound < q.x && q.x < bound && -bound < q.y && q.y < bound);
					uses_wide |= q.x <= -fast_bound || q.x >= fast_bound || q.y <= -fast_bound || q.y >= fast_bound;
				}
			}
			return uses_wide;
		}
	}

	namespace detail {
		/**
		 * @brief Guibas-Stolfi divide and conquer over a primal quad-edge store:
		 * directed edges with `origin` and the CCW-around-origin ring
		 * (`onext` plus its maintained inverse `oprev`); `sym` = id ^ 1.
		 */
		template<coordinate T>
		class delaunay_builder {
			using vid = uint32;

		public:
			/**
			 * @p points must be lexicographically sorted and deduplicated, and
			 * must outlive the builder (it keeps a reference).
			 */
			explicit delaunay_builder(std::vector<point<T>> const &points, bool const uses_wide) : pts_(points), uses_wide_(uses_wide) {}

			void run() {
				atl_assert(uint64(pts_.size()) < (uint64(1) << 32));
				if (pts_.size() >= 2) build(0, uint32(pts_.size()));
			}

			/**
			 * @brief The triangles (CCW, indices into the builder's points)
			 *        and their adjacency, read off the faces of the quad-edge
			 *        store; non-triangular and clockwise faces are the outside.
			 */
			delaunay_result result() const {
				delaunay_result out;
				atl_assert(uint64(onext_.size()) < (uint64(1) << 32));
				uint32 const m = uint32(onext_.size());
				if (m == 0) return out;
				constexpr uint32 outer = ~uint32(0) - 1, unseen = ~uint32(0);
				std::vector<uint32> face(m, unseen), slot(m, 3);
				exact const p{};
				for (uint32 e = 0; e < m; ++e) {
					if (dead_[e >> 1] || face[e] != unseen) continue;

					std::array<uint32, 3> ring{};
					uint32 count = 0;
					uint32 walk = e;
					do {
						if (count < 3) ring[count] = walk;
						++count;
						walk = lnext(walk);
					} while (walk != e && count <= 3);
					bool is_valid_triangle = count == 3;
					if (is_valid_triangle) {
						point<T> const a = pts_[origin_[ring[0]]];
						point<T> const b = pts_[origin_[ring[1]]];
						point<T> const c = pts_[origin_[ring[2]]];
						is_valid_triangle = p.orientation(a, b, c) > 0;
					}
					if (!is_valid_triangle) {
						uint32 w2 = e;
						do {
							face[w2] = outer;
							w2 = lnext(w2);
						} while (w2 != e);
						continue;
					}
					atl_assert(out.triangles.size() < std::size_t(delaunay_result::no_neighbor) - 1);
					uint32 const t = uint32(out.triangles.size());
					out.triangles.push_back(
						{origin_[ring[0]], origin_[ring[1]], origin_[ring[2]]});
					for (uint32 k = 0; k < 3; ++k) {
						face[ring[k]] = t;
						slot[ring[k]] = k;
					}
				}
				out.neighbors.assign(out.triangles.size(),
									 {delaunay_result::no_neighbor,
									  delaunay_result::no_neighbor,
									  delaunay_result::no_neighbor});
				for (uint32 e = 0; e < m; ++e) {
					if (dead_[e >> 1] || face[e] >= outer) continue;
					if (face[e ^ 1] < outer)
						out.neighbors[face[e]][slot[e]] = face[e ^ 1];
				}
				return out;
			}

			/**
			 * @brief Every alive undirected edge (Delaunay edges; for collinear
			 * input, the sorted path).
			 */
			std::vector<std::array<vertex_index, 2>> edge_list() const {
				std::vector<std::array<vertex_index, 2>> out;
				atl_assert(uint64(onext_.size()) < (uint64(1) << 32));
				for (uint32 e = 0; e < uint32(onext_.size()); e += 2)
					if (!dead_[e >> 1]) out.push_back({origin_[e], origin_[e + 1]});
				return out;
			}

		private:
			uint32 sym(uint32 const e) const { return e ^ 1; }

			uint32 lnext(uint32 const e) const { return oprev_[sym(e)]; }

			uint32 rprev(uint32 const e) const { return onext_[sym(e)]; }

			vid dest(uint32 const e) const { return origin_[sym(e)]; }

			uint32 make_edge(vid const from, vid const to) {
				atl_assert(uint64(onext_.size()) + 2 < (uint64(1) << 32));
				uint32 const e = uint32(onext_.size());
				onext_.push_back(e);
				onext_.push_back(e + 1);
				oprev_.push_back(e);
				oprev_.push_back(e + 1);
				origin_.push_back(from);
				origin_.push_back(to);
				dead_.push_back(0);
				return e;
			}

			/** @brief Guibas-Stolfi splice: swaps the origin rings of @p a and @p b (joins or splits them). */
			void splice(uint32 const a, uint32 const b) {
				uint32 const an = onext_[a], bn = onext_[b];
				onext_[a] = bn;
				onext_[b] = an;
				oprev_[bn] = a;
				oprev_[an] = b;
			}

			/** @brief A new edge from the destination of @p a to the origin of @p b, spliced into both rings. */
			uint32 connect(uint32 const a, uint32 const b) {
				uint32 const e = make_edge(dest(a), origin_[b]);
				splice(e, lnext(a));
				splice(sym(e), b);
				return e;
			}

			/** @brief Unlinks edge @p e from both rings and marks the pair dead (ids are not reused). */
			void delete_edge(uint32 const e) {
				splice(e, oprev_[e]);
				splice(sym(e), oprev_[sym(e)]);
				dead_[e >> 1] = 1;
			}

			/** @brief Exact orientation of the three indexed points. */
			int ccw(vid const a, vid const b, vid const c) const {
				return exact{}.orientation(pts_[a], pts_[b], pts_[c]);
			}

			bool right_of(vid const q, uint32 const e) const {
				return ccw(q, dest(e), origin_[e]) > 0;
			}

			bool left_of(vid const q, uint32 const e) const {
				return ccw(q, origin_[e], dest(e)) > 0;
			}

			bool in_circle(vid const a, vid const b, vid const c, vid const d) const {
				if constexpr (std::signed_integral<T>) {
					if (uses_wide_) return incircle_sign_int256(pts_[a], pts_[b], pts_[c], pts_[d]) > 0;
					return adaptive_incircle(to_float64(pts_[a]), to_float64(pts_[b]), to_float64(pts_[c]), to_float64(pts_[d])) > 0;
				} else {
					return incircle_sign(pts_[a], pts_[b], pts_[c], pts_[d]) > 0;
				}
			}

			/**
			 * @brief Triangulate pts_[lo, hi); returns (ccw hull edge out of the
			 * leftmost point, cw hull edge out of the rightmost point).
			 */
			std::pair<uint32, uint32> build(uint32 const lo, uint32 const hi) {
				uint32 const n = hi - lo;
				if (n == 2) {
					uint32 const e = make_edge(lo, lo + 1);
					return {e, sym(e)};
				}
				if (n == 3) {
					uint32 const a = make_edge(lo, lo + 1);
					uint32 const b = make_edge(lo + 1, lo + 2);
					splice(sym(a), b);
					int const o = ccw(lo, lo + 1, lo + 2);
					if (o > 0) {
						connect(b, a);
						return {a, sym(b)};
					}
					if (o < 0) {
						uint32 const c = connect(b, a);
						return {sym(c), c};
					}
					return {a, sym(b)}; // collinear: a path
				}
				uint32 const mid = lo + n / 2;
				auto [ldo, ldi] = build(lo, mid);
				auto [rdi, rdo] = build(mid, hi);

				for (;;) {
					if (left_of(origin_[rdi], ldi))
						ldi = lnext(ldi);
					else if (right_of(origin_[ldi], rdi))
						rdi = rprev(rdi);
					else
						break;
				}
				uint32 basel = connect(sym(rdi), ldi);
				if (origin_[ldi] == origin_[ldo]) ldo = sym(basel);
				if (origin_[rdi] == origin_[rdo]) rdo = basel;

				for (;;) {
					uint32 lcand = onext_[sym(basel)];
					if (right_of(dest(lcand), basel))
						while (in_circle(dest(basel), origin_[basel], dest(lcand),
										 dest(onext_[lcand]))) {
							uint32 const t = onext_[lcand];
							delete_edge(lcand);
							lcand = t;
						}
					uint32 rcand = oprev_[basel];
					if (right_of(dest(rcand), basel))
						while (in_circle(dest(basel), origin_[basel], dest(rcand),
										 dest(oprev_[rcand]))) {
							uint32 const t = oprev_[rcand];
							delete_edge(rcand);
							rcand = t;
						}
					bool const is_left_valid = right_of(dest(lcand), basel);
					bool const is_right_valid = right_of(dest(rcand), basel);
					if (!is_left_valid && !is_right_valid) break;
					if (!is_left_valid
						|| (is_right_valid
							&& in_circle(dest(lcand), origin_[lcand], origin_[rcand],
										 dest(rcand))))
						basel = connect(rcand, sym(basel));
					else
						basel = connect(sym(basel), sym(lcand));
				}
				return {ldo, rdo};
			}

			std::vector<point<T>> const &pts_; ///< the sorted, distinct points
			bool uses_wide_;
			std::vector<uint32> onext_;	 ///< Per directed edge: CCW ring successor.
			std::vector<uint32> oprev_;	 ///< Per directed edge: inverse ring successor.
			std::vector<uint32> origin_; ///< Per directed edge: origin vertex.
			std::vector<char> dead_;	 ///< per edge pair: deleted
		};
	}

	/**
	 * @brief Delaunay triangulation of @p points (any sized random-access
	 *        range of `point<T>`): Guibas-Stolfi divide and conquer with
	 *        EXACT predicates for both coordinate kinds: integral
	 *        coordinates use `orientation` + an int256 `incircle`, floating
	 *        ones the adaptive orientation and the staged adaptive incircle.
	 *        Indices in the result refer to the ORIGINAL input.
	 * @pre Integral `int32` has no magnitude bound; `int64` needs `|x|, |y| < 2^61`
	 *      (asserted). Floating products follow the group contract.
	 * @return CCW triangles and matching neighbors; order and cocircular diagonals unspecified.
	 * @complexity O(n log n)
	 *
	 * @note Not constexpr (builds a runtime quad-edge store or uses runtime math).
	 */
	template<vertex_range R>
	[[nodiscard]] delaunay_result delaunay(R &&points) {
		using T = vertex_coordinate_t<R>;
		bool const uses_wide = detail::check_delaunay_input(points);
		uint32 const n = uint32(std::ranges::size(points));
		auto const first = std::ranges::begin(points);

		struct entry {
			point<T> position;
			uint32 index;
		};

		std::vector<entry> order;
		order.reserve(n);
		for (uint32 i = 0; i < n; ++i) order.push_back({first[i], i});
		std::ranges::sort(order, [](entry const &l, entry const &r) {
			return l.position != r.position ? l.position < r.position : l.index < r.index;
		});
		auto const [d0, d1] = std::ranges::unique(
			order, [](entry const &l, entry const &r) { return l.position == r.position; });
		order.erase(d0, d1);
		std::vector<point<T>> distinct(order.size());
		for (std::size_t k = 0; k < order.size(); ++k) distinct[k] = order[k].position;
		detail::delaunay_builder<T> builder(distinct, uses_wide);
		builder.run();
		auto out = builder.result();
		for (auto &tri : out.triangles)
			for (auto &v : tri) v = order[v].index; // back to original indices
		return out;
	}

	/**
	 * @brief All edges of the Delaunay triangulation of @p points, as index
	 *        pairs into the ORIGINAL input (smallest index per duplicate
	 *        group). Unlike @ref atl::geometry::delaunay this is meaningful for DEGENERATE
	 *        inputs too: a collinear set yields its sorted adjacency path.
	 *        The foundation for @ref atl::geometry::euclidean_mst and
	 *        @ref atl::geometry::closest_pair_between.
	 * @pre As @ref atl::geometry::delaunay (asserted).
	 *
	 * @note Not constexpr (builds a runtime quad-edge store or uses runtime math).
	 */
	template<vertex_range R>
	[[nodiscard]] std::vector<std::array<vertex_index, 2>> delaunay_edges(R &&points) {
		using T = vertex_coordinate_t<R>;
		bool const uses_wide = detail::check_delaunay_input(points);
		uint32 const n = uint32(std::ranges::size(points));
		auto const first = std::ranges::begin(points);

		struct entry {
			point<T> position;
			uint32 index;
		};

		std::vector<entry> order;
		order.reserve(n);
		for (uint32 i = 0; i < n; ++i) order.push_back({first[i], i});
		std::ranges::sort(order, [](entry const &l, entry const &r) {
			return l.position != r.position ? l.position < r.position : l.index < r.index;
		});
		auto const [d0, d1] = std::ranges::unique(
			order, [](entry const &l, entry const &r) { return l.position == r.position; });
		order.erase(d0, d1);
		std::vector<point<T>> distinct(order.size());
		for (std::size_t k = 0; k < order.size(); ++k) distinct[k] = order[k].position;
		detail::delaunay_builder<T> builder(distinct, uses_wide);
		builder.run();
		auto edges = builder.edge_list();
		for (auto &e : edges)
			for (auto &v : e) v = order[v].index;
		return edges;
	}

	/**
	 * @brief One edge of a Voronoi diagram: the dual of the Delaunay edge
	 *        between `site_a` and `site_b`. Interior duals are segments
	 *        between two circumcenters; hull duals are RAYS from one
	 *        circumcenter along `direction` (outward hull normal,
	 *        unnormalized). Coordinates are `float64` constructions.
	 */
	struct voronoi_edge {
		vertex_index site_a;	  ///< First input site of the primal edge.
		vertex_index site_b;	  ///< Second input site of the primal edge.
		point<float64> from;	  ///< circumcenter of the first triangle
		point<float64> to;		  ///< second circumcenter (segments only)
		point<float64> direction; ///< outward ray direction (rays only)
		bool is_unbounded;		  ///< true = ray, false = segment
	};

	/**
	 * @brief The Voronoi diagram as the Delaunay dual: one vertex per
	 *        triangle (its circumcenter), one edge per Delaunay edge. The
	 *        cell of site s is bounded by the edges with `site_a == s` or
	 *        `site_b == s` (group by site to walk cells); hull sites have
	 *        unbounded cells closed by two rays.
	 */
	struct voronoi_diagram {
		std::vector<point<float64>> vertices; ///< circumcenter of triangle t
		std::vector<voronoi_edge> edges;	  ///< Dual segments and outward hull rays.
	};

	/** @brief The Voronoi diagram dual to @p d (which must come from @p points).
	 * @note Not constexpr (builds a runtime quad-edge store or uses runtime math).
	 */
	template<vertex_range R>
	[[nodiscard]] voronoi_diagram voronoi_from_delaunay(R &&points, delaunay_result const &d) {
		using T = vertex_coordinate_t<R>;
		detail::check_delaunay_input(points);
		atl_assert(d.triangles.size() < std::size_t(delaunay_result::no_neighbor));
		atl_assert(d.neighbors.size() == d.triangles.size());
		auto const first = std::ranges::begin(points);
		voronoi_diagram out;
		out.vertices.reserve(d.triangles.size());
		for (auto const &t : d.triangles)
			out.vertices.push_back(circumcircle(first[t[0]], first[t[1]], first[t[2]]).center);
		for (uint32 t = 0; t < uint32(d.triangles.size()); ++t)
			for (uint32 e = 0; e < 3; ++e) {
				vertex_index const u = d.triangles[t][e];
				vertex_index const v = d.triangles[t][(e + 1) % 3];
				triangle_index const o = d.neighbors[t][e];
				if (o == delaunay_result::no_neighbor) {
					// hull edge: ray outward (right of the CCW edge u -> v)
					// the edge vector is subtracted in the widened type before rounding, so far-away sites
					// that are close together keep their direction
					auto const edge = detail::float64_difference(point<T>(first[u]), point<T>(first[v]));
					out.edges.push_back({u, v, out.vertices[t], {}, {edge.y, -edge.x}, true});
				} else if (t < o) {
					out.edges.push_back(
						{u, v, out.vertices[t], out.vertices[o], {}, false});
				}
			}
		return out;
	}

	/**
	 * @brief Euclidean minimum spanning tree of @p points: `n - 1` index
	 *        edges spanning EVERY input index (duplicates attach to their
	 *        representative with zero-length edges). Kruskal over the
	 *        Delaunay edges (which always contain an EMST) with integral-exact
	 *        squared-length comparisons (floating keys are rounded once); collinear sets reduce to the sorted
	 *        path.
	 * @pre `size(points) >= 1`; coordinate bounds as @ref atl::geometry::delaunay (both
	 *      asserted).
	 * @complexity O(n log n)
	 *
	 * @note Not constexpr (builds a runtime quad-edge store or uses runtime math).
	 */
	template<vertex_range R>
	[[nodiscard]] std::vector<std::array<vertex_index, 2>> euclidean_mst(R &&points) {
		using T = vertex_coordinate_t<R>;
		atl_assert(std::ranges::size(points) >= 1);
		detail::check_delaunay_input(points);
		uint32 const n = uint32(std::ranges::size(points));
		auto const first = std::ranges::begin(points);

		// each squared length computed once, then sorted as a key (a
		// comparator recomputing both measured 73.4 ms versus 25.9 ms on uniform 1e5)
		struct keyed_edge {
			difference_product_t<T> key;
			std::array<vertex_index, 2> edge;
		};

		std::vector<keyed_edge> keyed;
		{
			auto const edges = delaunay_edges(points);
			keyed.reserve(edges.size());
			for (auto const &e : edges) keyed.push_back({squared_distance(first[e[0]], first[e[1]]), e});
		}
		std::ranges::sort(keyed, {}, &keyed_edge::key);
		std::vector<std::array<vertex_index, 2>> out;
		out.reserve(n - 1);
		dsu components(n);
		for (auto const &[key, e] : keyed)
			if (components.unite(e[0], e[1])) out.push_back(e);
		// duplicates: not present in the deduplicated edge set — attach each
		// to its identical representative (zero-length edges)
		if (out.size() + 1 < n) {
			struct entry {
				point<T> position;
				uint32 index;
			};

			std::vector<entry> order;
			order.reserve(n);
			for (uint32 i = 0; i < n; ++i) order.push_back({first[i], i});
			std::ranges::sort(order, [](entry const &l, entry const &r) {
				return l.position != r.position ? l.position < r.position
												: l.index < r.index;
			});
			for (std::size_t k = 1; k < order.size(); ++k)
				if (order[k].position == order[k - 1].position
					&& components.unite(order[k].index, order[k - 1].index))
					out.push_back({order[k - 1].index, order[k].index});
		}
		atl_assert(out.size() + 1 == n);
		return out;
	}

	/**
	 * @brief Indices of a closest pair BETWEEN two point sets (one index
	 *        into each input) — the bichromatic closest pair: it is always
	 *        an edge of the Delaunay triangulation of the colored UNION.
	 *        Integral-exact squared-distance selection (floating distances are rounded); shared positions across the
	 *        sets give distance 0. Ties resolve arbitrarily.
	 * @pre Both inputs non-empty (asserted); coordinate bounds as
	 *      @ref atl::geometry::delaunay.
	 * @complexity O((n + m) log(n + m))
	 *
	 * @note Not constexpr (builds a runtime quad-edge store or uses runtime math).
	 */
	template<vertex_range RA, vertex_range RB>
		requires std::same_as<vertex_coordinate_t<RA>, vertex_coordinate_t<RB>>
	[[nodiscard]] std::array<vertex_index, 2> closest_pair_between_indices(RA &&a_points, RB &&b_points) {
		using T = vertex_coordinate_t<RA>;
		atl_assert(std::ranges::size(a_points) >= 1 && std::ranges::size(b_points) >= 1);
		atl_assert(uint64(std::ranges::size(a_points)) + uint64(std::ranges::size(b_points)) < (uint64(1) << 32));
		detail::check_delaunay_input(a_points);
		detail::check_delaunay_input(b_points);
		uint32 const n = uint32(std::ranges::size(a_points));
		uint32 const m = uint32(std::ranges::size(b_points));
		auto const a_first = std::ranges::begin(a_points);
		auto const b_first = std::ranges::begin(b_points);
		std::vector<point<T>> merged;
		merged.reserve(std::size_t(n) + m);
		for (uint32 i = 0; i < n; ++i) merged.push_back(a_first[i]);
		for (uint32 j = 0; j < m; ++j) merged.push_back(b_first[j]);
		// shared positions dedupe away inside delaunay_edges, so cross-set
		// duplicates must be caught FIRST (they are distance 0)
		{
			struct entry {
				point<T> position;
				uint32 index;
			};

			std::vector<entry> order;
			order.reserve(merged.size());
			for (uint32 i = 0; i < uint32(merged.size()); ++i)
				order.push_back({merged[i], i});
			std::ranges::sort(order, [](entry const &l, entry const &r) {
				return l.position != r.position ? l.position < r.position
												: l.index < r.index;
			});
			for (std::size_t k = 1; k < order.size(); ++k)
				if (order[k].position == order[k - 1].position
					&& order[k - 1].index < n && order[k].index >= n)
					return {order[k - 1].index, order[k].index - n};
		}
		auto const edges = delaunay_edges(merged);
		auto best = std::array<vertex_index, 2>{~vertex_index(0), ~vertex_index(0)};
		difference_product_t<T> best_gap{};
		bool is_found = false;
		for (auto const &e : edges) {
			bool const is_first_in_a = e[0] < n, is_second_in_a = e[1] < n;
			if (is_first_in_a == is_second_in_a) continue;
			auto const gap = squared_distance(merged[e[0]], merged[e[1]]);
			if (!is_found || gap < best_gap) {
				is_found = true;
				best_gap = gap;
				uint32 const ai = is_first_in_a ? e[0] : e[1];
				uint32 const bi = is_first_in_a ? e[1] : e[0];
				best = {ai, bi - n};
			}
		}
		atl_assert(is_found); // both sets non-empty: a cross edge always exists
		return best;
	}

	/** @brief @ref atl::geometry::closest_pair_between_indices materialized as points.
	 * @note Not constexpr (builds a runtime quad-edge store or uses runtime math).
	 */
	template<vertex_range RA, vertex_range RB>
		requires std::same_as<vertex_coordinate_t<RA>, vertex_coordinate_t<RB>>
	[[nodiscard]] std::array<point<vertex_coordinate_t<RA>>, 2> closest_pair_between(RA &&a_points,
																					 RB &&b_points) {
		auto const [i, j] = closest_pair_between_indices(a_points, b_points);
		return {std::ranges::begin(a_points)[i], std::ranges::begin(b_points)[j]};
	}

	/**
	 * @brief The largest circle whose center lies in the CONVEX HULL of
	 *        @p points and whose interior contains no point — center and
	 *        radius as `double` constructions. The optimum is a Voronoi
	 *        vertex inside the hull or a Voronoi-edge/hull-boundary
	 *        crossing; both candidate families are enumerated (each ray
	 *        is extended past the hull's bounding box before clipping).
	 *        Degenerate hulls: a segment optimizes over the largest gap
	 *        along it; a single position gives radius 0.
	 * @pre `size(points) >= 1`; coordinate bounds as @ref atl::geometry::delaunay (both
	 *      asserted).
	 * @complexity O(n log n + n * h) worst-case time, where h is the hull size;
	 *             O(n log n) allocated storage. Rounding can affect candidate clipping.
	 *
	 * @note Not constexpr (builds a runtime quad-edge store or uses runtime math).
	 */
	template<vertex_range R>
	[[nodiscard]] circle<float64> largest_empty_circle(R &&points) {
		using T = vertex_coordinate_t<R>;
		atl_assert(std::ranges::size(points) >= 1);
		detail::check_delaunay_input(points);
		auto const first = std::ranges::begin(points);
		auto const hull = convex_hull(points);
		if (hull.size() == 1)
			return {to_float64(hull[0]), 0.0};
		if (hull.size() == 2) { // collinear: the largest gap along the segment
			std::vector<point<T>> sorted;
			sorted.reserve(std::ranges::size(points));
			for (auto const q : points) sorted.push_back(q);
			std::ranges::sort(sorted);
			point<float64> best_center = to_float64(sorted[0]);
			float64 best_radius = 0;
			for (std::size_t k = 1; k < sorted.size(); ++k) {
				point<float64> const mid{
					(float64(sorted[k - 1].x) + float64(sorted[k].x)) / 2,
					(float64(sorted[k - 1].y) + float64(sorted[k].y)) / 2};
				float64 const r = distance(sorted[k - 1], sorted[k]) / 2;
				if (r > best_radius) {
					best_radius = r;
					best_center = mid;
				}
			}
			return {best_center, best_radius};
		}
		auto const d = delaunay(points);
		auto const vor = voronoi_from_delaunay(points, d);
		std::vector<point<float64>> hull_f;
		hull_f.reserve(hull.size());
		for (auto const q : hull) hull_f.push_back(to_float64(q));
		circle<float64> best{to_float64(hull[0]), 0.0};
		// every candidate's nearest site is known: a Voronoi vertex is
		// equidistant to its triangle's corners, a point on a dual edge to
		// the edge's two sites, and no site is closer (the empty circle)
		auto const consider = [&](point<float64> const q, point<T> const nearest) {
			float64 const r = distance(q, to_float64(nearest));
			if (r > best.radius) best = {q, r};
		};
		auto const inside_hull = [&](point<float64> const q) {
			return convex_contains(hull_f, q) != containment::outside;
		};
		for (uint32 t = 0; t < uint32(vor.vertices.size()); ++t)
			if (inside_hull(vor.vertices[t])) consider(vor.vertices[t], first[d.triangles[t][0]]);

		point<float64> box_low = hull_f[0], box_high = hull_f[0];
		for (auto const q : hull_f) {
			box_low = {(std::min)(box_low.x, q.x), (std::min)(box_low.y, q.y)};
			box_high = {(std::max)(box_high.x, q.x), (std::max)(box_high.y, q.y)};
		}
		for (auto const &e : vor.edges) {
			// a segment with both ends in the convex hull stays inside it:
			// its clip is its two ends, already candidates as vertices
			if (!e.is_unbounded && inside_hull(e.from) && inside_hull(e.to)) continue;
			point<float64> to = e.to;
			if (e.is_unbounded) {
				// a ray starts at a circumcenter, which a thin triangle puts
				// arbitrarily far outside the hull: extend it past the
				// farthest corner of the hull's bounding box AS SEEN FROM
				// ITS OWN ORIGIN, or it can stop short of the hull
				float64 const reach = std::hypot(
					(std::max)(std::abs(e.from.x - box_low.x), std::abs(e.from.x - box_high.x)),
					(std::max)(std::abs(e.from.y - box_low.y), std::abs(e.from.y - box_high.y)));
				float64 const len = std::hypot(e.direction.x, e.direction.y);
				float64 const scale = 2 * reach / (len > 0 ? len : 1);
				to = {e.from.x + e.direction.x * scale, e.from.y + e.direction.y * scale};
			}
			auto const clipped = detail::segment_clipped_to_hull(
				segment<float64>{e.from, to}, hull_f, exact{}, false);
			for (auto const q : clipped) consider(q, first[e.site_a]);
		}
		return best;
	}

	/** @} */
}
