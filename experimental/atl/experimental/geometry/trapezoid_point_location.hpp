#pragma once
#include <atl/geometry/arrangement.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/point_location.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/detail/int256.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

/**
 * @file
 * @brief The trapezoid-map point-location contender that lost the race to
 *        the library persistence-slab atl::geometry::point_location.
 */
namespace atl::experimental {
	/**
	 * @brief The TRAPEZOID MAP contender raced against the library
	 *        persistence-slab @ref atl::geometry::point_location . randomized
	 *        incremental construction of the trapezoidal decomposition of
	 *        the arrangement's edges (de Berg ch. 6), queries descending the
	 *        history DAG. Expected O(n + V + E log(E + 2)) build, expected O(log(E + 2))
	 *        query, O(n + V + E) expected space (vs the slabs' expected
	 *        O(log) query and O(n + V + E log(E + 2)) space).
	 *
	 * Degeneracies via the SHEAR transform, symbolically: all point orders
	 * are lexicographic (x, then y), so no two distinct points share an
	 * x-coordinate and verticals become ordinary steep segments. The
	 * arrangement's edges are interior-disjoint and share only endpoints —
	 * exactly the input class the map wants. Exact rationals throughout
	 * (same machinery as the slabs); same classified @ref
	 * atl::geometry::location results, bit-agreeing with the slab structure
	 * in the differential tests.
	 *
	 * Integral coordinates only, deliberately: the
	 * exact rationals are int128 numerators built from the coordinates, and a
	 * floating path would need the slabs' separate float machinery for a
	 * contender that lost every column. Otherwise the interface mirrors
	 * @ref atl::geometry::point_location . the same constructor, deduction
	 * guide and `location_of`.
	 *
	 * @pre `arr == arrangement_of(segments)` and @p arr outlives this object
	 *      (it is held by pointer); integral coordinates, `|x|, |y| < 2^32`.
	 * @complexity Expected O(n + V + E log(E + 2)) build,
	 * expected O(log(E + 2) + log(V + 2)) query and O(n + V + E) space,
	 * including n original input segments.
	 *
	 * The slabs win all recorded build/query shapes: Intel Xeon Platinum 8358,
	 * GCC 15.2.0, RelWithDebInfo, assertions off, 2026-10-15, -O2 / native.
	 * Dense build 671 / 707 ms versus 375 / 381; dense queries 305 / 320 ms
	 * versus 182 / 203. See the point-location guide for all rows.
	 * @note Not constexpr (runtime randomized decomposition). All DAG walks are iterative.
	 */
	template<geometry::coordinate T>
		requires std::signed_integral<T>
	class point_location_by_trapezoid_map {
		using W = int128;
		using arrangement = geometry::arrangement;
		static constexpr uint32 none = arrangement::none;

		/** @brief An exact rational point (nx/d, ny/d), d > 0. */
		struct rat {
			W nx, ny, d;
		};

		/** @brief A supporting line a x + b y + c = 0, oriented so its
		 *         positive side is shear-above (b > 0 for non-verticals). */
		struct line {
			W a, b, c;
		};

		/**
		 * @brief A trapezoid: bounded above/below by edges (`none` = open to
		 *        infinity), left/right by vertices (`none` = open).
		 *
		 * Neighbors are the up-to-two trapezoids sharing each vertical side.
		 */
		struct trap {
			uint32 top = none, bottom = none;
			uint32 leftp = none, rightp = none;
			uint32 upper_left = none, lower_left = none;
			uint32 upper_right = none, lower_right = none;
			uint32 leaf = 0; // its current leaf in the DAG
		};

		/** @brief History-DAG node kinds: a shear x-test on a vertex, a
		 *         side test against an edge, or a trapezoid leaf. */
		enum class node_kind : uint8 { point,
									   edge,
									   leaf };

		/** @brief A history-DAG node: its kind, the vertex / edge / trapezoid
		 *         it tests or names, and the two children. */
		struct dag_node {
			node_kind kind;
			uint32 value, left = 0, right = 0; // left = shear-before / above
		};

	public:
		/** @brief Deleted: a temporary arrangement would dangle, since the
		 *         structure references it. */
		template<geometry::segment_range R>
			requires geometry::segment_range<R const> && std::same_as<geometry::segment_coordinate_t<R>, T>
		point_location_by_trapezoid_map(arrangement const &&, R const &) = delete;

		/** @brief Builds the randomized map for an unchanged, longer-lived arrangement.
		 * @note Not constexpr (runtime decomposition). */
		template<geometry::segment_range R>
			requires geometry::segment_range<R const> && std::same_as<geometry::segment_coordinate_t<R>, T>
		point_location_by_trapezoid_map(arrangement const &arr, R const &segments)
			: arr_(&arr) {
			atl_assert(std::ranges::size(segments) < (uint64(1) << 32) && arr.vertices.size() < none);
			uint32 const segment_count = uint32(std::ranges::size(segments));
			lines_.reserve(segment_count);
			for (uint32 s = 0; s < segment_count; ++s) {
				auto const sg = std::ranges::begin(segments)[s];
				geometry::detail::slab_frame<T>::check_query(sg.a);
				geometry::detail::slab_frame<T>::check_query(sg.b);
				W a = W(sg.b.y) - W(sg.a.y), b = W(sg.a.x) - W(sg.b.x);
				W c = -(a * W(sg.a.x) + b * W(sg.a.y));
				// positive value = shear-ABOVE: b > 0 for non-verticals, and
				// a < 0 for verticals (under the shear x' = x + eps*y a
				// vertical leans right going up, so above means x < k)
				if (b < 0 || (b == 0 && a > 0)) {
					a = -a;
					b = -b;
					c = -c;
				}
				lines_.push_back({a, b, c});
			}
			uint32 const vertex_count = uint32(arr.vertices.size());
			uint32 const edge_count = arr.edge_count();
			std::vector<std::array<uint32, 2>> support(vertex_count, {none, none});
			for (uint32 h = 0; h < 2 * edge_count; ++h) {
				uint32 const v = arr.origin[h];
				uint32 const s = arr.supporting[arrangement::edge_of(h)][0];
				auto &sup = support[v];
				if (sup[0] == none)
					sup[0] = s;
				else if (sup[1] == none && sup[0] != s
						 && lines_[sup[0]].a * lines_[s].b
								!= lines_[s].a * lines_[sup[0]].b)
					sup[1] = s;
			}
			rats_.reserve(vertex_count);
			for (uint32 v = 0; v < vertex_count; ++v) {
				if (support[v][1] != none) {
					auto const &l0 = lines_[support[v][0]];
					auto const &l1 = lines_[support[v][1]];
					W d = l0.a * l1.b - l1.a * l0.b;
					W nx = l0.b * l1.c - l1.b * l0.c;
					W ny = l0.c * l1.a - l1.c * l0.a;
					if (d < 0) {
						d = -d;
						nx = -nx;
						ny = -ny;
					}
					rats_.push_back({nx, ny, d});
				} else {
					auto const p = arr.vertices[v];
					rats_.push_back({W(T(p.x)), W(T(p.y)), W(1)});
				}
			}

			traps_.push_back({});
			dag_.push_back({node_kind::leaf, 0});
			traps_[0].leaf = 0;
			std::vector<uint32> order(edge_count);
			for (uint32 i = 0; i < edge_count; ++i) order[i] = i;
			uint64 rng = atl::detail::hash_seed() ^ 0x6a09e667f3bcc909ULL;
			for (uint32 i = edge_count; i > 1; --i) {
				rng = atl::detail::splitmix64(rng + 0x9e3779b97f4a7c15ULL);
				std::swap(order[i - 1], order[std::size_t(rng % i)]);
			}
			dir_.reserve(edge_count);
			for (uint32 e = 0; e < edge_count; ++e) {
				auto const sg = std::ranges::begin(segments)[arr.supporting[e][0]];
				bool const is_forward = sg.a.x != sg.b.x ? sg.a.x < sg.b.x : sg.a.y <= sg.b.y;
				dir_.push_back(is_forward ? std::pair{W(sg.b.x) - W(sg.a.x), W(sg.b.y) - W(sg.a.y)}
										  : std::pair{W(sg.a.x) - W(sg.b.x), W(sg.a.y) - W(sg.b.y)});
			}
			for (uint32 const e : order) insert(e);
			// isolated vertices (point segments on nothing) have no edges and
			// therefore no DAG presence: a sorted side catalog answers them
			std::vector<char> touched(vertex_count, 0);
			for (uint32 h = 0; h < 2 * edge_count; ++h) touched[arr.origin[h]] = 1;
			for (uint32 v = 0; v < vertex_count; ++v)
				if (!touched[v]) isolated_.push_back(v); // vertex ids are in
														 // sweep = shear order
		}

		/**
		 * @brief Classified position of @p q — vertex, edge interior, or face.
		 * @complexity Expected O(log(E + 1) + log(V + 1)).
		 * @note Not constexpr (runtime DAG traversal).
		 */
		[[nodiscard]] geometry::location location_of(geometry::point<T> const q) const {
			geometry::detail::slab_frame<T>::check_query(q);
			rat const p{W(q.x), W(q.y), 1};
			uint32 lo = 0, hi = uint32(isolated_.size());
			while (lo < hi) { // shear binary search of the isolated catalog
				uint32 const mid = lo + (hi - lo) / 2;
				int const c = shear_compare(rats_[isolated_[mid]], p);
				if (c == 0) return {geometry::location_kind::vertex, isolated_[mid]};
				if (c < 0)
					lo = mid + 1;
				else
					hi = mid;
			}
			uint32 n = 0;
			while (dag_[n].kind != node_kind::leaf) {
				if (dag_[n].kind == node_kind::point) {
					int const c = shear_compare(p, rats_[dag_[n].value]);
					if (c == 0) return {geometry::location_kind::vertex, dag_[n].value};
					n = c < 0 ? dag_[n].left : dag_[n].right;
				} else {
					uint32 const e = dag_[n].value;
					int const side = line_side(e, p);
					if (side == 0) {
						// on the supporting line: an endpoint is a vertex,
						// the interior of THIS edge's span is the edge
						uint32 const lv = arr_->origin[2 * std::size_t(e)], rv = arr_->origin[2 * std::size_t(e) + 1];
						if (shear_compare(p, rats_[lv]) == 0)
							return {geometry::location_kind::vertex, lv};
						if (shear_compare(p, rats_[rv]) == 0)
							return {geometry::location_kind::vertex, rv};
						if (shear_compare(rats_[lv], p) < 0 && shear_compare(p, rats_[rv]) < 0)
							return {geometry::location_kind::edge, e};
						// beyond the span on the same line (a collinear
						// continuation): the containing region touches the
						// line, either branch converges — descend below
						n = dag_[n].right;
					} else
						n = side > 0 ? dag_[n].left : dag_[n].right;
				}
			}
			uint32 const t = dag_[n].value;
			if (traps_[t].bottom == none)
				return {geometry::location_kind::face, arrangement::unbounded_face};
			return {geometry::location_kind::face, arr_->face[2 * std::size_t(traps_[t].bottom)]};
		}

	private:
		[[nodiscard]] static int sign_of(W const v) {
			return v > 0 ? 1 : v < 0 ? -1
									 : 0;
		}

		/** @brief Three-way shear (lexicographic x, then y) comparison of two
		 *         exact points. */
		[[nodiscard]] int shear_compare(rat const &l, rat const &r) const {
			int const by_x = atl::detail::compare_products(l.nx, r.d, r.nx, l.d);
			if (by_x != 0) return by_x;
			return atl::detail::compare_products(l.ny, r.d, r.ny, l.d);
		}

		/** @brief Sign of edge @p e's supporting-line value at @p p:
		 *         positive = p above. */
		[[nodiscard]] int line_side(uint32 const e, rat const &p) const {
			auto const &l = lines_[arr_->supporting[e][0]];
			auto acc = atl::detail::int256::product(l.a, p.nx);
			acc += atl::detail::int256::product(l.b, p.ny);
			acc += atl::detail::int256::product(l.c, p.d);
			return acc.sign();
		}

		/**
		 * @brief Side of edge @p e for the symbolically perturbed point
		 *        `p + epsilon * dir(s)` with p ON e's line — the
		 *        shared-endpoint tie-break: the sign of the direction against
		 *        the line.
		 */
		[[nodiscard]] int perturbed_side(uint32 const e, uint32 const s) const {
			auto const &l = lines_[arr_->supporting[e][0]];
			auto const d = dir_[s];
			return sign_of(l.a * d.first + l.b * d.second);
		}

		/** @brief Trapezoid containing the point just right (shear) of edge
		 *         @p e's left endpoint, along e. */
		[[nodiscard]] uint32 locate_start(uint32 const e) const {
			rat const &p = rats_[arr_->origin[2 * std::size_t(e)]];
			uint32 n = 0;
			while (dag_[n].kind != node_kind::leaf) {
				if (dag_[n].kind == node_kind::point) {
					int const c = shear_compare(p, rats_[dag_[n].value]);
					// equality: the perturbed point sits right of the vertex
					n = c < 0 ? dag_[n].left : dag_[n].right;
				} else {
					uint32 const g = dag_[n].value;
					int side = line_side(g, p);
					if (side == 0) side = perturbed_side(g, e);
					atl_assert(side != 0); // collinear continuations are
										   // shielded by the endpoint x-node
					n = side > 0 ? dag_[n].left : dag_[n].right;
				}
			}
			return dag_[n].value;
		}

		/** @brief Whether edge-end @p rp stops within trapezoid @p t (at or
		 *         before its right border). */
		[[nodiscard]] bool stops_in(uint32 const t, rat const &rp) const {
			return traps_[t].rightp == none
				|| shear_compare(rp, rats_[traps_[t].rightp]) <= 0;
		}

		uint32 new_leaf(uint32 const t) {
			atl_assert(dag_.size() < none); // node ids are uint32, none reserved
			dag_.push_back({node_kind::leaf, t});
			traps_[t].leaf = uint32(dag_.size()) - 1;
			return traps_[t].leaf;
		}

		uint32 new_trap(trap const &t) {
			atl_assert(traps_.size() < none); // trapezoid ids are uint32, none reserved
			traps_.push_back(t);
			uint32 const id = uint32(traps_.size()) - 1;
			new_leaf(id);
			return id;
		}

		/** @brief Repoint trapezoid @p t's left neighbor links from @p was to
		 *         @p now (no-op for `none`). */
		void replace_left(uint32 const t, uint32 const was, uint32 const now) {
			if (t == none) return;
			if (traps_[t].upper_left == was) traps_[t].upper_left = now;
			if (traps_[t].lower_left == was) traps_[t].lower_left = now;
		}

		/** @brief Repoint trapezoid @p t's right neighbor links from @p was
		 *         to @p now (no-op for `none`). */
		void replace_right(uint32 const t, uint32 const was, uint32 const now) {
			if (t == none) return;
			if (traps_[t].upper_right == was) traps_[t].upper_right = now;
			if (traps_[t].lower_right == was) traps_[t].lower_right = now;
		}

		/** @brief Insert edge @p e: split the trapezoids it crosses, merge the
		 *         runs on each side, and rewrite their leaves into local
		 *         search subtrees. */
		void insert(uint32 const e) {
			uint32 const lv = arr_->origin[2 * std::size_t(e)], rv = arr_->origin[2 * std::size_t(e) + 1];
			rat const &lp = rats_[lv];
			rat const &rp = rats_[rv];

			// the crossed trapezoids, left to right: past a right border, the
			// edge continues through the neighbor on its own side
			std::vector<uint32> crossed{locate_start(e)};
			while (!stops_in(crossed.back(), rp)) {
				trap const &t = traps_[crossed.back()];
				int const side = line_side(e, rats_[t.rightp]);
				atl_assert(side != 0); // interior vertices cannot exist
				crossed.push_back(side > 0 ? t.lower_right : t.upper_right);
				atl_assert(crossed.back() != none);
			}
			std::size_t const k = crossed.size() - 1;
			std::vector<trap> consumed(crossed.size());
			for (std::size_t i = 0; i <= k; ++i) consumed[i] = traps_[crossed[i]];

			bool const cuts_left =
				consumed[0].leftp == none || shear_compare(rats_[consumed[0].leftp], lp) < 0;
			bool const cuts_right =
				consumed[k].rightp == none || shear_compare(rp, rats_[consumed[k].rightp]) < 0;
			uint32 const a_piece =
				cuts_left ? new_trap({consumed[0].top, consumed[0].bottom, consumed[0].leftp, lv,
									  consumed[0].upper_left, consumed[0].lower_left, none, none})
						  : none;
			uint32 const b_piece =
				cuts_right ? new_trap({consumed[k].top, consumed[k].bottom, rv, consumed[k].rightp,
									   none, none, consumed[k].upper_right, consumed[k].lower_right})
						   : none;

			// upper and lower runs: a run BREAKS at a border vertex on its own
			// side of e (that vertex's vertical extension survives there and
			// still splits the region; on the other side e now blocks it)
			std::vector<uint32> upper_of(k + 1), lower_of(k + 1);
			uint32 upper = new_trap({consumed[0].top, e, lv, none, none, none, none, none});
			uint32 lower = new_trap({e, consumed[0].bottom, lv, none, none, none, none, none});
			upper_of[0] = upper;
			lower_of[0] = lower;
			for (std::size_t i = 1; i <= k; ++i) {
				uint32 const w = consumed[i - 1].rightp;
				int const side = line_side(e, rats_[w]);
				// a crossed trapezoid's left-bar side that edges END into can
				// hold a LIVE outer neighbor besides the consumed previous one
				uint32 const outer_ul =
					consumed[i].upper_left == crossed[i - 1] ? none : consumed[i].upper_left;
				uint32 const outer_ll =
					consumed[i].lower_left == crossed[i - 1] ? none : consumed[i].lower_left;
				if (side > 0) { // w above e: the upper run closes at w
					traps_[upper].rightp = w;
					traps_[upper].upper_right = consumed[i - 1].upper_right;
					// the new piece's left bar: [e, w] borders the closed run
					// (its LOWER portion), [w, top] the outer neighbor
					uint32 const next =
						new_trap({consumed[i].top, e, w, none, outer_ul, upper, none, none});
					traps_[upper].lower_right = next;
					replace_left(consumed[i - 1].upper_right, crossed[i - 1], upper);
					upper = next;
				} else { // w below e: the lower run closes at w
					traps_[lower].rightp = w;
					traps_[lower].lower_right = consumed[i - 1].lower_right;
					// mirror: [w, e] borders the closed run (UPPER portion),
					// [bottom, w] the outer neighbor
					uint32 const next =
						new_trap({e, consumed[i].bottom, w, none, lower, outer_ll, none, none});
					traps_[lower].upper_right = next;
					replace_left(consumed[i - 1].lower_right, crossed[i - 1], lower);
					lower = next;
				}
				upper_of[i] = upper;
				lower_of[i] = lower;
				if (outer_ul != none) replace_right(outer_ul, crossed[i], upper_of[i]);
				if (outer_ll != none) replace_right(outer_ll, crossed[i], lower_of[i]);
			}

			traps_[upper].rightp = rv;
			traps_[lower].rightp = rv;
			if (cuts_right) {
				traps_[b_piece].upper_left = upper;
				traps_[b_piece].lower_left = lower;
				traps_[upper].upper_right = b_piece;
				traps_[lower].lower_right = b_piece;
				replace_left(consumed[k].upper_right, crossed[k], b_piece);
				replace_left(consumed[k].lower_right, crossed[k], b_piece);
			} else {
				traps_[upper].upper_right = consumed[k].upper_right;
				traps_[lower].lower_right = consumed[k].lower_right;
				replace_left(consumed[k].upper_right, crossed[k], upper);
				replace_left(consumed[k].lower_right, crossed[k], lower);
			}

			if (cuts_left) {
				traps_[a_piece].upper_right = upper_of[0];
				traps_[a_piece].lower_right = lower_of[0];
				traps_[upper_of[0]].upper_left = a_piece;
				traps_[lower_of[0]].lower_left = a_piece;
				replace_right(consumed[0].upper_left, crossed[0], a_piece);
				replace_right(consumed[0].lower_left, crossed[0], a_piece);
			} else {
				traps_[upper_of[0]].upper_left = consumed[0].upper_left;
				traps_[lower_of[0]].lower_left = consumed[0].lower_left;
				replace_right(consumed[0].upper_left, crossed[0], upper_of[0]);
				replace_right(consumed[0].lower_left, crossed[0], lower_of[0]);
			}

			for (std::size_t i = 0; i <= k; ++i) {
				uint32 const leaf_id = consumed[i].leaf;
				atl_assert(uint64(dag_.size()) + 2 < none); // up to three nodes below
				dag_.push_back({node_kind::edge, e, traps_[upper_of[i]].leaf,
								traps_[lower_of[i]].leaf});
				uint32 sub = uint32(dag_.size()) - 1;
				if (i == k && cuts_right) {
					dag_.push_back(
						{node_kind::point, rv, sub, traps_[b_piece].leaf});
					sub = uint32(dag_.size()) - 1;
				}
				if (i == 0 && cuts_left) {
					dag_.push_back(
						{node_kind::point, lv, traps_[a_piece].leaf, sub});
					sub = uint32(dag_.size()) - 1;
				}
				dag_[leaf_id] = dag_[sub];
			}
		}

		arrangement const *arr_;
		std::vector<line> lines_;
		std::vector<rat> rats_;
		std::vector<std::pair<W, W>> dir_; // set in ctor after lines
		std::vector<trap> traps_;
		std::vector<dag_node> dag_;
		std::vector<uint32> isolated_;
	};

	/** @brief Deduces the integral coordinate from supporting segments. */
	template<geometry::segment_range R>
	point_location_by_trapezoid_map(geometry::arrangement const &, R const &)
		-> point_location_by_trapezoid_map<geometry::segment_coordinate_t<R>>;
}
