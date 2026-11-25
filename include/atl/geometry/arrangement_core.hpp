#pragma once

/** @file
 * @brief Policy-based arrangements with scalar hashing and without adaptive floating kernels.
 */
#include <atl/ds/detail/edge_id_table.hpp>
#include <atl/geometry/angular_sort_core.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/segment_core.hpp>
#include <atl/geometry/segment_intersections_core.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::geometry {
	/**
	 * @brief The planar subdivision induced by a set of segments — vertices
	 *        (endpoints and intersection points), edges (maximal uncut
	 *        pieces), and faces (connected regions of the complement,
	 *        holes attached), as a half-edge structure. Built by
	 *        @ref atl::geometry::arrangement_of.
	 *
	 * Half-edge `h`: its opposite is `twin(h) = h ^ 1`, its undirected edge
	 * `h >> 1`; `next[h]` walks the boundary cycle of `face[h]` with the
	 * face's interior on the LEFT (bounded-face outer boundaries are CCW
	 * cycles, hole boundaries CW). Half-edge `2e` of every edge points
	 * sweep-forward (lexicographically increasing), so its left side is the
	 * region ABOVE the edge. Vertex ids are in sweep order.
	 *
	 * COMBINATORIALLY EXACT for integral inputs: vertex identity comes from
	 * the sweep's exact rational event points and every angular or turn
	 * decision uses exact integer directions. The stored vertex POSITIONS
	 * are rounded `float64` constructions (crossings are rationals), as are
	 * face areas.
	 */
	struct arrangement {
		/** @brief Sentinel for an absent vertex, half-edge or face. */
		static constexpr uint32 none = ~uint32(0);

		/**
		 * @brief One face: its outer boundary, its inner boundaries (holes and
		 * slits), the isolated vertices it contains, and its area.
		 */
		struct face_data {
			uint32 outer;				  ///< a half-edge of the CCW outer cycle;
										  ///< `none` for the unbounded face
			std::vector<uint32> holes;	  ///< one half-edge per inner (CW or
										  ///< zero-area slit) cycle
			std::vector<uint32> isolated; ///< vertex ids with no incident edge
			float64 area;				  ///< sum of the signed cycle areas: the
										  ///< bounded region's area net of holes;
										  ///< non-positive in exact arithmetic for the unbounded face (stored rounded)
		};

		std::vector<point<float64>> vertices; ///< positions, sweep order
		std::vector<uint32> origin;			  ///< per half-edge: source vertex
		std::vector<uint32> next;			  ///< per half-edge: cycle successor
		std::vector<uint32> face;			  ///< per half-edge: face id
		/**
		 * @brief Per undirected edge `e`: the input segments containing the piece
		 * (more than one where inputs overlap).
		 */
		std::vector<std::vector<uint32>> supporting;
		std::vector<face_data> faces; ///< `faces[0]` is the unbounded face
		/** @brief Index of the unbounded face. */
		static constexpr uint32 unbounded_face = 0;

		/** @brief Opposite half-edge id. */
		[[nodiscard]] static constexpr uint32 twin(uint32 const h) { return h ^ 1; }

		/** @brief Undirected edge id of a half-edge. */
		[[nodiscard]] static constexpr uint32 edge_of(uint32 const h) { return h >> 1; }

		/** @brief Target vertex of half-edge h. @pre h is a valid half-edge. */
		[[nodiscard]] constexpr uint32 to(uint32 const h) const { return origin[twin(h)]; }

		/** @brief Number of undirected edges. */
		[[nodiscard]] constexpr uint32 edge_count() const { return uint32(supporting.size()); }
	};

	namespace detail {
		/**
		 * @brief A `(uint32 segment) -> point<T>` callback giving a segment's
		 *        sweep-forward direction (see `assemble_arrangement`).
		 */
		template<typename Function, typename T>
		concept piece_direction_function =
			coordinate<T> && std::regular_invocable<Function &, uint32>
			&& std::convertible_to<std::invoke_result_t<Function &, uint32>, point<T>>;

		/**
		 * @brief A `(uint32 vertex) -> std::pair<uint32, uint32>` callback
		 *        giving the segment strictly below a vertex and its piece
		 *        index (see `assemble_arrangement`).
		 */
		template<typename Function>
		concept piece_anchor_function =
			std::regular_invocable<Function &, uint32>
			&& std::convertible_to<std::invoke_result_t<Function &, uint32>, std::pair<uint32, uint32>>;

		/**
		 * @brief The shared back half of the arrangement builders: given the
		 *        vertex positions in SWEEP ORDER (ids must be sweep-sorted —
		 *        cycle classification and nesting find leftmost vertices by
		 *        id), the vertex ids along each segment (in order along it)
		 *        and each segment's sweep-forward direction, builds pieces
		 *        (overlaps merged), stitches half-edges angularly, walks and
		 *        classifies cycles, and nests inner cycles and isolated
		 *        vertices into faces through @p anchor_of.
		 *
		 * @tparam T coordinate type of the edge directions (callers widen
		 *         integral coordinates so a direction cannot overflow).
		 * @param along per segment, the ids of the vertices on it, sweep-sorted
		 *        along the segment (consecutive ids bound one piece).
		 * @param forward_direction `(uint32 segment) -> point<T>`: the
		 *        segment's sweep-forward (lexicographically increasing)
		 *        direction vector, used only for exact angular decisions.
		 * @param anchor_of `(uint32 vertex) -> std::pair<uint32, uint32>`:
		 *        the segment strictly below the vertex and the index of its
		 *        piece spanning the vertex's x, or `{none, none}` when
		 *        nothing is below. Called only for the leftmost vertex of
		 *        each inner cycle and for isolated vertices, so an O(n)
		 *        on-demand scan is acceptable.
		 * @pre Fewer than 2^32 segments and vertices and 2^31 pieces (asserted).
		 */
		template<coordinate T, piece_direction_function<T> ForwardDirection, piece_anchor_function AnchorOf>
		[[nodiscard]] arrangement assemble_arrangement(std::vector<point<float64>> vertices,
													   std::vector<std::vector<uint32>> const &along,
													   ForwardDirection &&forward_direction,
													   AnchorOf &&anchor_of) {
			atl_assert(uint64(along.size()) < (uint64(1) << 32));
			atl_assert(uint64(vertices.size()) < (uint64(1) << 32));
			uint32 const segment_count = uint32(along.size());
			arrangement out;
			out.vertices = std::move(vertices);

			// Consecutive cuts bound a piece; identical vertex pairs merge overlaps.
			atl::detail::edge_id_table edge_id;
			std::vector<point<T>> edge_direction; // sweep-forward, per edge
			std::vector<std::vector<uint32>> piece_edge(segment_count);
			for (uint32 s = 0; s < segment_count; ++s)
				for (uint32 j = 0; j + 1 < uint32(along[s].size()); ++j) {
					uint32 const a = along[s][j], b = along[s][j + 1];
					uint64 const key = (uint64(a) << 32) | b;
					uint32 e;
					if (uint32 const found = edge_id.lookup_id(key, arrangement::none); found != arrangement::none)
						e = found;
					else {
						atl_assert(uint64(edge_direction.size()) + 1 < (uint64(1) << 31));
						e = uint32(edge_direction.size());
						edge_id.insert_entry(key, e);
						edge_direction.push_back(forward_direction(s));
						out.origin.push_back(a);
						out.origin.push_back(b);
						out.supporting.emplace_back();
					}
					out.supporting[e].push_back(s);
					piece_edge[s].push_back(e);
				}
			// the 2E half-edge ids must stay below the `none` sentinel
			atl_assert(uint64(edge_direction.size()) < (uint64(1) << 31));
			uint32 const half_count = uint32(out.origin.size());

			// Rotating clockwise at the target keeps the walked face on the left.
			auto const direction = [&](uint32 const h) {
				auto const d = edge_direction[arrangement::edge_of(h)];
				return (h & 1) == 0 ? d : point<T>{-d.x, -d.y};
			};
			std::vector<std::vector<uint32>> around(out.vertices.size());
			for (uint32 h = 0; h < half_count; ++h) around[out.origin[h]].push_back(h);
			polar_angle_less<> const angle_less{};
			std::vector<uint32> slot(half_count);
			for (auto &ring : around) {
				std::ranges::sort(ring, [&](uint32 const x, uint32 const y) {
					return angle_less(direction(x), direction(y));
				});
				for (uint32 k = 0; k < uint32(ring.size()); ++k) slot[ring[k]] = k;
			}
			out.next.assign(half_count, arrangement::none);
			for (uint32 h = 0; h < half_count; ++h) {
				auto const &ring = around[out.to(h)];
				uint32 const k = slot[arrangement::twin(h)];
				out.next[h] = ring[k == 0 ? uint32(ring.size()) - 1 : k - 1];
			}

			// Twin halves in one cycle cancel (a slit walk); among the
			// surviving directions the angularly lowest is an OUT-half for a
			// CCW (outer) cycle and an IN-half for a CW (inner) one; all
			// canceled = a pure slit, which nests like a hole.
			std::vector<uint32> cycle(half_count, arrangement::none);
			std::vector<uint32> cycle_head; // one half-edge per cycle
			std::vector<uint32> cycle_low;	// its leftmost (minimum) vertex
			std::vector<bool> cycle_outer;
			for (uint32 h0 = 0; h0 < half_count; ++h0) {
				if (cycle[h0] != arrangement::none) continue;
				uint32 const id = uint32(cycle_head.size());
				uint32 low = out.origin[h0];
				for (uint32 h = h0; cycle[h] == arrangement::none; h = out.next[h]) {
					cycle[h] = id;
					low = std::min(low, out.origin[h]);
				}

				bool is_found = false, is_found_out = false;
				point<T> best{};
				for (uint32 h = out.next[h0];; h = out.next[h]) {
					for (int side = 0; side < 2; ++side) {
						uint32 const g = side == 0 ? h : arrangement::twin(h);
						if (out.origin[g] != low) continue;
						if (cycle[arrangement::twin(h)] == id)
							continue; // both halves present: a slit edge, cancels
						auto const d = direction(g);
						if (!is_found || angle_less(d, best)) {
							is_found = true;
							is_found_out = side == 0;
							best = d;
						}
					}
					if (h == h0) break;
				}
				cycle_head.push_back(h0);
				cycle_low.push_back(low);
				cycle_outer.push_back(is_found && is_found_out);
			}

			// Inner cycles join the region above their leftmost vertex through the
			// below-anchor edge's forward half (left side up); no anchor means unbounded.
			// Isolated vertices nest the same way.
			uint32 const cycle_count = uint32(cycle_head.size());
			std::vector<uint32> parent(cycle_count + 1);
			for (uint32 i = 0; i <= cycle_count; ++i) parent[i] = i;
			auto find = [&](uint32 x) {
				while (parent[x] != x) x = parent[x] = parent[parent[x]];
				return x;
			};
			uint32 const outside = cycle_count; // the virtual unbounded node
			auto const region_above = [&](uint32 const v) {
				auto const [s, piece] = anchor_of(v);
				if (s == arrangement::none) return outside;
				return cycle[2 * piece_edge[s][piece]];
			};
			for (uint32 c = 0; c < cycle_count; ++c)
				if (!cycle_outer[c]) parent[find(c)] = find(region_above(cycle_low[c]));

			std::vector<uint32> face_of_component(cycle_count + 1, arrangement::none);
			out.faces.push_back({arrangement::none, {}, {}, 0.0});
			face_of_component[find(outside)] = arrangement::unbounded_face;
			for (uint32 c = 0; c < cycle_count; ++c) {
				if (!cycle_outer[c]) continue;
				uint32 const f = uint32(out.faces.size());
				atl_assert(face_of_component[find(c)] == arrangement::none);
				face_of_component[find(c)] = f;
				out.faces.push_back({cycle_head[c], {}, {}, 0.0});
			}
			out.face.assign(half_count, arrangement::none);
			for (uint32 c = 0; c < cycle_count; ++c) {
				uint32 const f = face_of_component[find(c)];
				atl_assert(f != arrangement::none);
				if (!cycle_outer[c]) out.faces[f].holes.push_back(cycle_head[c]);
				for (uint32 h = cycle_head[c];;) {
					out.face[h] = f;
					h = out.next[h];
					if (h == cycle_head[c]) break;
				}
			}
			for (uint32 v = 0; v < uint32(out.vertices.size()); ++v) {
				if (!around[v].empty()) continue;
				uint32 const component = find(region_above(v));
				out.faces[face_of_component[component]].isolated.push_back(v);
			}

			for (uint32 c = 0; c < cycle_count; ++c) {
				// shoelace about the cycle's own first vertex: about the
				// origin, a small face far away sums products ~|position|^2
				// whose rounding swamps the face's area
				auto const anchor = out.vertices[out.origin[cycle_head[c]]];
				float64 doubled = 0;
				for (uint32 h = cycle_head[c];;) {
					auto const p = out.vertices[out.origin[h]] - anchor;
					auto const q = out.vertices[out.to(h)] - anchor;
					doubled += p.x * q.y - p.y * q.x;
					h = out.next[h];
					if (h == cycle_head[c]) break;
				}
				out.faces[out.face[cycle_head[c]]].area += doubled / 2;
			}
			return out;
		}
	}

	/**
	 * @brief Builds the @ref atl::geometry::arrangement of @p segments: one pass of the
	 *        Bentley-Ottmann engine reporting EVERY event (plain endpoints
	 *        included), then the shared assembly — pieces cut per segment,
	 *        overlapping pieces merged, angular half-edge stitching, cycle
	 *        classification by the exact turn structure at each cycle's
	 *        leftmost vertex, and hole/slit cycles nested into faces through
	 *        the sweep's below-neighbor.
	 *
	 * Degenerate inputs follow the sweep's contract: overlaps collapse into
	 * shared edges (all supporting segments listed), k-fold crossings make
	 * one vertex, point segments become vertices (isolated when on nothing).
	 *
	 * @tparam R Sized random-access segment range.
	 * @param segments Closed input segments; all degeneracies are allowed.
	 * @return Owned half-edge subdivision, with the unbounded face at index zero.
	 * @pre For integral coordinates: `|x|, |y| < 2^32` (asserted); floating
	 *      inputs get best-effort ordering.
	 * @complexity Expected O((n + I) log(n + I + 1)) time, O(n + I) storage, I = segment/event incidences.
	 * Includes angular stitching and overlap support lists.
	 * @pre int32 accepts its full range; fewer than 2^32 vertices and 2^31 edges (asserted).
	 * @note Not constexpr (runtime sweep and hash containers). All walks are iterative.
	 */
	template<segment_range R>
	[[nodiscard]] arrangement arrangement_of(R &&segments) {
		using T = segment_coordinate_t<R>;
		detail::segment_sweep<T> sweep(segments, /*report_singletons=*/true);
		uint32 const segment_count = uint32(std::ranges::size(segments));

		// the sweep pass: vertices in event order (= sweep order), the vertex
		// ids along each segment, and per vertex the below-anchor (segment +
		// its piece counter at event time) for the later hole nesting
		std::vector<point<float64>> vertices;
		std::vector<std::vector<uint32>> along(segment_count);
		std::vector<std::pair<uint32, uint32>> anchors;
		while (sweep.advance()) {
			auto const ev = sweep.current_event();
			atl_assert(vertices.size() < arrangement::none);
			uint32 const v = uint32(vertices.size());
			vertices.push_back(ev.at);
			anchors.push_back(ev.below == intersection_event::no_below
								  ? std::pair{arrangement::none, arrangement::none}
								  : std::pair{ev.below,
											  uint32(along[ev.below].size()) - 1});
			auto const touch = [&](uint32 const s) {
				if (along[s].empty() || along[s].back() != v) along[s].push_back(v);
			};
			for (auto const s : ev.starting) touch(s);
			for (auto const s : ev.passing) touch(s);
			for (auto const s : ev.ending) touch(s);
		}
		// integral directions difference in int64: int32 spans reach 2^32
		// (no sweep bound below 64 bits), int64 ones stay under 2^33
		using Direction = std::conditional_t<std::floating_point<T>, T, int64>;
		auto const forward_direction = [&](uint32 const s) {
			auto const seg = std::ranges::begin(segments)[s];
			bool const is_forward = seg.a.x != seg.b.x ? seg.a.x < seg.b.x : seg.a.y <= seg.b.y;
			point<Direction> const a{Direction(seg.a.x), Direction(seg.a.y)};
			point<Direction> const b{Direction(seg.b.x), Direction(seg.b.y)};
			return is_forward ? b - a : a - b;
		};
		return detail::assemble_arrangement<Direction>(
			std::move(vertices), along, forward_direction,
			[&](uint32 const v) { return anchors[v]; });
	}
}
