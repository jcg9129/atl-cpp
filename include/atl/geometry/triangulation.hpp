#pragma once
#include <atl/ds/flat_hash_map.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <set>
#include <utility>
#include <vector>

namespace atl::geometry {
	/**
	 * @brief A triangulation of a simple polygon: `n - 2` triangles as
	 *        CCW-oriented index triples into the input range (regardless of
	 *        the polygon's orientation), plus the dual TREE — one edge per
	 *        diagonal, connecting the two triangles that share it. Plain
	 *        data: consume `triangles` for per-triangle sums and sampling,
	 *        `dual_edges` for tree DPs and 3-colorings.
	 */
	struct triangulation {
		std::vector<std::array<vertex_index, 3>> triangles;	   ///< Strictly positive CCW triples into the input.
		std::vector<std::array<triangle_index, 2>> dual_edges; ///< Adjacent triangle ids, one pair per diagonal.
	};

	namespace detail {
		/**
		 * @brief Whether @p q lies in the CLOSED triangle (a, b, c), given CCW.
		 * Boundary counts: a vertex on a prospective diagonal must block
		 * the ear, else it would be stranded on a triangle edge.
		 */
		template<coordinate T, sign_policy<T> Policy>
		[[nodiscard]] constexpr bool in_closed_triangle(point<T> const a, point<T> const b, point<T> const c,
														point<T> const q, Policy const p) {
			return p.orientation(a, b, q) >= 0 && p.orientation(b, c, q) >= 0
				&& p.orientation(c, a, q) >= 0;
		}
	}

	namespace detail {
		/**
		 * @brief Ear clipping, the library choice for small r (see the
		 *        adaptive gate in @ref atl::geometry::triangulation_of): repeatedly cut a
		 *        strictly convex vertex whose triangle contains no other
		 *        active vertex. Containment is tested against the
		 *        reflex vertices only — kept in
		 *        an intrusive list — and the test is closed, so no vertex
		 *        ever ends up on a triangle edge. Every emitted triangle has
		 *        strictly positive area; the dual tree is recorded as
		 *        clipping proceeds. Degenerate inputs (fewer than 3
		 *        vertices, or zero area) yield an empty result.
		 * @pre @p poly is simple; `size(poly) < 2^32` (asserted). For
		 *      `int64` coordinates: `|x|, |y| < 2^62`.
		 * @complexity O(n * (r + 1)), r = reflex vertices — O(n) on
		 *             convex input, O(n^2) worst case.
		 */
		template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
		[[nodiscard]] constexpr triangulation ear_clipping_triangulation(R &&poly, Policy const p = {}) {
			using T = vertex_coordinate_t<R>;
			atl_assert(std::ranges::size(poly) < (uint64(1) << 32)); // 32-bit indexed
			uint32 const n = uint32(std::ranges::size(poly));
			triangulation result;
			if (n < 3) return result;
			auto const direction = winding_of(poly, p);
			if (direction == winding::none) return result; // no positive-area triangulation
			auto const first = std::ranges::begin(poly);
			auto const at = [&](uint32 i) -> point<T> { return first[i]; };

			bool const is_reversed = direction == winding::clockwise;
			std::vector<uint32> next(n), prev(n);
			for (uint32 i = 0; i < n; ++i) {
				uint32 const j = i + 1 == n ? 0 : i + 1;
				uint32 const from = is_reversed ? j : i, to = is_reversed ? i : j;
				next[from] = to;
				prev[to] = from;
			}

			constexpr uint32 none = ~uint32(0);
			// triangle owning the active edge that STARTS at vertex u (u ->
			// next[u]); none = an edge of the original polygon. When an ear
			// consumes an owned edge, owner and ear are dual-tree neighbours.
			std::vector<uint32> edge_owner(n, none);

			auto const corner_sign = [&](uint32 v) {
				return p.orientation(at(prev[v]), at(v), at(next[v]));
			};

			// Reflex vertices block closed ear triangles. A straight run on a
			// diagonal has a reflex endpoint there, unless clipping would leave
			// the entire active polygon collinear; that case is guarded below.
			std::vector<uint32> bnext(n, none), bprev(n, none);
			std::vector<char> in_blockers(n, 0);
			uint32 blocker_head = none;
			auto const blocker_add = [&](uint32 v) {
				if (in_blockers[v]) return;
				in_blockers[v] = 1;
				bprev[v] = none;
				bnext[v] = blocker_head;
				if (blocker_head != none) bprev[blocker_head] = v;
				blocker_head = v;
			};
			auto const blocker_remove = [&](uint32 v) {
				if (!in_blockers[v]) return;
				in_blockers[v] = 0;
				(bprev[v] != none ? bnext[bprev[v]] : blocker_head) = bnext[v];
				if (bnext[v] != none) bprev[bnext[v]] = bprev[v];
			};
			uint32 strict_corners = 0;
			for (uint32 v = 0; v < n; ++v) {
				int const turn = corner_sign(v);
				strict_corners += turn != 0;
				if (turn < 0) blocker_add(v);
			}

			auto const is_ear = [&](uint32 v) {
				if (corner_sign(v) <= 0) return false; // strictly convex corners only
				// Preserve a positive-area remainder, including every straight
				// boundary vertex, without scanning straight vertices per ear.
				if (strict_corners == 3
					&& p.orientation(at(prev[prev[v]]), at(prev[v]), at(next[v])) == 0
					&& p.orientation(at(prev[v]), at(next[v]), at(next[next[v]])) == 0) return false;
				point<T> const a = at(prev[v]), b = at(v), c = at(next[v]);
				for (uint32 q = blocker_head; q != none; q = bnext[q]) {
					if (q == prev[v] || q == next[v]) continue; // triangle corners
					if (detail::in_closed_triangle(a, b, c, at(q), p)) return false;
				}
				return true;
			};

			result.triangles.reserve(n - 2);
			result.dual_edges.reserve(n - 3);
			auto const emit = [&](uint32 a, uint32 v, uint32 b) {
				uint32 const t = uint32(result.triangles.size());
				result.triangles.push_back({a, v, b});
				if (edge_owner[a] != none) result.dual_edges.push_back({edge_owner[a], t}); // (a, v)
				if (edge_owner[v] != none) result.dual_edges.push_back({edge_owner[v], t}); // (v, b)
				return t;
			};

			uint32 active = n, v = 0, since_clip = 0;
			while (active > 3) {
				if (is_ear(v)) {
					uint32 const a = prev[v], b = next[v];
					edge_owner[a] = emit(a, v, b); // the new active edge (a, b)
					strict_corners -= uint32(corner_sign(a) != 0) + 1 + uint32(corner_sign(b) != 0);
					next[a] = b;
					prev[b] = a;
					--active;
					// neighbours may have turned convex (never the reverse)
					for (uint32 const u : {a, b}) {
						int const turn = corner_sign(u);
						strict_corners += turn != 0;
						if (turn >= 0) blocker_remove(u);
					}
					v = a;
					since_clip = 0;
				} else {
					v = next[v];
					// a simple polygon always has an ear; a full fruitless lap
					// means the precondition is violated
					++since_clip;
					atl_assert(since_clip <= active);
					if (since_clip > active) return result; // non-simple: bail out
				}
			}
			uint32 const a = v, b = next[v], c = next[b];
			atl_assert(p.orientation(at(a), at(b), at(c)) > 0); // simple => positive finish
			uint32 const last = emit(a, b, c);
			if (edge_owner[c] != none) result.dual_edges.push_back({edge_owner[c], last}); // (c, a)
			return result;
		}
	}

	namespace detail {
		/**
		 * @brief De Berg et al.'s O(n log n) triangulation: sweep-decompose
		 *        the polygon into y-monotone pieces (five vertex classes,
		 *        helper diagonals, a transparent status set — the
		 *        Bentley-Ottmann pattern with input-vertex events only, so
		 *        every decision is one exact orientation), then triangulate
		 *        each piece with the two-chain stack scan. Zero-area
		 *        triangles that straight vertices can force out of the stack
		 *        scan are repaired by local flips across their long edge
		 *        (always interior in a simple polygon). Dual tree from a
		 *        shared-edge map at the end.
		 * @pre As `ear_clipping_triangulation` (the size bound asserted).
		 * @complexity O(n log n)
		 */
		template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
		[[nodiscard]] triangulation monotone_sweep_triangulation(R &&poly, Policy const p = {}) {
			using T = vertex_coordinate_t<R>;
			triangulation result;
			atl_assert(std::ranges::size(poly) < (uint64(1) << 32)); // 32-bit indexed
			uint32 const n = uint32(std::ranges::size(poly));
			if (n < 3) return result;
			auto const direction = winding_of(poly, p);
			if (direction == winding::none) return result;
			auto const first = std::ranges::begin(poly);
			auto const at = [&](uint32 i) -> point<T> { return first[i]; };

			bool const is_reversed = direction == winding::clockwise;
			std::vector<uint32> next(n), prev(n);
			for (uint32 i = 0; i < n; ++i) {
				uint32 const j = i + 1 == n ? 0 : i + 1;
				uint32 const from = is_reversed ? j : i, to = is_reversed ? i : j;
				next[from] = to;
				prev[to] = from;
			}

			auto const above = [&](uint32 const u, uint32 const v) {
				return at(u).y != at(v).y ? at(u).y > at(v).y : at(u).x < at(v).x;
			};
			std::vector<uint32> order(n);
			for (uint32 i = 0; i < n; ++i) order[i] = i;
			std::ranges::sort(order, [&](uint32 const u, uint32 const v) { return above(u, v); });

			enum class vertex_class : uint8 { start,
											  split,
											  finish,
											  merge,
											  regular };
			auto const classify = [&](uint32 const v) {
				bool const is_prev_below = above(v, prev[v]), is_next_below = above(v, next[v]);
				int const turn = p.orientation(at(prev[v]), at(v), at(next[v]));
				if (is_prev_below && is_next_below)
					return turn >= 0 ? vertex_class::start : vertex_class::split;
				if (!is_prev_below && !is_next_below)
					return turn >= 0 ? vertex_class::finish : vertex_class::merge;
				return vertex_class::regular;
			};

			// status: downward boundary edges (interior to their right),
			// identified by their UPPER vertex; ordered left-to-right at the
			// sweep, with the current vertex as a transparent probe.
			// edge id e = the edge from e to next[e] (e is the upper end).
			constexpr uint32 nil = ~uint32(0);
			std::vector<point<T>> pts(n);
			for (uint32 i = 0; i < n; ++i) pts[i] = at(i);

			struct probe {};

			uint32 current = 0;

			struct edge_less {
				std::vector<uint32> const *nextp;
				std::vector<point<T>> const *ptsp;
				Policy policy;
				uint32 const *currentp;

				point<T> pt(uint32 i) const { return (*ptsp)[i]; }

				using is_transparent [[maybe_unused]] = void;

				bool ab(uint32 u, uint32 v) const { // above
					return pt(u).y != pt(v).y ? pt(u).y > pt(v).y : pt(u).x < pt(v).x;
				}

				bool operator()(uint32 const e, uint32 const f) const {
					if (e == f) return false;
					uint32 const le = (*nextp)[e], lf = (*nextp)[f];
					if (pt(e) == pt(f)) {
						int const o = policy.orientation(pt(e), pt(le), pt(lf));
						if (o != 0) return o > 0;
						return e < f;
					}
					if (ab(e, f)) { // f's upper is lower: test it against e
						int const o = policy.orientation(pt(e), pt(le), pt(f));
						if (o != 0) return o > 0; // f on the +x side => e left
						return e < f;
					}
					int const o = policy.orientation(pt(f), pt(lf), pt(e));
					if (o != 0) return o < 0; // e on the -x side of f => e left
					return e < f;
				}

				// edge strictly left of the current vertex <=> the vertex is
				// on the +x side of the downward edge
				bool operator()(uint32 const e, probe) const {
					return policy.orientation(pt(e), pt((*nextp)[e]), pt(*currentp)) > 0;
				}

				bool operator()(probe, uint32 const e) const {
					return policy.orientation(pt(e), pt((*nextp)[e]), pt(*currentp)) < 0;
				}
			};

			std::set<uint32, edge_less> status(edge_less{&next, &pts, p, &current});
			std::vector<uint32> helper(n, nil);
			std::vector<vertex_class> cls(n);
			for (uint32 v = 0; v < n; ++v) cls[v] = classify(v);

			std::vector<std::pair<uint32, uint32>> diagonals;
			auto const fix_helper = [&](uint32 const edge, uint32 const v) {
				uint32 const h = helper[edge];
				if (h != nil && cls[h] == vertex_class::merge)
					diagonals.emplace_back(v, h);
			};
			auto const left_edge_of = [&](uint32 const v) {
				current = v;
				auto it = status.lower_bound(probe{});
				atl_assert(it != status.begin()); // simple polygon: exists
				return *std::prev(it);
			};

			for (uint32 const v : order) {
				current = v;
				uint32 const e_in = prev[v]; // edge (prev[v] -> v)
				switch (cls[v]) {
				case vertex_class::start:
					status.insert(v);
					helper[v] = v;
					break;
				case vertex_class::finish:
					fix_helper(e_in, v);
					status.erase(e_in);
					break;
				case vertex_class::split: {
					uint32 const e_j = left_edge_of(v);
					diagonals.emplace_back(v, helper[e_j]);
					helper[e_j] = v;
					status.insert(v);
					helper[v] = v;
					break;
				}
				case vertex_class::merge: {
					fix_helper(e_in, v);
					status.erase(e_in);
					uint32 const e_j = left_edge_of(v);
					fix_helper(e_j, v);
					helper[e_j] = v;
					break;
				}
				case vertex_class::regular:
					if (above(prev[v], v)) { // interior right: on the left chain
						fix_helper(e_in, v);
						status.erase(e_in);
						status.insert(v);
						helper[v] = v;
					} else {
						uint32 const e_j = left_edge_of(v);
						fix_helper(e_j, v);
						helper[e_j] = v;
					}
					break;
				}
			}

			std::vector<std::vector<uint32>> out_edges(n);
			for (uint32 v = 0; v < n; ++v) {
				out_edges[v].push_back(next[v]);
				out_edges[v].push_back(prev[v]); // reverse boundary edges: the
												 // face walk's (v -> u) lookups
			}
			for (auto const [u, v] : diagonals) {
				out_edges[u].push_back(v);
				out_edges[v].push_back(u);
			}
			for (uint32 v = 0; v < n; ++v)
				std::ranges::sort(out_edges[v], [&](uint32 const x, uint32 const y) {
					auto const o = at(v), a = at(x), b = at(y);
					auto const half = [&](point<T> const q) {
						return q.y < o.y ? 0 : q.y > o.y ? 2
										 : q.x >= o.x	 ? 1
														 : 3;
					};
					int const ha = half(a), hb = half(b);
					return ha != hb ? ha < hb : p.orientation(o, a, b) > 0;
				});

			flat_hash_map<uint64, uint32> slot;
			for (uint32 v = 0; v < n; ++v)
				for (uint32 k = 0; k < uint32(out_edges[v].size()); ++k)
					slot[(uint64(v) << 32) | out_edges[v][k]] = k;
			// next directed edge in the face of (u -> v): rotate CW from
			// (v -> u) at v = the clockwise-adjacent slot
			auto const face_next = [&](uint32 const u, uint32 const v) {
				uint32 const k = slot.at((uint64(v) << 32) | u);
				auto const &around = out_edges[v];
				uint32 const k2 = k == 0 ? uint32(around.size()) - 1 : k - 1;
				return std::pair{v, around[k2]};
			};

			flat_hash_map<uint64, uint8> used; // directed edge visited
			auto const emit_ccw = [&](uint32 a, uint32 b, uint32 c) {
				if (p.orientation(at(a), at(b), at(c)) < 0) std::swap(b, c);
				result.triangles.push_back({a, b, c});
			};
			result.triangles.reserve(n - 2);

			auto const triangulate_piece = [&](std::vector<uint32> const &piece) {
				uint32 const m = uint32(piece.size());
				if (m == 3) {
					emit_ccw(piece[0], piece[1], piece[2]);
					return;
				}

				uint32 top_pos = 0, bottom_pos = 0;
				for (uint32 k = 1; k < m; ++k) {
					if (above(piece[k], piece[top_pos])) top_pos = k;
					if (above(piece[bottom_pos], piece[k])) bottom_pos = k;
				}
				// merged descending sequence with chain flags; the piece is
				// CCW, so forward from top = the LEFT chain
				std::vector<std::pair<uint32, int>> merged;
				merged.reserve(m);
				{
					uint32 i = top_pos + 1 == m ? 0 : top_pos + 1, j = top_pos == 0 ? m - 1 : top_pos - 1;
					merged.emplace_back(piece[top_pos], 0);
					while (i != bottom_pos || j != bottom_pos) {
						bool takes_left;
						if (i == bottom_pos)
							takes_left = false;
						else if (j == bottom_pos)
							takes_left = true;
						else
							takes_left = above(piece[i], piece[j]);
						if (takes_left) {
							merged.emplace_back(piece[i], +1);
							i = (i + 1) % m;
						} else {
							merged.emplace_back(piece[j], -1);
							j = j == 0 ? m - 1 : j - 1;
						}
					}
					merged.emplace_back(piece[bottom_pos], 0);
				}
				std::vector<std::pair<uint32, int>> stack{merged[0], merged[1]};
				for (uint32 k = 2; k + 1 < m; ++k) {
					auto const [v, chain] = merged[k];
					if (chain != stack.back().second && stack.back().second != 0) {
						auto const saved_top = stack.back();
						for (uint32 t = 0; t + 1 < uint32(stack.size()); ++t)
							emit_ccw(v, stack[t].first, stack[t + 1].first);
						stack.clear();
						stack.push_back(saved_top);
						stack.emplace_back(v, chain);
					} else {
						auto last = stack.back();
						stack.pop_back();
						while (!stack.empty()) {
							int const turn = p.orientation(at(stack.back().first),
														   at(last.first), at(v));
							if (chain * turn <= 0) break; // reflex or straight
							emit_ccw(stack.back().first, last.first, v);
							last = stack.back();
							stack.pop_back();
						}
						stack.push_back(last);
						stack.emplace_back(v, chain);
					}
				}
				auto const bottom = merged[m - 1].first;
				for (uint32 t = 0; t + 1 < uint32(stack.size()); ++t)
					emit_ccw(bottom, stack[t].first, stack[t + 1].first);
			};

			std::vector<uint32> piece;
			for (uint32 v = 0; v < n; ++v)
				for (uint32 const w : out_edges[v]) {
					if (used.at_or((uint64(v) << 32) | w, uint8(0))) continue;
					piece.clear();
					uint32 cu = v, cv = w;
					while (!used.at_or((uint64(cu) << 32) | cv, uint8(0))) {
						used[(uint64(cu) << 32) | cv] = 1;
						piece.push_back(cu);
						auto const [nu, nv] = face_next(cu, cv);
						cu = nu;
						cv = nv;
					}
					auto const vertices = piece | std::views::transform(at);
					if (winding_of(vertices, p) != winding::counterclockwise) continue;
					triangulate_piece(piece);
				}

			// Each undirected edge maps to up to two triangles; a flip
			// patches the four edges whose owners change, so finding the
			// neighbour across the long edge is O(1)
			flat_hash_map<uint64, std::array<uint32, 2>> edge_tri;
			auto const edge_key = [](uint32 a, uint32 b) {
				return a < b ? (uint64(a) << 32) | b : (uint64(b) << 32) | a;
			};
			auto const attach = [&](uint32 const a, uint32 const b, uint32 const t) {
				uint64 const key = edge_key(a, b);
				if (!edge_tri.try_emplace(key, std::array<uint32, 2>{t, nil})) edge_tri[key][1] = t;
			};
			auto const reattach = [&](uint32 const a, uint32 const b, uint32 const from, uint32 const to) {
				auto &owners = edge_tri[edge_key(a, b)];
				owners[owners[0] == from ? 0 : 1] = to;
			};
			for (uint32 t = 0; t < uint32(result.triangles.size()); ++t) {
				auto const &tri = result.triangles[t];
				for (int e = 0; e < 3; ++e) attach(tri[e], tri[(e + 1) % 3], t);
			}
			for (uint32 t = 0; t < uint32(result.triangles.size()); ++t) {
				auto &tri = result.triangles[t];
				if (p.orientation(at(tri[0]), at(tri[1]), at(tri[2])) != 0) continue;

				auto const between = [&](uint32 a, uint32 b, uint32 c) {
					// b between a and c on the line (lexicographic works)
					auto const lo = std::min(at(a), at(c)), hi = std::max(at(a), at(c));
					return !(at(b) < lo) && !(hi < at(b));
				};
				uint32 mid = 0;
				for (int e = 0; e < 3; ++e)
					if (between(tri[(e + 1) % 3], tri[e], tri[(e + 2) % 3])) mid = uint32(e);
				uint32 const b = tri[mid], a = tri[(mid + 1) % 3], c = tri[(mid + 2) % 3];
				auto const &long_owners = edge_tri[edge_key(a, c)];
				uint32 const other = long_owners[0] == t ? long_owners[1] : long_owners[0];
				atl_assert(other != nil); // the long edge is interior
				auto const &tr = result.triangles[other];
				uint32 d = nil;
				for (auto const vertex : tr)
					if (vertex != a && vertex != c) d = vertex;
				atl_assert(d != nil);
				result.triangles[t] = {a, b, d};
				result.triangles[other] = {b, c, d};
				if (p.orientation(at(a), at(b), at(d)) < 0)
					std::swap(result.triangles[t][1], result.triangles[t][2]);
				if (p.orientation(at(b), at(c), at(d)) < 0)
					std::swap(result.triangles[other][1], result.triangles[other][2]);
				// (b, c) moves from t to other, (d, a) from other to t, (b, d)
				// is new in both; (a, c) is gone and is never asked for again
				reattach(b, c, t, other);
				reattach(d, a, other, t);
				edge_tri[edge_key(b, d)] = {t, other};
			}

			result.dual_edges.reserve(n - 3);
			flat_hash_map<uint64, uint32> owner;
			for (uint32 t = 0; t < uint32(result.triangles.size()); ++t) {
				auto const &tri = result.triangles[t];
				for (int e = 0; e < 3; ++e) {
					uint64 const key = edge_key(tri[e], tri[(e + 1) % 3]);
					if (auto const o = owner.at_or(key, nil); o != nil)
						result.dual_edges.push_back({o, t});
					else
						owner[key] = t;
				}
			}
			return result;
		}
	}

	/**
	 * @brief Triangulates a simple polygon by ear clipping at compile time or runtime.
	 * Either winding and straight boundary vertices are accepted; the result has
	 * the same positive CCW orientation and original indices as triangulation_of.
	 * Fewer than three vertices or zero area yields an empty result.
	 * Prefer triangulation_of for large reflex-heavy polygons and its O(n log n) bound.
	 * @tparam R Sized random-access vertex range.
	 * @tparam Policy Sign policy for the input coordinate type.
	 * @param poly Cyclic polygon with implicit closing edge.
	 * @param p Predicate policy.
	 * @return Positive CCW triangles and their dual tree.
	 * @pre The polygon is simple; fewer than 2^32 vertices (asserted).
	 * @pre int64 coordinates have magnitude below 2^62 (asserted); int32 uses its full range.
	 * Floating predicates inherit orientation expansion bounds; approximate policies
	 * must give consistent signs on the input topology.
	 * @complexity O(n (r + 1)) time, r = reflex vertices, quadratic
	 * worst case; O(n) storage.
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] constexpr triangulation triangulation_of_by_ear_clipping(R &&poly, Policy const p = {}) {
		atl_assert(std::ranges::size(poly) < (uint64(1) << 32));
		for (auto const vertex : poly) {
			point<vertex_coordinate_t<R>> const value = vertex;
			detail::check_difference_bounds(value);
			detail::check_floating_magnitude(value);
		}
		return detail::ear_clipping_triangulation(std::forward<R>(poly), p);
	}

	/**
	 * @brief Triangulates the SIMPLE polygon @p poly (either orientation,
	 *        collinear vertices allowed): `n - 2` strictly-positive CCW
	 *        triangles plus the dual tree. ADAPTIVE: one O(n) pass counts
	 *        the non-convex (reflex + straight) vertices r; small r runs
	 *        ear clipping (O(n * (r + 1)), no sorting), otherwise the de Berg monotone-decomposition sweep
	 *        (O(n log n) sweep work). The dispatch threshold is 8 bit_width(n).
	 *
	 * Degenerate inputs (fewer than 3 vertices, zero area) yield an empty
	 * result.
	 * @note Not constexpr (the sweep path uses runtime set/map machinery).
	 * All polygon walks and work stacks are iterative; no recursion per vertex.
	 *
	 * @tparam R Sized random-access vertex range.
	 * @tparam Policy Sign policy for the input coordinate type.
	 * @param poly Cyclic polygon with its closing edge implicit.
	 * @param p Predicate policy.
	 * @return Positive CCW triples into poly and their dual tree, or empty for degenerate input.
	 * @pre @p poly is simple. For `int64` coordinates: `|x|, |y| < 2^62`.
	 * @pre Fewer than 2^32 vertices (asserted); int32 accepts its full range.
	 * Floating predicates inherit the orientation expansion bounds; approximate policies
	 * require consistent signs on the input topology.
	 * @complexity O(n log(n + 1)) geometric work and O(n) allocated memory;
	 * expected time with constant-time hash-map operations.
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] triangulation triangulation_of(R &&poly, Policy const p = {}) {
		atl_assert(std::ranges::size(poly) < (uint64(1) << 32));
		uint32 const n = uint32(std::ranges::size(poly));
		for (auto const vertex : poly) {
			point<vertex_coordinate_t<R>> const value = vertex;
			detail::check_difference_bounds(value);
			detail::check_floating_magnitude(value);
		}
		if (n < 3) return {};
		auto const first = std::ranges::begin(poly);
		uint64 non_convex = 0;
		bool const is_counterclockwise = winding_of(poly, p) != winding::clockwise;
		for (uint32 v = 0; v < n; ++v) {
			uint32 const j = v + 1 == n ? 0 : v + 1;
			uint32 const k = j + 1 == n ? 0 : j + 1;
			int const turn = p.orientation(first[v], first[j], first[k]);
			non_convex += (is_counterclockwise ? turn : -turn) <= 0;
		}
		// Bounding the initial blockers by 8 bit_width(n) caps the ear path at O(n log n).
		uint64 const log_n = uint64(std::bit_width(n));
		if (non_convex <= 8 * log_n) return detail::ear_clipping_triangulation(poly, p);
		return detail::monotone_sweep_triangulation(poly, p);
	}
}
