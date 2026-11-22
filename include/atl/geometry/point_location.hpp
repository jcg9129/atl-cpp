#pragma once
#include <atl/ds/detail/persistent_arena.hpp>
#include <atl/ds/detail/treap_base.hpp>
#include <atl/geometry/arrangement.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/geometry/segment_intersections.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/detail/int256.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::geometry {
	/** @brief What a query point of @ref atl::geometry::point_location landed on. */
	enum class location_kind { vertex, ///< Coincides with a vertex.
							   edge,   ///< Lies in an edge interior.
							   face	   ///< Lies in an open face.
	};

	/**
	 * @brief A classified position in an @ref atl::geometry::arrangement . the vertex it coincides
	 * with, the edge whose interior it lies on, or the face containing it.
	 */
	struct location {
		location_kind kind; ///< Kind selecting the index interpretation.
		uint32 index;		///< vertex / edge / face id in the arrangement

		/** @brief Memberwise equality. */
		[[nodiscard]] friend constexpr bool operator==(location const &, location const &) = default;
	};

	namespace detail {
		/**
		 * @brief The slab decomposition both point-location modes share:
		 *        normalized supporting lines, EXACT vertex rationals
		 *        (recomputed from incident supporting lines — the
		 *        arrangement itself stores rounded positions only), the
		 *        distinct-x boundary groups with their vertex ranges and
		 *        vertical-edge catalogs, and per-boundary lists of the
		 *        non-vertical edges entering and leaving. The online
		 *        @ref atl::geometry::point_location runs a persistent pass over it, the
		 *        offline @ref atl::geometry::locations_of a single mutable one.
		 */
		template<coordinate T>
		struct slab_frame {
			using W = std::conditional_t<std::floating_point<T>, float64, int128>;

			/** @brief (nx/d, ny/d), d > 0 — exact position (floating: rounded). */
			struct rat {
				W nx, ny, d;
			};

			/**
			 * @brief `a x + b y + c = 0` with `b > 0` for non-verticals, so the
			 * sign of the evaluation at a point is the sign of
			 * (point y - line y).
			 */
			struct line {
				W a, b, c;
			};

			/**
			 * @brief One slab boundary (a distinct vertex x): its y-sorted
			 *        vertex id range and the vertical edges lying on it.
			 */
			struct boundary_data {
				uint32 first_vertex;		   ///< First vertex id, inclusive.
				uint32 last_vertex;			   ///< Last vertex id, inclusive; ids are y-sorted.
				std::vector<uint32> verticals; ///< vertical edges on this boundary, by y
			};

			arrangement const *arr;				   ///< the referenced arrangement (not owned)
			std::vector<line> lines;			   ///< per input segment: its normalized supporting line
			std::vector<rat> rats;				   ///< per vertex: its exact position
			std::vector<uint32> group_of;		   ///< per vertex: its boundary group
			std::vector<boundary_data> boundaries; ///< the distinct-x boundaries, left to right
			/** @brief per boundary group: the non-vertical edges entering / leaving there */
			std::vector<std::vector<uint32>> add_at;	///< Edges entering at each boundary.
			std::vector<std::vector<uint32>> remove_at; ///< Edges leaving at each boundary.

			[[nodiscard]] static int sign_of(W const v) {
				return v > 0 ? 1 : v < 0 ? -1
										 : 0;
			}

			/** @brief Sign of `l*r - a*b`, exact for the integral W. */
			[[nodiscard]] static int product_compare(W const l, W const r, W const a, W const b) {
				if constexpr (std::same_as<W, int128>)
					return atl::detail::compare_products(l, r, a, b);
				else
					return sign_of(l * r - a * b);
			}

			/**
			 * @brief Builds the frame of @p arrangement_in, keeping a pointer to
			 *        it.
			 * @pre @p arrangement_in is `arrangement_of(segments)` and
			 *      outlives the frame; fewer than 2^32 segments and vertices
			 *      (asserted, as is every supporting id being a segment).
			 * @complexity O(n + V + E log(E + 2)), including n supporting input lines.
			 */
			template<segment_range R>
				requires segment_range<R const> && std::same_as<segment_coordinate_t<R>, T>
			slab_frame(arrangement const &arrangement_in, R const &segments)
				: arr(&arrangement_in) {
				atl_assert(uint64(std::ranges::size(segments)) < (uint64(1) << 32));
				atl_assert(uint64(arrangement_in.vertices.size()) < (uint64(1) << 32));
				uint32 const segment_count = uint32(std::ranges::size(segments));
				lines.reserve(segment_count);
				for (uint32 s = 0; s < segment_count; ++s) {
					auto const sg = std::ranges::begin(segments)[s];
					check_query(sg.a);
					check_query(sg.b);
					W a = W(sg.b.y) - W(sg.a.y), b = W(sg.a.x) - W(sg.b.x);
					W c = -(a * W(sg.a.x) + b * W(sg.a.y));
					if (b < 0 || (b == 0 && a < 0)) {
						a = -a;
						b = -b;
						c = -c;
					}
					lines.push_back({a, b, c});
				}
				uint32 const vertex_count = uint32(arr->vertices.size());
				uint32 const edge_count = arr->edge_count();

				// A non-endpoint vertex has incident nonparallel supporting lines.
				// Other integral vertices are endpoints, converted exactly within the bound.
				std::vector<std::array<uint32, 2>> support(
					vertex_count, {arrangement::none, arrangement::none});
				for (uint32 h = 0; h < 2 * edge_count; ++h) {
					uint32 const v = arr->origin[h];
					uint32 const s = arr->supporting[arrangement::edge_of(h)][0];
					atl_assert(s < segment_count); // arr was built from these segments
					auto &sup = support[v];
					if (sup[0] == arrangement::none)
						sup[0] = s;
					else if (sup[1] == arrangement::none && sup[0] != s
							 && lines[sup[0]].a * lines[s].b
									!= lines[s].a * lines[sup[0]].b)
						sup[1] = s;
				}
				rats.reserve(vertex_count);
				for (uint32 v = 0; v < vertex_count; ++v) {
					if (support[v][1] != arrangement::none) {
						auto const &l0 = lines[support[v][0]];
						auto const &l1 = lines[support[v][1]];
						W d = l0.a * l1.b - l1.a * l0.b;
						W nx = l0.b * l1.c - l1.b * l0.c;
						W ny = l0.c * l1.a - l1.c * l0.a;
						if (d < 0) {
							d = -d;
							nx = -nx;
							ny = -ny;
						}
						rats.push_back({nx, ny, d});
					} else {
						auto const p = arr->vertices[v];
						rats.push_back({W(T(p.x)), W(T(p.y)), W(1)});
					}
				}

				// slab boundaries: vertex ids are sweep-ordered, so equal-x
				// runs are consecutive and y-sorted — group them
				group_of.resize(vertex_count);
				for (uint32 v = 0; v < vertex_count; ++v) {
					if (v > 0
						&& product_compare(rats[v - 1].nx, rats[v].d, rats[v].nx,
										   rats[v - 1].d)
							   == 0)
						group_of[v] = group_of[v - 1];
					else {
						group_of[v] = uint32(boundaries.size());
						boundaries.push_back({v, v, {}});
					}
					boundaries[group_of[v]].last_vertex = v;
				}

				// classify edges: verticals join their boundary's catalog,
				// everything else enters at its left group and leaves at its
				// right group
				uint32 const group_count = uint32(boundaries.size());
				add_at.resize(group_count);
				remove_at.resize(group_count);
				for (uint32 e = 0; e < edge_count; ++e) {
					uint32 const gl = group_of[arr->origin[2 * std::size_t(e)]];
					uint32 const gr = group_of[arr->origin[2 * std::size_t(e) + 1]];
					if (gl == gr)
						boundaries[gl].verticals.push_back(e); // y-sorted:
															   // lower id order
					else {
						add_at[gl].push_back(e);
						remove_at[gr].push_back(e);
					}
				}
				for (auto &b : boundaries)
					std::ranges::sort(b.verticals, [&](uint32 const x, uint32 const y) {
						return arr->origin[2 * std::size_t(x)]
							 < arr->origin[2 * std::size_t(y)];
					});
			}

			static void check_query(point<T> const q) {
				if constexpr (std::signed_integral<T> && sizeof(T) == 8) {
					[[maybe_unused]] constexpr T bound = T(1) << 32;
					atl_assert(-bound < q.x && q.x < bound && -bound < q.y && q.y < bound);
				} else if constexpr (std::floating_point<T>) {
					atl_assert(geometry::detail::is_finite(q.x) && geometry::detail::is_finite(q.y));
				} else {
					(void)q;
				}
			}

			/**
			 * @brief Sign of the supporting-line value at the query — positive
			 * means the query is above edge @p e.
			 */
			[[nodiscard]] int line_side(uint32 const e, W const qx, W const qy) const {
				auto const &l = lines[arr->supporting[e][0]];
				return sign_of(l.a * qx + l.b * qy + l.c);
			}

			[[nodiscard]] uint32 left_group(uint32 const e) const {
				return group_of[arr->origin[2 * std::size_t(e)]];
			}

			/**
			 * @brief Strict y-order of two distinct COEXISTING edges — STATELESS:
			 * two edges that are ever alive together keep one relative
			 * order over their whole shared lifetime (a swap would be a
			 * crossing, i.e. a vertex, where both would END), so comparing
			 * in the slab starting at the LATER of their two left
			 * boundaries answers for every slab they share.
			 */
			[[nodiscard]] bool edge_less(uint32 const e1, uint32 const e2) const {
				return edge_less_in(e1, e2, std::max(left_group(e1), left_group(e2)));
			}

			/**
			 * @brief The underlying per-slab order: inside slab @p g the two edges
			 * cannot cross (a crossing is a vertex, hence a boundary), so
			 * the order is decided by where their supporting lines cross
			 * relative to the slab: parallels by offset, otherwise the
			 * slope order flips across the crossing.
			 */
			[[nodiscard]] bool edge_less_in(uint32 const e1, uint32 const e2, uint32 const g) const {
				auto const &l1 = lines[arr->supporting[e1][0]];
				auto const &l2 = lines[arr->supporting[e2][0]];
				W const d = l1.a * l2.b - l2.a * l1.b;
				if (d == 0) return l2.c * l1.b - l1.c * l2.b < 0;
				// crossing x of the two lines vs the slab's left boundary
				W nx = l1.b * l2.c - l2.b * l1.c, dd = d;
				if (dd < 0) {
					dd = -dd;
					nx = -nx;
				}
				rat const &left = rats[boundaries[g].first_vertex];
				if (product_compare(nx, left.d, left.nx, dd) <= 0)
					return d > 0; // slab right of the crossing: slope order
				return d < 0;	  // slab left of it: reversed
			}

			/**
			 * @brief The last boundary with x <= qx, or `arrangement::none` when
			 * the query lies left of everything.
			 */
			[[nodiscard]] uint32 boundary_of(W const qx) const {
				uint32 lo = 0, hi = uint32(boundaries.size());
				while (lo < hi) {
					uint32 const mid = lo + (hi - lo) / 2;
					rat const &r = rats[boundaries[mid].first_vertex];
					if (product_compare(r.nx, W(1), qx, r.d) <= 0)
						lo = mid + 1;
					else
						hi = mid;
				}
				return lo == 0 ? arrangement::none : lo - 1;
			}

			/**
			 * @brief Exact on-boundary resolution when the query x equals boundary
			 * @p g's x: the vertex it coincides with or the vertical edge it
			 * lies on, if any.
			 */
			[[nodiscard]] std::optional<location> on_boundary(uint32 const g, W const qx,
															  W const qy) const {
				auto const &boundary = boundaries[g];
				rat const &bx = rats[boundary.first_vertex];
				if (product_compare(bx.nx, W(1), qx, bx.d) != 0) return {};
				// vertices first (the group's id range is y-sorted)
				uint32 vlo = boundary.first_vertex, vhi = boundary.last_vertex + 1;
				while (vlo < vhi) {
					uint32 const mid = vlo + (vhi - vlo) / 2;
					int const c = product_compare(rats[mid].ny, W(1), qy, rats[mid].d);
					if (c == 0) {
						if (product_compare(rats[mid].nx, W(1), qx, rats[mid].d) == 0)
							return location{location_kind::vertex, mid};
						break; // same y, different x cannot happen in a group
					}
					if (c < 0)
						vlo = mid + 1;
					else
						vhi = mid;
				}
				auto const &vs = boundary.verticals;
				uint32 elo = 0, ehi = uint32(vs.size());
				while (elo < ehi) { // last vertical whose lower end y <= qy
					uint32 const mid = elo + (ehi - elo) / 2;
					rat const &r = rats[arr->origin[2 * std::size_t(vs[mid])]];
					if (product_compare(r.ny, W(1), qy, r.d) <= 0)
						elo = mid + 1;
					else
						ehi = mid;
				}
				if (elo > 0) {
					uint32 const e = vs[elo - 1];
					rat const &top = rats[arr->origin[2 * std::size_t(e) + 1]];
					if (product_compare(qy, top.d, top.ny, W(1)) <= 0)
						return location{location_kind::edge, e};
				}
				return {};
			}
		};
	}

	/**
	 * @brief O(log) point location over an @ref atl::geometry::arrangement — the
	 *        Sarnak-Tarjan PERSISTENCE-SLAB structure. Between consecutive
	 *        distinct vertex x-coordinates the non-vertical edges crossing a
	 *        slab are fixed and totally y-ordered (edges only meet at
	 *        vertices), so one left-to-right pass maintains that order in a
	 *        PATH-COPYING treap (`treap_base` over a persistent arena) and
	 *        keeps one root per slab. A query binary-searches its slab by
	 *        exact rational-x comparison, then descends the slab's version
	 *        comparing the query against each edge through its SUPPORTING
	 *        input segment's line — integral inputs get exact answers,
	 *        including ON-boundary ones: each slab boundary keeps its
	 *        y-sorted vertex range and vertical edges, and orientation zero
	 *        during the descent identifies a point on an edge's interior.
	 *
	 * The face containing an off-boundary point is the region above the
	 * topmost edge below it — `face[2e]`, the same identity the
	 * arrangement's hole nesting uses; no edge below means the unbounded
	 * face. For a MEMORY-LEAN one-shot batch (no persistent arena) see
	 * @ref atl::geometry::locations_of.
	 *
	 * @tparam T Coordinate type shared by supporting segments and queries.
	 * @pre @p arr was built by `arrangement_of(segments)` over exactly the
	 *      @p segments handed to the constructor, and outlives this object
	 *      (it is referenced, not copied — never pass a temporary).
	 * @pre Integral coordinates: `|x|, |y| < 2^32` (the arrangement's own
	 *      bound). Floating inputs run the same code over rounded `float64`
	 *      (best-effort, like the sweep).
	 * @complexity Expected build O(n + V + E log(E + 2)) time and storage,
	 * including n original input segments;
	 * expected query O(log(V + 1) + log(E + 1)); treap height is not a worst-case guarantee.
	 * @note Not constexpr (runtime persistent arena). Updates use allocated paths,
	 * not recursive calls, so stack use is independent of the number of edges.
	 * int32 accepts its full range. Queries have the same int64 bound as segments.
	 */
	template<coordinate T>
	class point_location {
		using frame = detail::slab_frame<T>;
		using W = typename frame::W;

		struct node {
			uint32 left, right, size, edge;
		};

		struct pool {
			uint64 rng;
			atl::detail::persistent_arena<node> nodes;

			pool() : rng(atl::detail::hash_seed() ^ 0x9e3779b97f4a7c15ULL),
					 nodes(node{0, 0, 0, 0}) {}

			uint32 left(uint32 const t) const { return nodes[t].left; }

			uint32 right(uint32 const t) const { return nodes[t].right; }

			uint32 size(uint32 const t) const { return nodes[t].size; }

			uint32 priority(uint32) const { return 0; } // weighted joins only

			uint32 push(uint32 const t) const { return t; }

			uint32 rebind(uint32 const t, uint32 const l, uint32 const r) {
				node nd = nodes[t];
				nd.left = l;
				nd.right = r;
				nd.size = 1 + nodes[l].size + nodes[r].size;
				return nodes.make(nd);
			}

			uint64 rand() { return atl::detail::splitmix64(rng += 0x9e3779b97f4a7c15ULL); }
		};

	public:
		/** @brief A temporary arrangement would dangle: the structure references it. */
		template<segment_range R>
			requires segment_range<R const> && std::same_as<segment_coordinate_t<R>, T>
		point_location(arrangement const &&, R const &) = delete;

		template<segment_range R>
			requires segment_range<R const> && std::same_as<segment_coordinate_t<R>, T>
		/** @brief Builds slab versions; arr must remain unchanged and outlive this object.
		 * @note Not constexpr (runtime persistent arena). */
		point_location(arrangement const &arr, R const &segments)
			: frame_(arr, segments) {
			uint32 const group_count = uint32(frame_.boundaries.size());
			roots_.assign(group_count, 0);
			uint32 root = 0;
			for (uint32 g = 0; g < group_count; ++g) {
				// leaving edges FIRST: they never coexist with the entering
				// ones, so no comparison between the two groups may happen
				for (uint32 const e : frame_.remove_at[g]) root = erased(root, e);
				for (uint32 const e : frame_.add_at[g]) root = inserted(root, e);
				roots_[g] = root;
			}
			atl_assert(root == 0); // every edge leaves before the sweep ends
		}

		/** @brief Classified position of @p q.
		 * @pre Query obeys the constructor coordinate bounds (asserted).
		 * @note Not constexpr (queries runtime slab state). */
		[[nodiscard]] location location_of(point<T> const q) const {
			frame::check_query(q);
			W const qx = W(q.x), qy = W(q.y);
			uint32 const g = frame_.boundary_of(qx);
			if (g == arrangement::none)
				return {location_kind::face, arrangement::unbounded_face};
			if (auto const hit = frame_.on_boundary(g, qx, qy)) return *hit;

			uint32 cur = roots_[g], best = arrangement::none;
			while (cur != 0) {
				uint32 const e = pool_.nodes[cur].edge;
				int const side = frame_.line_side(e, qx, qy);
				if (side == 0) return {location_kind::edge, e};
				if (side > 0) { // q above this edge: try higher
					best = e;
					cur = pool_.right(cur);
				} else
					cur = pool_.left(cur);
			}
			if (best == arrangement::none)
				return {location_kind::face, arrangement::unbounded_face};
			return {location_kind::face, frame_.arr->face[2 * std::size_t(best)]};
		}

	private:
		// Explicit paths keep even an unlucky tree off the call stack; weighted joins reuse the treap coin.
		template<typename GoesLeft>
			requires std::predicate<GoesLeft &, pool &, uint32>
		std::pair<uint32, uint32> split(uint32 root, GoesLeft &&goes_left) {
			path_.clear();
			while (root != 0) {
				bool const joins_left = std::invoke(goes_left, pool_, root);
				path_.emplace_back(root, joins_left);
				root = joins_left ? pool_.right(root) : pool_.left(root);
			}
			uint32 left = 0, right = 0;
			for (auto it = path_.rbegin(); it != path_.rend(); ++it) {
				if (it->second)
					left = pool_.rebind(it->first, pool_.left(it->first), left);
				else
					right = pool_.rebind(it->first, right, pool_.right(it->first));
			}
			return {left, right};
		}

		uint32 merge(uint32 left, uint32 right) {
			path_.clear();
			while (left != 0 && right != 0) {
				uint64 const left_size = pool_.size(left), total = left_size + pool_.size(right);
				bool const takes_left = atl::detail::treap_base::bounded_draw(pool_.rand(), total) < left_size;
				path_.emplace_back(takes_left ? left : right, takes_left);
				if (takes_left)
					left = pool_.right(left);
				else
					right = pool_.left(right);
			}
			uint32 result = left | right;
			for (auto it = path_.rbegin(); it != path_.rend(); ++it) {
				result = it->second ? pool_.rebind(it->first, pool_.left(it->first), result)
									: pool_.rebind(it->first, result, pool_.right(it->first));
			}
			return result;
		}

		uint32 inserted(uint32 const root, uint32 const e) {
			auto goes_left = [&](pool &p, uint32 const t) {
				return frame_.edge_less(p.nodes[t].edge, e);
			};
			auto [below, above] = split(root, goes_left);
			node nd{0, 0, 1, e};
			uint32 const single = pool_.nodes.make(nd);
			return merge(merge(below, single), above);
		}

		uint32 erased(uint32 const root, uint32 const e) {
			auto strictly_below = [&](pool &p, uint32 const t) {
				return frame_.edge_less(p.nodes[t].edge, e);
			};
			auto below_or_target = [&](pool &p, uint32 const t) {
				return p.nodes[t].edge == e || frame_.edge_less(p.nodes[t].edge, e);
			};
			auto [below, rest] =
				split(root, strictly_below);
			auto [target, above] =
				split(rest, below_or_target);
			atl_assert(pool_.size(target) == 1 && pool_.nodes[target].edge == e);
			return merge(below, above);
		}

		frame frame_;
		std::vector<uint32> roots_; // slab [x_g, x_{g+1}) -> treap root
		pool pool_;
		std::vector<std::pair<uint32, bool>> path_;
	};

	/** @brief Deduces the coordinate type from the supporting segments. */
	template<segment_range R>
	point_location(arrangement const &, R const &)
		-> point_location<segment_coordinate_t<R>>;

	/**
	 * @brief OFFLINE batch point location: every query answered in ONE
	 *        mutable status-set sweep over the slab decomposition — no
	 *        persistent arena, so O(n + V + E + Q) memory instead of the online
	 *        structure's expected O(n + V + E log(E + 2)), at the same exactness and the
	 *        same per-query O(log) cost. Results align with @p queries by
	 *        index.
	 *
	 * Queries are bucketed into their slabs by exact rational-x binary
	 * search; the sweep maintains the alive non-vertical edges in a
	 * `std::set` under a STATELESS transparent comparator: coexisting
	 * arrangement edges never swap order, so the pair's shared-lifetime
	 * order — evaluated at the later of their two left boundaries — is a
	 * function of the pair alone. The heterogeneous probe idea is the
	 * Bentley-Ottmann one: the probe is the query point itself, so
	 * `equal_range(probe)` is non-empty exactly when the query lies ON an
	 * edge, and otherwise the element before the range is the topmost edge
	 * below.
	 *
	 * @tparam R Const-readable sized random-access segment range.
	 * @tparam Q Const-readable sized random-access vertex range.
	 * @param arr Unchanged arrangement built from segments.
	 * @param segments Original input segments, in the original order.
	 * @param queries Points to classify.
	 * @return Locations in query input order.
	 * @pre As @ref atl::geometry::point_location . `arr == arrangement_of(segments)`,
	 *      `|x|, |y| < 2^32` for integral inputs.
	 * @note Not constexpr (runtime ordered status set). All sweeps are iterative.
	 * @complexity O(n + (V + E + Q) log(V + E + Q + 2)) time,
	 * O(n + V + E + Q) space, including n original input segments.
	 */
	template<segment_range R, vertex_range Q>
		requires segment_range<R const> && vertex_range<Q const> && std::same_as<segment_coordinate_t<R>, vertex_coordinate_t<Q>>
	[[nodiscard]] std::vector<location> locations_of(arrangement const &arr, R const &segments,
													 Q const &queries) {
		using T = segment_coordinate_t<R>;
		using frame = detail::slab_frame<T>;
		using W = typename frame::W;
		frame const f(arr, segments);
		auto const raw_query_count = std::ranges::size(queries);
		atl_assert(uint64(raw_query_count) < (uint64(1) << 32));
		std::size_t const query_count = std::size_t(raw_query_count);
		std::vector<location> out(query_count,
								  {location_kind::face, arrangement::unbounded_face});
		uint32 const group_count = uint32(f.boundaries.size());
		std::vector<std::vector<uint32>> bucket(group_count);
		for (std::size_t i = 0; i < query_count; ++i) {
			auto const q = std::ranges::begin(queries)[i];
			frame::check_query(q);
			W const qx = W(q.x), qy = W(q.y);
			uint32 const g = f.boundary_of(qx);
			if (g == arrangement::none) continue; // left of everything
			if (auto const hit = f.on_boundary(g, qx, qy))
				out[i] = *hit;
			else
				bucket[g].push_back(uint32(i));
		}

		struct probe {
			W x, y;
		};

		struct status_less {
			frame const *f;
			using is_transparent [[maybe_unused]] = void;

			bool operator()(uint32 const a, uint32 const b) const {
				return a != b && f->edge_less(a, b); // stateless: see edge_less
			}

			bool operator()(uint32 const e, probe const p) const {
				return f->line_side(e, p.x, p.y) > 0; // e strictly below p
			}

			bool operator()(probe const p, uint32 const e) const {
				return f->line_side(e, p.x, p.y) < 0; // e strictly above p
			}
		};

		using status_set = std::set<uint32, status_less>;
		status_set status(status_less{&f});
		// each edge's node, kept from its insertion: erasing through it skips
		// a descent of exact comparisons
		std::vector<typename status_set::iterator> node_of(arr.origin.size() / 2);
		for (uint32 g = 0; g < group_count; ++g) {
			// leaving edges FIRST: they never coexist with the entering ones,
			// so no comparison between the two groups may happen
			for (uint32 const e : f.remove_at[g]) status.erase(node_of[e]);
			for (uint32 const e : f.add_at[g]) node_of[e] = status.insert(e).first;
			for (uint32 const i : bucket[g]) {
				auto const q = std::ranges::begin(queries)[i];
				frame::check_query(q);
				probe const p{W(q.x), W(q.y)};
				auto const [lo, hi] = status.equal_range(p);
				if (lo != hi)
					out[i] = {location_kind::edge, *lo};
				else if (lo != status.begin())
					out[i] = {location_kind::face,
							  arr.face[2 * std::size_t(*std::prev(lo))]};
			}
		}
		return out;
	}
}
