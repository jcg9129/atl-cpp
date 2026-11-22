#pragma once
#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::geometry {
	namespace detail {
		template<typename R>
		inline constexpr bool is_initializer_list = false;

		template<typename Value>
		inline constexpr bool is_initializer_list<std::initializer_list<Value>> = true;

		template<vertex_range R>
		using caliper_storage_t = std::conditional_t<!std::is_lvalue_reference_v<R> && is_initializer_list<std::remove_cvref_t<R>>,
													 std::vector<point<vertex_coordinate_t<R>>>, R>;

		template<vertex_range R, typename Source>
			requires std::constructible_from<R, Source>
		constexpr caliper_storage_t<R> store_caliper_range(Source &&source) {
			if constexpr (!std::same_as<caliper_storage_t<R>, R>)
				return {std::ranges::begin(source), std::ranges::begin(source) + std::ranges::ssize(source)};
			else
				return std::forward<Source>(source);
		}
	}

	/**
	 * @brief One configuration of the rotating calipers on a single convex
	 *        polygon: the caliper line is aligned with the edge
	 *        `edge_start -> edge_end` while the anti-parallel caliper supports
	 *        `opposite`. Indices are positions IN THE POLYGON passed to
	 *        the view.
	 */
	struct antipodal_event {
		vertex_index edge_start; ///< first endpoint of the caliper-aligned edge
		vertex_index edge_end;	 ///< second endpoint of the caliper-aligned edge
		vertex_index opposite;	 ///< support vertex of the anti-parallel caliper

		/** @brief Compares all stored indices. */
		[[nodiscard]] friend constexpr bool operator==(antipodal_event const &, antipodal_event const &) = default;

		/**
		 * @brief The antipodal vertex pairs this configuration certifies (they
		 * share `opposite` — the aligned support line contains both edge
		 * endpoints). Union over all events = ALL antipodal pairs of the
		 * polygon; pairs may repeat across events.
		 */
		[[nodiscard]] constexpr std::array<std::array<vertex_index, 2>, 2> antipodal_pairs() const {
			return {{{edge_start, opposite}, {edge_end, opposite}}};
		}
	};

	/**
	 * @brief Lazy forward range of the @ref atl::geometry::antipodal_event stream of one
	 *        STRICTLY convex CCW polygon (feed @ref atl::geometry::convex_hull output —
	 *        asserted): one event per edge over a full rotation, plus a tie
	 *        event (same edge, `opposite` advanced) whenever the opposite
	 *        edge is exactly parallel. O(h) events, O(1) amortized advance;
	 *        polygons with fewer than 3 vertices yield an empty range.
	 *
	 * Borrow semantics: an lvalue polygon is referenced (keep it alive), an
	 * rvalue is moved into the view; rvalue initializer lists are materialized.
	 * Other non-owning ranges require their backing storage to stay alive. Changes
	 * to polygon coordinates or structure invalidate
	 * the view; moving or destroying the view invalidates its iterators.
	 * @pre Fewer than 2^32 vertices (asserted); coordinate bounds as convex_hull.
	 * @complexity O(h) construction and iteration, O(1) begin, O(1) storage apart from owned inputs.
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	class antipodal_view {
		using T = vertex_coordinate_t<R>;

		detail::caliper_storage_t<R> poly_;
#if defined(_MSC_VER)
		[[msvc::no_unique_address]]
#else
		[[no_unique_address]]
#endif
		Policy policy_;
		uint32 vertex_count_ = 0, initial_support_ = 1;

		constexpr point<T> at(uint32 const i) const { return std::ranges::begin(poly_)[i]; }

		constexpr uint32 next_index(uint32 const i) const { return i + 1 == vertex_count_ ? 0 : i + 1; }

		/** @brief Sign of cross(direction of edge at `e`, direction of edge at `j`). */
		constexpr int edge_cross_sign(uint32 const e, uint32 const j) const {
			return policy_.cross_sign(at(e), at(next_index(e)), at(j), at(next_index(j)));
		}

	public:
		/** @brief References an lvalue polygon or owns an rvalue; checks strict CCW convexity. */
		template<typename Q>
			requires(!std::same_as<std::remove_cvref_t<Q>, antipodal_view>) && std::constructible_from<R, Q>
		constexpr explicit antipodal_view(Q &&poly, Policy const policy = {})
			: poly_(detail::store_caliper_range<R>(std::forward<Q>(poly))), policy_(policy) {
			atl_assert(std::ranges::size(poly_) < (uint64(1) << 32));
			vertex_count_ = uint32(std::ranges::size(poly_));
			atl_assert(vertex_count_ < 3 || [&] {
				auto const c = convexity_of(poly_, policy_);
				return c.is_convex && c.is_strict && c.direction == winding::counterclockwise;
			}());
			if (vertex_count_ >= 3)
				while (edge_cross_sign(0, initial_support_) > 0) initial_support_ = next_index(initial_support_);
		}

		/** @brief Multipass event iterator; its parent view must remain alive and unchanged. */
		class iterator {
			antipodal_view const *view_ = nullptr;
			uint32 edge_start_ = 0, opposite_ = 0;
			bool is_on_tie_event_ = false;

			constexpr void advance_support() { // advance the opposite support for edge edge_start_
				while (view_->edge_cross_sign(edge_start_, opposite_) > 0) opposite_ = view_->next_index(opposite_);
			}

			friend antipodal_view;

			constexpr iterator(antipodal_view const *view, uint32 const edge_start,
							   uint32 const opposite)
				: view_(view), edge_start_(edge_start), opposite_(opposite) {
			}

		public:
			/** @brief Event yielded by dereference. */
			using value_type = antipodal_event;
			/** @brief Signed iterator distance type. */
			using difference_type = std::ptrdiff_t;
			/** @brief Forward iteration category. */
			using iterator_concept = std::forward_iterator_tag;

			/** @brief Creates a singular iterator. */
			constexpr iterator() = default;

			/** @brief Returns the current event; requires a dereferenceable iterator. */
			[[nodiscard]] constexpr antipodal_event operator*() const {
				return {edge_start_, view_->next_index(edge_start_), is_on_tie_event_ ? view_->next_index(opposite_) : opposite_};
			}

			/** @brief Advances to the next event; requires a dereferenceable iterator. */
			constexpr iterator &operator++() {
				if (!is_on_tie_event_ && view_->edge_cross_sign(edge_start_, opposite_) == 0) {
					is_on_tie_event_ = true; // parallel opposite edge: emit it once more
					return *this;
				}
				is_on_tie_event_ = false;
				++edge_start_;
				if (edge_start_ < view_->vertex_count_) advance_support();
				return *this;
			}

			/** @brief Advances and returns the iterator at the current event. */
			constexpr iterator operator++(int) {
				auto copy = *this;
				++*this;
				return copy;
			}

			/** @brief Compares positions within the same stream, or two singular iterators. */
			[[nodiscard]] friend constexpr bool operator==(iterator const &a, iterator const &b) {
				return a.edge_start_ == b.edge_start_ && a.is_on_tie_event_ == b.is_on_tie_event_;
			}
		};

		/** @brief Iterator to the first event, or end for a degenerate stream. */
		[[nodiscard]] constexpr iterator begin() const { return vertex_count_ < 3 ? end() : iterator(this, 0, initial_support_); }

		/** @brief Iterator past the last event. */
		[[nodiscard]] constexpr iterator end() const { return iterator(this, vertex_count_, 0); }
	};

	/** @brief Deduces ownership from the polygon value category. */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	antipodal_view(R &&, Policy = {}) -> antipodal_view<R, Policy>;

	/**
	 * @brief Tag: co-rotate the two calipers in OPPOSED directions (max-distance /
	 *        critical support line problems) instead of parallel (Minkowski,
	 *        merging).
	 */
	struct anti_parallel_t {
		/** @brief Creates the opposed-support tag. */
		explicit anti_parallel_t() = default;
	};

	/** @brief The @ref atl::geometry::anti_parallel_t tag value. */
	inline constexpr anti_parallel_t anti_parallel{};

	/**
	 * @brief One configuration of two co-rotating support lines: the sweep
	 *        direction just aligned with the edge `edge_start -> edge_end` of the
	 *        polygon named by `is_on_second`, while the supports are
	 *        `p_support` on P and `q_support` on Q (each an index in its own
	 *        polygon).
	 */
	struct support_event {
		vertex_index edge_start; ///< first endpoint of the aligned edge, on the polygon named by `is_on_second`
		vertex_index edge_end;	 ///< second endpoint of the aligned edge, on the same polygon
		bool is_on_second;		 ///< which polygon's edge the sweep reached
		vertex_index p_support;	 ///< current support vertex on P
		vertex_index q_support;	 ///< current support vertex on Q

		/** @brief Compares all stored fields. */
		[[nodiscard]] friend constexpr bool operator==(support_event const &, support_event const &) = default;

		/**
		 * @brief The (P index, Q index) support pairs of this configuration. On
		 * exactly parallel P/Q edges the two consecutive events cover the
		 * extremal combinations (the fully crossed combination is not
		 * enumerated — sufficient for max/min folds).
		 */
		[[nodiscard]] constexpr std::array<std::array<vertex_index, 2>, 2> support_pairs() const {
			if (!is_on_second) return {{{edge_start, q_support}, {edge_end, q_support}}};
			return {{{p_support, edge_start}, {p_support, edge_end}}};
		}
	};

	/**
	 * @brief Lazy forward range merging the edge streams of TWO strictly
	 *        convex CCW polygons in sweep order — parallel co-rotation by
	 *        default (the merged edge order IS the Minkowski sum's), opposed
	 *        with the @ref atl::geometry::anti_parallel tag (antipodal pairs BETWEEN the
	 *        polygons: max distance, critical support lines). hP + hQ
	 *        events; single-vertex polygons are legal (no edges, constant
	 *        support), 2-vertex segments too (distinct endpoints, asserted).
	 *        An EMPTY polygon on either side yields an empty range — there
	 *        is no support to pair.
	 * @pre Fewer than 2^32 vertices per input (asserted); coordinate bounds as convex_hull.
	 * @note Lvalues are referenced, rvalues owned; rvalue initializer lists are
	 * materialized. Other non-owning ranges require live backing storage. Changes
	 * to input coordinates or structure, or
	 * moving/destroying the view invalidate iterators. Both supports are constant
	 * for point/point input, which emits no events.
	 * @complexity O(hP + hQ) construction and iteration; O(1) begin and extra storage apart from owned inputs.
	 */
	template<vertex_range RP, vertex_range RQ, sign_policy<vertex_coordinate_t<RP>> Policy = exact>
		requires std::same_as<vertex_coordinate_t<RP>, vertex_coordinate_t<RQ>>
	class dual_support_view {
		using T = vertex_coordinate_t<RP>;

		detail::caliper_storage_t<RP> p_;
		detail::caliper_storage_t<RQ> q_;
#if defined(_MSC_VER)
		[[msvc::no_unique_address]]
#else
		[[no_unique_address]]
#endif
		Policy policy_;
		uint32 p_vertex_count_ = 0, q_vertex_count_ = 0, p_start_ = 0, q_start_ = 0;
		bool is_anti_parallel_ = false;

		constexpr point<T> p_at(uint32 const i) const { return std::ranges::begin(p_)[i]; }

		constexpr point<T> q_at(uint32 const i) const { return std::ranges::begin(q_)[i]; }

		constexpr uint32 p_next_index(uint32 const i) const { return i + 1 == p_vertex_count_ ? 0 : i + 1; }

		constexpr uint32 q_next_index(uint32 const i) const { return i + 1 == q_vertex_count_ ? 0 : i + 1; }

		/**
		 * @brief The (y, x)-extreme vertex of @p poly — the support where the
		 * sweep begins: the minimum for the parallel co-rotation, the
		 * maximum for a polygon rotated in opposition.
		 */
		template<vertex_range Poly>
			requires std::same_as<vertex_coordinate_t<Poly>, T>
		constexpr vertex_index start_vertex(Poly const &poly, bool const picks_maximum) const {
			auto const vertex = std::ranges::begin(poly);
			vertex_index extreme = 0;
			for (vertex_index candidate = 1; candidate < uint32(std::ranges::size(poly));
				 ++candidate) {
				point<T> const c = vertex[candidate], e = vertex[extreme];
				bool const is_below = c.y != e.y ? c.y < e.y : c.x < e.x;
				if (is_below != picks_maximum) extreme = candidate;
			}
			return extreme;
		}

		constexpr void init() {
			atl_assert(std::ranges::size(p_) < (uint64(1) << 32));
			atl_assert(std::ranges::size(q_) < (uint64(1) << 32));
			p_vertex_count_ = uint32(std::ranges::size(p_));
			q_vertex_count_ = uint32(std::ranges::size(q_));
			atl_assert(p_vertex_count_ < 3 || [&] {
				auto const c = convexity_of(p_, policy_);
				return c.is_convex && c.is_strict && c.direction == winding::counterclockwise;
			}());
			atl_assert(q_vertex_count_ < 3 || [&] {
				auto const c = convexity_of(q_, policy_);
				return c.is_convex && c.is_strict && c.direction == winding::counterclockwise;
			}());
			// a 2-vertex "polygon" is a segment: endpoints must differ or
			// its edges are zero vectors with no direction
			atl_assert(p_vertex_count_ != 2 || p_at(0) != p_at(1));
			atl_assert(q_vertex_count_ != 2 || q_at(0) != q_at(1));
			p_start_ = start_vertex(p_, false);
			// opposed rotation supports the OPPOSITE extreme of Q
			q_start_ = start_vertex(q_, is_anti_parallel_);
		}

	public:
		/** @brief Stores both polygons and initializes the requested support rotation. */
		template<typename SP, typename SQ>
			requires std::constructible_from<RP, SP> && std::constructible_from<RQ, SQ>
		constexpr dual_support_view(SP &&p, SQ &&q, Policy const policy = {})
			: p_(detail::store_caliper_range<RP>(std::forward<SP>(p))), q_(detail::store_caliper_range<RQ>(std::forward<SQ>(q))), policy_(policy) {
			init();
		}

		/** @brief Stores both polygons and initializes the requested support rotation. */
		template<typename SP, typename SQ>
			requires std::constructible_from<RP, SP> && std::constructible_from<RQ, SQ>
		constexpr dual_support_view(SP &&p, SQ &&q, anti_parallel_t, Policy const policy = {})
			: p_(detail::store_caliper_range<RP>(std::forward<SP>(p))), q_(detail::store_caliper_range<RQ>(std::forward<SQ>(q))), policy_(policy), is_anti_parallel_(true) {
			init();
		}

		/** @brief Multipass event iterator; its parent view must remain alive and unchanged. */
		class iterator {
			dual_support_view const *view_ = nullptr;
			uint32 p_edges_taken_ = 0, q_edges_taken_ = 0, p_support_ = 0, q_support_ = 0;

			friend dual_support_view;

			constexpr iterator(dual_support_view const *view, uint32 const p_edges_taken,
							   uint32 const q_edges_taken)
				: view_(view), p_edges_taken_(p_edges_taken), q_edges_taken_(q_edges_taken) {
				if (view_ != nullptr) {
					p_support_ = view_->p_start_;
					q_support_ = view_->q_start_;
				}
			}

			/**
			 * @brief Does the next step consume P's edge? Q's candidate edge is
			 * reversed under opposed rotation; exhausted sides fall through.
			 */
			constexpr bool takes_p() const {
				if (p_edges_taken_ == view_->p_vertex_count_ || view_->p_vertex_count_ < 2) return false;
				if (q_edges_taken_ == view_->q_vertex_count_ || view_->q_vertex_count_ < 2) return true;
				auto const pa = view_->p_at(p_support_), pb = view_->p_at(view_->p_next_index(p_support_));
				auto qa = view_->q_at(q_support_), qb = view_->q_at(view_->q_next_index(q_support_));
				if (view_->is_anti_parallel_) std::swap(qa, qb);			  // negated direction
				return !detail::sweep_before(qa, qb, pa, pb, view_->policy_); // ties: P first
			}

		public:
			/** @brief Event yielded by dereference. */
			using value_type = support_event;
			/** @brief Signed iterator distance type. */
			using difference_type = std::ptrdiff_t;
			/** @brief Forward iteration category. */
			using iterator_concept = std::forward_iterator_tag;

			/** @brief Creates a singular iterator. */
			constexpr iterator() = default;

			/** @brief Returns the current event; requires a dereferenceable iterator. */
			[[nodiscard]] constexpr support_event operator*() const {
				if (takes_p()) return {p_support_, view_->p_next_index(p_support_), false, p_support_, q_support_};
				return {q_support_, view_->q_next_index(q_support_), true, p_support_, q_support_};
			}

			/** @brief Advances to the next event; requires a dereferenceable iterator. */
			constexpr iterator &operator++() {
				if (takes_p()) {
					p_support_ = view_->p_next_index(p_support_);
					++p_edges_taken_;
				} else {
					q_support_ = view_->q_next_index(q_support_);
					++q_edges_taken_;
				}
				return *this;
			}

			/** @brief Advances and returns the iterator at the current event. */
			constexpr iterator operator++(int) {
				auto copy = *this;
				++*this;
				return copy;
			}

			/** @brief Compares positions within the same stream, or two singular iterators. */
			[[nodiscard]] friend constexpr bool operator==(iterator const &a, iterator const &b) {
				return a.p_edges_taken_ == b.p_edges_taken_ && a.q_edges_taken_ == b.q_edges_taken_;
			}
		};

		/** @brief Iterator to the first event, or end for a degenerate stream. */
		[[nodiscard]] constexpr iterator begin() const {
			// an empty side has no support at all: nothing to pair
			if (p_vertex_count_ == 0 || q_vertex_count_ == 0) return end();
			return iterator(this, 0, 0);
		}

		/** @brief Iterator past the last event. */
		[[nodiscard]] constexpr iterator end() const {
			return iterator(this, p_vertex_count_ < 2 ? 0 : p_vertex_count_, q_vertex_count_ < 2 ? 0 : q_vertex_count_);
		}
	};

	/** @brief Deduces ownership independently for each polygon. */
	template<vertex_range RP, vertex_range RQ, sign_policy<vertex_coordinate_t<RP>> Policy = exact>
	dual_support_view(RP &&, RQ &&, Policy = {}) -> dual_support_view<RP, RQ, Policy>;

	/** @brief Deduces ownership independently for each polygon. */
	template<vertex_range RP, vertex_range RQ, sign_policy<vertex_coordinate_t<RP>> Policy = exact>
	dual_support_view(RP &&, RQ &&, anti_parallel_t, Policy = {}) -> dual_support_view<RP, RQ, Policy>;

	/**
	 * @brief Indices (into @p points) of a pair at maximum distance — the
	 *        diameter pair. Hull + one antipodal sweep; exact selection for
	 *        integral coordinates (rounded squared-distance comparisons for
	 *        `double`). Ties resolve arbitrarily; the indices are DISTINCT
	 *        whenever `size(points) >= 2` (a fully duplicated input returns
	 *        two distinct indices at distance 0) — only a single-point input
	 *        pairs the point with itself.
	 * @param points Nonempty input point cloud; not modified.
	 * @param p Predicate policy.
	 * @pre `size(points) >= 1` (asserted); for `int64`: `|x|, |y| < 2^62`.
	 * @complexity O(n log n) time and O(n) extra memory, including hull construction.
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] constexpr std::array<vertex_index, 2> farthest_pair_indices(R &&points, Policy const p = {}) {
		using T = vertex_coordinate_t<R>;
		atl_assert(std::ranges::size(points) >= 1);
		auto const ids = convex_hull_indices(points, p);
		auto const first = std::ranges::begin(points);
		if (ids.size() == 1) {
			// one hull vertex = every point equal: any two elements are an
			// optimal pair; self-pair only when there is just one element
			if (std::ranges::size(points) == 1) return {ids[0], ids[0]};
			return {ids[0], ids[0] == 0 ? vertex_index(1) : vertex_index(0)};
		}
		if (ids.size() == 2) return {ids[0], ids[1]};
		std::vector<point<T>> hull;
		hull.reserve(ids.size());
		for (vertex_index const i : ids) hull.push_back(first[i]);
		std::array<vertex_index, 2> best{ids[0], ids[0]};
		difference_product_t<T> best_distance{};
		for (auto const event : antipodal_view(hull, p))
			for (auto const [i, j] : event.antipodal_pairs()) {
				auto const d = squared_distance(hull[i], hull[j]);
				if (d > best_distance) {
					best_distance = d;
					best = {ids[i], ids[j]};
				}
			}
		return best;
	}

	/** @brief @ref atl::geometry::farthest_pair_indices materialized as points. */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] constexpr std::array<point<vertex_coordinate_t<R>>, 2> farthest_pair(R &&points,
																					   Policy const p = {}) {
		auto const [i, j] = farthest_pair_indices(points, p);
		auto const first = std::ranges::begin(points);
		return {first[i], first[j]};
	}

	/**
	 * @brief Indices of a farthest pair BETWEEN two point sets — one index
	 *        into each input. The maximum distance between two sets is
	 *        always realized at hull vertices, so raw point sets lose
	 *        nothing: both are hulled internally and the opposed dual sweep
	 *        folds the support pairs. Exact selection for integral
	 *        coordinates; ties resolve arbitrarily.
	 * @param a_points First nonempty input cloud; not modified.
	 * @param b_points Second nonempty input cloud; not modified.
	 * @param p Predicate policy.
	 * @pre Both inputs non-empty (asserted); for `int64`: `|x|, |y| < 2^62`.
	 * @complexity O((n + m) log(n + m)) time and O(n + m) extra memory.
	 */
	template<vertex_range RA, vertex_range RB,
			 sign_policy<vertex_coordinate_t<RA>> Policy = exact>
		requires std::same_as<vertex_coordinate_t<RA>, vertex_coordinate_t<RB>>
	[[nodiscard]] constexpr std::array<vertex_index, 2> farthest_pair_between_indices(RA &&a_points,
																					  RB &&b_points,
																					  Policy const p = {}) {
		using T = vertex_coordinate_t<RA>;
		atl_assert(std::ranges::size(a_points) >= 1 && std::ranges::size(b_points) >= 1);
		auto const a_ids = convex_hull_indices(a_points, p);
		auto const b_ids = convex_hull_indices(b_points, p);
		auto const a_first = std::ranges::begin(a_points);
		auto const b_first = std::ranges::begin(b_points);
		std::vector<point<T>> a_hull, b_hull;
		for (vertex_index const i : a_ids) a_hull.push_back(a_first[i]);
		for (vertex_index const i : b_ids) b_hull.push_back(b_first[i]);
		std::array<vertex_index, 2> best{a_ids[0], b_ids[0]};
		auto best_distance = squared_distance(a_hull[0], b_hull[0]);
		for (auto const event : dual_support_view(a_hull, b_hull, anti_parallel, p))
			for (auto const [i, j] : event.support_pairs()) {
				auto const d = squared_distance(a_hull[i], b_hull[j]);
				if (d > best_distance) {
					best_distance = d;
					best = {a_ids[i], b_ids[j]};
				}
			}
		return best;
	}

	/** @brief @ref atl::geometry::farthest_pair_between_indices materialized as points. */
	template<vertex_range RA, vertex_range RB,
			 sign_policy<vertex_coordinate_t<RA>> Policy = exact>
		requires std::same_as<vertex_coordinate_t<RA>, vertex_coordinate_t<RB>>
	[[nodiscard]] constexpr std::array<point<vertex_coordinate_t<RA>>, 2>
	farthest_pair_between(RA &&a_points, RB &&b_points, Policy const p = {}) {
		auto const [i, j] = farthest_pair_between_indices(a_points, b_points, p);
		return {std::ranges::begin(a_points)[i], std::ranges::begin(b_points)[j]};
	}

	namespace detail {
		/**
		 * @brief The Minkowski difference `A - B` of two hulled point sets: the
		 * boundary vertices from the parallel dual sweep, each with the
		 * supporting vertex indices into `a_hull` / `b_hull` (the hull of
		 * the NEGATED second set) that produced it.
		 */
		template<coordinate T>
		struct minkowski_difference {
			std::vector<point<T>> a_hull, negated_b_hull;
			std::vector<point<T>> boundary;
			std::vector<std::array<vertex_index, 2>> supports;
		};

		/**
		 * @brief Hulls both point sets (the second negated) and walks the
		 *        parallel dual sweep to produce their Minkowski difference.
		 * @pre Both inputs non-empty (asserted).
		 */
		template<vertex_range RA, vertex_range RB, sign_policy<vertex_coordinate_t<RA>> Policy>
			requires std::same_as<vertex_coordinate_t<RA>, vertex_coordinate_t<RB>>
		[[nodiscard]] minkowski_difference<vertex_coordinate_t<RA>> difference_of(RA &&a_points, RB &&b_points,
																				  Policy const p) {
			using T = vertex_coordinate_t<RA>;
			atl_assert(std::ranges::size(a_points) >= 1 && std::ranges::size(b_points) >= 1);
			if constexpr (std::signed_integral<T> && sizeof(T) == 8) {
				[[maybe_unused]] auto const fits = [](point<T> const q) {
					constexpr T bound = T(1) << 61;
					return -bound < q.x && q.x < bound && -bound < q.y && q.y < bound;
				};
				atl_assert(std::ranges::all_of(a_points, fits) && std::ranges::all_of(b_points, fits));
			}
			minkowski_difference<T> d;
			d.a_hull = convex_hull(a_points, p);
			std::vector<point<T>> b_negated;
			for (auto const q : std::ranges::subrange(std::ranges::begin(b_points),
													  std::ranges::end(b_points)))
				b_negated.push_back(-point<T>(q));
			d.negated_b_hull = convex_hull(b_negated, p);
			for (auto const event : dual_support_view(d.a_hull, d.negated_b_hull, p)) {
				d.boundary.push_back(d.a_hull[event.p_support]
									 + d.negated_b_hull[event.q_support]);
				d.supports.push_back({event.p_support, event.q_support});
			}
			if (d.boundary.empty()) { // point vs point
				d.boundary.push_back(d.a_hull[0] + d.negated_b_hull[0]);
				d.supports.push_back({0, 0});
			}
			return d;
		}
	}

	/**
	 * @brief Distance between the convex hulls of two point sets, as filled
	 *        regions: 0 when they overlap or touch, else the smallest
	 *        distance between their boundaries (a `double` construction).
	 *        NOT the closest pair between the sets — interior points are
	 *        irrelevant here. Implementation: the distance between two
	 *        convex sets is the distance from the origin to their Minkowski
	 *        difference, whose boundary the parallel dual sweep produces
	 *        directly; integral overlap is an exact `contains` decision. Floating
	 * differences are rounded in T, so their overlap classification and nonzero
	 * distances are construction-grade, including under the exact policy.
	 * @param a_points First nonempty input cloud; not modified.
	 * @param b_points Second nonempty input cloud; not modified.
	 * @param p Predicate policy.
	 * @pre Both inputs non-empty (asserted); for `int64`: `|x|, |y| < 2^61`
	 *      (the difference doubles coordinate magnitudes).
	 * @note Not constexpr (uses runtime segment distances and square roots).
	 * @pre Negation and coordinate differences of the input sets must fit T.
	 * @complexity O((n + m) log(n + m)) time and O(n + m) extra memory.
	 */
	template<vertex_range RA, vertex_range RB,
			 sign_policy<vertex_coordinate_t<RA>> Policy = exact>
		requires std::same_as<vertex_coordinate_t<RA>, vertex_coordinate_t<RB>>
	[[nodiscard]] float64 distance_between_convex_hulls(RA &&a_points, RB &&b_points, Policy const p = {}) {
		using T = vertex_coordinate_t<RA>;
		auto const d = detail::difference_of(a_points, b_points, p);
		point<T> const origin{};
		if (d.boundary.size() >= 3) {
			if (contains(d.boundary, origin, p) != containment::outside) return 0.0;
		}
		float64 best = std::numeric_limits<float64>::infinity();
		for (std::size_t i = 0, j = d.boundary.size() - 1; i < d.boundary.size(); j = i++)
			best = (std::min)(best, distance(segment<T>{d.boundary[j], d.boundary[i]}, origin));
		return best;
	}

	namespace detail {
		/**
		 * @brief Both minimal enclosing rectangles in ONE flush-edge sweep (a
		 * minimal rectangle has a side on some hull edge): per edge, three
		 * cyclically monotone support pointers — farthest/nearest along the
		 * edge direction and farthest above it — advanced by exact policy
		 * signs; objectives and corners are `double` constructions.
		 */
		template<coordinate T, sign_policy<T> Policy>
		struct rectangle_sweep {
			std::array<point<float64>, 4> min_area_corners{}, min_perimeter_corners{};

			rectangle_sweep(std::vector<point<T>> const &hull, Policy const p) {
				uint32 const h = uint32(hull.size());
				auto const at = [&](uint32 i) { return hull[i >= h ? i - h : i]; };
				if (h == 1) {
					point<float64> const only = to_float64(hull[0]);
					min_area_corners = min_perimeter_corners = {only, only, only, only};
					return;
				}
				auto const next_index = [&](uint32 i) { return i + 1 == h ? 0 : i + 1; };
				auto const edge_dot_sign = [&](uint32 e, uint32 from, uint32 to) {
					return dot_sign(at(e), at(e + 1), at(from), at(to), p);
				};
				auto const edge_cross_sign = [&](uint32 e, uint32 from, uint32 to) {
					return p.cross_sign(at(e), at(e + 1), at(from), at(to));
				};
				uint32 far_along = 0, near_along = 0, top = 0;
				for (uint32 v = 1; v < h; ++v) {
					if (edge_dot_sign(0, far_along, v) > 0) far_along = v;
					if (edge_dot_sign(0, near_along, v) < 0) near_along = v;
					if (edge_cross_sign(0, top, v) > 0) top = v;
				}
				float64 best_area = std::numeric_limits<float64>::infinity();
				float64 best_perimeter = best_area;
				for (uint32 e = 0; e < h; ++e) {
					if (e > 0) {
						while (edge_dot_sign(e, far_along, next_index(far_along)) > 0)
							far_along = next_index(far_along);
						while (edge_dot_sign(e, near_along, next_index(near_along)) < 0)
							near_along = next_index(near_along);
						while (edge_cross_sign(e, top, next_index(top)) > 0) top = next_index(top);
					}
					point<float64> const a = to_float64(at(e));
					// offsets from the edge's start, differenced exactly: a
					// short edge far from the origin keeps its direction
					point<float64> const u = float64_difference(at(e), at(e + 1));
					float64 const uu = u.x * u.x + u.y * u.y;
					auto const along = [&](uint32 v) {
						auto const w = float64_difference(at(e), at(v));
						return w.x * u.x + w.y * u.y;
					};
					float64 const lo = along(near_along), hi = along(far_along);
					auto const to_top = float64_difference(at(e), at(top));
					float64 const height2 = u.x * to_top.y - u.y * to_top.x; // height * |u|
					float64 const area = (hi - lo) * height2 / uu;
					float64 const perimeter = 2 * ((hi - lo) + height2) / std::sqrt(uu);
					auto const corners = [&] {
						point<float64> const base_low{a.x + (lo / uu) * u.x, a.y + (lo / uu) * u.y};
						point<float64> const base_high{a.x + (hi / uu) * u.x,
													   a.y + (hi / uu) * u.y};
						point<float64> const up{-(height2 / uu) * u.y, (height2 / uu) * u.x};
						return std::array<point<float64>, 4>{base_low, base_high, base_high + up,
															 base_low + up};
					};
					if (area < best_area) {
						best_area = area;
						min_area_corners = corners();
					}
					if (perimeter < best_perimeter) {
						best_perimeter = perimeter;
						min_perimeter_corners = corners();
					}
				}
			}
		};
	}

	/**
	 * @note Not constexpr (compares objectives using runtime std::sqrt).
	 * @brief The minimum-AREA rectangle containing every point, as its 4
	 *        corners in CCW order — `double` constructions (the first corner
	 *        sits at the low-projection end of the flush hull edge); area
	 *        and perimeter are one cross()/distance() away. Flush-side
	 *        theorem: some side lies on a hull edge, so one O(h)
	 *        three-pointer sweep after the hull. Degenerate input gives a
	 *        degenerate rectangle (a point: four equal corners; collinear
	 *        input: a zero-height strip).
	 * @return Four constructed corners; tied objectives resolve arbitrarily.
	 * @pre Floating projection products and objective intermediates must be finite;
	 * nonzero squared edge lengths must be finite and nonzero in float64.
	 * @param points Nonempty input point cloud; not modified.
	 * @param p Predicate policy.
	 * @pre `size(points) >= 1` (asserted); for `int64`: `|x|, |y| < 2^62`.
	 * @complexity O(n log n) time and O(n) extra memory, including hull construction.
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] std::array<point<float64>, 4> min_area_enclosing_rectangle(R &&points, Policy const p = {}) {
		atl_assert(std::ranges::size(points) >= 1);
		auto const hull = convex_hull(points, p);
		return detail::rectangle_sweep(hull, p).min_area_corners;
	}

	/**
	 * @note Not constexpr (compares objectives using runtime std::sqrt).
	 * @brief The minimum-PERIMETER rectangle containing every point — a different
	 * rectangle than the minimum-area one in general; same conventions as
	 * @ref atl::geometry::min_area_enclosing_rectangle.
	 * @param points Nonempty input cloud; not modified.
	 * @param p Predicate policy.
	 * @return Four CCW corners of a minimum-perimeter enclosing rectangle.
	 * @pre The coordinate and construction bounds of min_area_enclosing_rectangle.
	 * @complexity O(n log n) time and O(n) auxiliary memory.
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] std::array<point<float64>, 4> min_perimeter_enclosing_rectangle(R &&points, Policy const p = {}) {
		atl_assert(std::ranges::size(points) >= 1);
		auto const hull = convex_hull(points, p);
		return detail::rectangle_sweep(hull, p).min_perimeter_corners;
	}

	/**
	 * @brief A strip between two parallel lines, described by three points:
	 *        one line through `edge_start` and `edge_end`, the other
	 *        through `opposite`. The result of @ref atl::geometry::narrowest_strip.
	 */
	template<coordinate T>
	struct strip {
		point<T> edge_start; ///< first point of the boundary line through the edge
		point<T> edge_end;	 ///< second point of that boundary line
		point<T> opposite;	 ///< the other boundary passes here, parallel

		/** @brief Compares the three defining points. */
		[[nodiscard]] friend constexpr bool operator==(strip const &, strip const &) = default;
	};

	/**
	 * @brief The narrowest strip (region between two parallel lines)
	 *        containing every point — the "width of a point set" of the
	 *        computational-geometry literature, as the OBJECT rather than
	 *        the number: one boundary line passes through the hull edge
	 *        `edge_start -> edge_end` (the flush side), the other through
	 *        `opposite`, parallel to it. All three are input points, so
	 *        the description is exact; the width is a caller one-liner:
	 *        `cross(edge_start, edge_end, opposite) / distance(edge_start,
	 *        edge_end)` (non-negative: `opposite` lies left of the edge).
	 *        Selection among edges compares rounded widths (construction
	 *        grade). Degenerate input collapses the strip: a single point
	 *        gives three equal points, collinear input a zero-width strip
	 *        along the line.
	 *
	 * @note Not constexpr (the width comparison uses std::sqrt, unavailable in
	 * constant evaluation on some supported compilers).
	 * @param points Nonempty input point cloud; not modified.
	 * @param p Predicate policy.
	 * @pre `size(points) >= 1` (asserted); for `int64`: `|x|, |y| < 2^62`.
	 * @complexity O(n log n) time and O(n) extra memory, including hull construction.
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] strip<vertex_coordinate_t<R>> narrowest_strip(R &&points, Policy const p = {}) {
		using T = vertex_coordinate_t<R>;
		atl_assert(std::ranges::size(points) >= 1);
		auto const hull = convex_hull(points, p);
		if (hull.size() == 1) return {hull[0], hull[0], hull[0]};
		if (hull.size() == 2) return {hull[0], hull[1], hull[0]};
		strip<T> best{};
		float64 best_width = std::numeric_limits<float64>::infinity();
		for (auto const event : antipodal_view(hull, p)) {
			float64 const doubled_area =
				float64(cross(hull[event.edge_start], hull[event.edge_end], hull[event.opposite]));
			float64 const width =
				doubled_area / distance(hull[event.edge_start], hull[event.edge_end]);
			if (width < best_width) {
				best_width = width;
				best = {hull[event.edge_start], hull[event.edge_end], hull[event.opposite]};
			}
		}
		return best;
	}
}
