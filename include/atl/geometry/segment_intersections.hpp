#pragma once
#include <atl/geometry/point.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/detail/int256.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <map>
#include <ranges>
#include <set>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::geometry {
	/**
	 * @brief Position of a segment within a sweep's input range (positional, the
	 * `vertex_index` pattern).
	 */
	using segment_index = uint32;

	/** @brief A sized random-access range of `segment<T>`. */
	template<typename R>
	concept segment_range =
		std::ranges::random_access_range<R> && std::ranges::sized_range<R>
		&& requires {
			   typename std::ranges::range_value_t<R>::coordinate_type;
			   requires std::same_as<
				   std::ranges::range_value_t<R>,
				   segment<typename std::ranges::range_value_t<R>::coordinate_type>>;
		   };

	/** @brief Coordinate type of a segment range. */
	template<segment_range R>
	using segment_coordinate_t = typename std::ranges::range_value_t<R>::coordinate_type;

	/**
	 * @brief One reported common point of two or more segments, with the de
	 *        Berg partition of the segments through it. The spans point into
	 *        the view's internal storage — valid until the iterator
	 *        advances. A degenerate (point) segment at `p` appears in BOTH
	 *        `starting` and `ending` (p is its first and last endpoint).
	 */
	struct intersection_event {
		/** @brief `below` when no segment is under the event point. */
		static constexpr segment_index no_below = ~segment_index(0);

		point<float64> at;						 ///< the common point (exact when it is an
												 ///< input endpoint — coordinates < 2^32 < 2^53)
		std::span<segment_index const> starting; ///< U(p): p is their sweep-first endpoint
		std::span<segment_index const> ending;	 ///< L(p): p is their sweep-last endpoint
		std::span<segment_index const> passing;	 ///< C(p): p is interior to them
		segment_index below;					 ///< the segment strictly below the event point
												 ///< (highest such, by the status order), or
												 ///< @ref atl::geometry::intersection_event::no_below — the hook planar-subdivision
												 ///< consumers need to nest components into faces
	};

	namespace detail {
		/**
		 * @brief The sweep engine: Bentley-Ottmann in the de Berg et al.
		 *        event-point formulation. Integral coordinates: every
		 *        ordering decision is EXACT — event points are unreduced
		 *        rationals `(nx/d, ny/d)` compared through int256 products;
		 *        status order evaluates lines at rational points the same
		 *        way. Floating coordinates run the same code over rounded
		 *        `float64` (best-effort ordering, documented).
		 *
		 * Status structure: `std::set` with a STATEFUL TRANSPARENT
		 * comparator — segments order by y at the current event's x (ties by
		 * slope, direction flipped before/after the event; verticals count
		 * as y = p.y while alive and slope +infinity; index as final tie).
		 * The event point doubles as a heterogeneous probe key, so the
		 * segments through p are exactly `equal_range(probe)` — both when
		 * erasing L(p) u C(p) and after reinserting C(p) u U(p) — and the
		 * neighbor checks are the elements flanking that range.
		 */
		template<coordinate T>
		class segment_sweep {
		public:
			/** @brief Working scalar: `int128` for integral coordinates (exact), `float64` otherwise. */
			using W = std::conditional_t<std::floating_point<T>, float64, int128>;

			/** @brief An event point as the unreduced rational `(nx/d, ny/d)`, `d > 0`. */
			struct sweep_point {
				W nx{};
				W ny{};
				W d{1};

				/** @brief Sign of `a*b - c*e` — exact through int256 products for `int128`. */
				[[nodiscard]] static int scaled_compare(W const a, W const b, W const c, W const e) {
					if constexpr (std::same_as<W, int128>)
						return atl::detail::compare_products(a, b, c, e);
					else {
						float64 const v = a * b - c * e;
						return v > 0 ? 1 : v < 0 ? -1
												 : 0;
					}
				}

				/** @brief Lexicographic `(x, then y)` order of the rational points — the sweep order. */
				[[nodiscard]] friend bool operator<(sweep_point const &l, sweep_point const &r) {
					int const by_x = scaled_compare(l.nx, r.d, r.nx, l.d);
					if (by_x != 0) return by_x < 0;
					return scaled_compare(l.ny, r.d, r.ny, l.d) < 0;
				}

				/** @brief The point as `double`s: exact for input endpoints (`d == 1`), rounded otherwise. */
				[[nodiscard]] point<float64> rounded() const {
					return {float64(nx) / float64(d), float64(ny) / float64(d)};
				}
			};

			/**
			 * @brief Starts a sweep, optionally including singleton endpoints.
			 * @param input Closed input segments.
			 * @param reports_singletons Also report events where only ONE
			 *        segment is present (plain endpoints, lone point
			 *        segments) — every arrangement vertex is then an event.
			 *        The intersection views keep the default (off).
			 */
			template<segment_range Segments>
				requires std::same_as<segment_coordinate_t<Segments>, T>
			explicit segment_sweep(Segments &&input, bool const reports_singletons = false)
				: segment_sweep(copy_segments(input), reports_singletons) {}

			/**
			 * @brief The sink form: an owned segment vector moves in instead of
			 * being copied (preferred over the template for any
			 * `std::vector<segment<T>>` argument).
			 */
			explicit segment_sweep(std::vector<segment<T>> segments, bool const reports_singletons = false)
				: segments_(std::move(segments)), reports_singletons_(reports_singletons),
				  status_(status_less{this}) {
				atl_assert(segments_.size() < (uint64(1) << 32));
				if constexpr (std::signed_integral<T>) {
					static_assert(sizeof(T) <= 8);
					if constexpr (sizeof(T) == 8) {
						[[maybe_unused]] constexpr T bound = T(1) << 32;
						for ([[maybe_unused]] auto const s : segments_) {
							atl_assert(-bound < s.a.x && s.a.x < bound);
							atl_assert(-bound < s.a.y && s.a.y < bound);
							atl_assert(-bound < s.b.x && s.b.x < bound);
							atl_assert(-bound < s.b.y && s.b.y < bound);
						}
					}
				}
				if constexpr (std::floating_point<T>) {
					for ([[maybe_unused]] auto const &s : segments_) {
						atl_assert(detail::is_finite(s.a.x) && detail::is_finite(s.a.y));
						atl_assert(detail::is_finite(s.b.x) && detail::is_finite(s.b.y));
					}
				}
				for (segment_index i = 0; i < segment_index(segments_.size()); ++i) {
					auto const [first, last] = ends_of(i);
					queue_[first].push_back(i);
					queue_[last]; // ensure the removal event exists
				}
			}

			// status_'s comparator holds `this`, and a std::set's comparator
			// cannot be re-pointed after a copy or move
			segment_sweep(segment_sweep const &) = delete;
			segment_sweep &operator=(segment_sweep const &) = delete;

			/**
			 * @brief Advance to the next event with >= 2 segments through it;
			 * false when the sweep is exhausted.
			 */
			bool advance() {
				while (!queue_.empty()) {
					auto node = queue_.extract(queue_.begin());
					current_ = node.key();
					is_after_event_ = false;
					starting_ = std::move(node.mapped());
					if (handle_event()) return true;
				}
				return false;
			}

			/** @brief The event `advance` stopped at; its spans point into this engine. */
			[[nodiscard]] intersection_event current_event() const {
				return {current_.rounded(),
						{starting_.data(), starting_.size()},
						{ending_.data(), ending_.size()},
						{passing_.data(), passing_.size()},
						below_};
			}

		private:
			template<segment_range Segments>
				requires std::same_as<segment_coordinate_t<Segments>, T>
			[[nodiscard]] static std::vector<segment<T>> copy_segments(Segments &&input) {
				atl_assert(std::ranges::size(input) < (uint64(1) << 32));
				std::vector<segment<T>> result;
				result.reserve(std::size_t(std::ranges::size(input)));
				for (auto const value : input) result.push_back(value);
				return result;
			}

			[[nodiscard]] std::pair<point<T>, point<T>> sweep_order_ends(segment_index const si) const {
				auto const s = segments_[si];
				bool const is_forward = s.a.x != s.b.x ? s.a.x < s.b.x : s.a.y <= s.b.y;
				return is_forward ? std::pair{s.a, s.b} : std::pair{s.b, s.a};
			}

			[[nodiscard]] std::pair<sweep_point, sweep_point> ends_of(segment_index const si) const {
				auto const [first, last] = sweep_order_ends(si);
				return {sweep_point{W(first.x), W(first.y), W(1)},
						sweep_point{W(last.x), W(last.y), W(1)}};
			}

			struct line_coeffs {
				W a, b, c;
			};

			/**
			 * @brief The supporting line of segment @p si, with `a = dy`,
			 *        `b = -dx`: exact in `int128` within the 2^32 bound.
			 */
			[[nodiscard]] line_coeffs line_of(segment_index const si) const {
				auto const s = segments_[si];
				W const a = W(s.b.y) - W(s.a.y);
				W const b = W(s.a.x) - W(s.b.x);
				return {a, b, -(a * W(s.a.x) + b * W(s.a.y))};
			}

			[[nodiscard]] static int sign_of(W const v) { return v > 0 ? 1 : v < 0 ? -1
																				   : 0; }

			/**
			 * @brief Sign of `a*x + b*y + c` at the rational point p — exact via
			 * an int256 three-product sum for integral coordinates.
			 */
			[[nodiscard]] int line_value_sign(line_coeffs const l, sweep_point const &p) const {
				if constexpr (std::same_as<W, int128>) {
					auto acc = atl::detail::int256::product(l.a, p.nx);
					acc += atl::detail::int256::product(l.b, p.ny);
					acc += atl::detail::int256::product(l.c, p.d);
					return acc.sign();
				} else {
					return sign_of(l.a * p.nx + l.b * p.ny + l.c * p.d);
				}
			}

			/**
			 * @brief Sign of `y_s(p.x) - p.y` for a NON-vertical segment: the line
			 * value divided by b (b = a.x - b.x != 0).
			 */
			[[nodiscard]] int y_at_versus(segment_index const si, sweep_point const &p) const {
				line_coeffs const l = line_of(si);
				// a x + b y + c = b (y - y_s(x)) => y_s(x) - y = -value / b
				return -line_value_sign(l, p) * sign_of(l.b);
			}

			[[nodiscard]] bool is_vertical(segment_index const si) const {
				return segments_[si].a.x == segments_[si].b.x;
			}

			/**
			 * @brief Sign of `y_s(p.x) - y_t(p.x)` at the current event's x; alive
			 * verticals read as exactly p.y (they always contain it).
			 */
			[[nodiscard]] int y_order(segment_index const si, segment_index const ti) const {
				bool const is_s_vertical = is_vertical(si), is_t_vertical = is_vertical(ti);
				if (is_s_vertical && is_t_vertical) return 0;
				if (is_s_vertical) return -y_at_versus(ti, current_);
				if (is_t_vertical) return y_at_versus(si, current_);
				line_coeffs const s = line_of(si), t = line_of(ti);
				// y_s - y_t = [-(as nx + cs d) bt + (at nx + ct d) bs] / (bs bt d)
				if constexpr (std::same_as<W, int128>) {
					auto acc = atl::detail::int256::product(t.a * s.b - s.a * t.b, current_.nx);
					acc += atl::detail::int256::product(t.c * s.b - s.c * t.b, current_.d);
					return acc.sign() * sign_of(s.b) * sign_of(t.b);
				} else {
					float64 const v = (t.a * s.b - s.a * t.b) * current_.nx
									+ (t.c * s.b - s.c * t.b) * current_.d;
					return sign_of(v) * sign_of(s.b) * sign_of(t.b);
				}
			}

			/**
			 * @brief Slope order; verticals are +infinity. Distinct alive segments
			 * through one point tie only when they overlap collinearly; the
			 * index tie-break then keeps the order total, and since such
			 * segments also tie against the probe they stay inside one
			 * event's `equal_range` run, which is all the event handling needs.
			 */
			[[nodiscard]] int slope_order(segment_index const si, segment_index const ti) const {
				auto const [sa, sb] = sweep_order_ends(si);
				auto const [ta, tb] = sweep_order_ends(ti);
				W const sdx = W(sb.x) - W(sa.x), sdy = W(sb.y) - W(sa.y);
				W const tdx = W(tb.x) - W(ta.x), tdy = W(tb.y) - W(ta.y);
				if (sdx == 0 && tdx == 0) return 0;
				if (sdx == 0) return 1;
				if (tdx == 0) return -1;
				return sign_of(sdy * tdx - tdy * sdx); // both dx > 0
			}

			struct probe {};

			/**
			 * @brief The stateful transparent status comparator: segments by y
			 *        at the current event's x, then slope in post-crossing order below the event,
			 *        pre-crossing order above, and the current phase at its height,
			 *        then index; against a `probe`, by the side of
			 *        the event point. Holds a pointer to its engine.
			 */
			struct status_less {
				segment_sweep const *sweep;
				using is_transparent = void;

				bool operator()(segment_index const s, segment_index const t) const {
					if (s == t) return false;
					int const by_y = sweep->y_order(s, t);
					if (by_y != 0) return by_y < 0;
					int const by_slope = sweep->slope_order(s, t);
					if (by_slope != 0) {
						// Crossings below this event have happened; those above
						// have not. Only the erased current run changes phase.
						int const side = sweep->versus_point(s);
						bool const is_after = side < 0 || (side == 0 && sweep->is_after_event_);
						return is_after ? by_slope < 0 : by_slope > 0;
					}
					return s < t;
				}

				bool operator()(segment_index const s, probe) const {
					return sweep->versus_point(s) < 0;
				}

				bool operator()(probe, segment_index const s) const {
					return sweep->versus_point(s) > 0;
				}
			};

			/**
			 * @brief Sign of `y_s(p.x) - p.y`, verticals via their y-interval
			 * (alive verticals always contain p.y, so 0).
			 */
			[[nodiscard]] int versus_point(segment_index const si) const {
				if (is_vertical(si)) {
					auto const [first, last] = ends_of(si);
					if (last < current_) return -1; // ended below (defensive)
					if (current_ < first) return 1;
					return 0;
				}
				return y_at_versus(si, current_);
			}

			/**
			 * @brief Queues the crossing point of segments @p si and @p ti
			 *        when they intersect at a single point strictly after the
			 *        current event (as an empty-starter event: the passing and
			 *        ending sets come from the status).
			 */
			void enqueue_crossing(segment_index const si, segment_index const ti) {
				auto const s = segments_[si], t = segments_[ti];
				if (!intersects(s, t)) return;
				line_coeffs const ls = line_of(si), lt = line_of(ti);
				W d = ls.a * lt.b - lt.a * ls.b;
				// parallel and intersecting means collinear: every set-change
				// point of the overlap is an input endpoint, already an event
				if (d == 0) return;
				W nx = ls.b * lt.c - lt.b * ls.c;
				W ny = ls.c * lt.a - lt.c * ls.a;
				if (d < 0) {
					d = -d;
					nx = -nx;
					ny = -ny;
				}
				sweep_point const q{nx, ny, d};
				if (!(current_ < q)) return; // at or before the current event
				queue_[q];					 // point-only: C/L come from the status
			}

			/**
			 * @brief Processes the event at `current_`: splits the alive run
			 *        through it into ending and passing, reinserts passing and
			 *        starting segments in the post-event order, records the
			 *        segment below, and queues the new neighbor crossings.
			 * @return whether the event is reported (at least two distinct
			 *         segments, or one when singletons are reported).
			 */
			bool handle_event() {
				ending_.clear();
				passing_.clear();

				auto const [run_begin, run_end] = status_.equal_range(probe{});
				std::vector<segment_index> run(run_begin, run_end);
				status_.erase(run_begin, run_end);
				for (segment_index const si : run) {
					auto const last = ends_of(si).second;
					if (!(last < current_) && !(current_ < last))
						ending_.push_back(si);
					else
						passing_.push_back(si);
				}
				// the DISTINCT segments through p: the alive run plus the
				// starters (degenerates appear in starting AND ending but
				// are one segment)
				bool const should_report =
					run.size() + starting_.size() >= (reports_singletons_ ? 1U : 2U);
				is_after_event_ = true;
				for (segment_index const si : starting_) {
					if (segments_[si].a == segments_[si].b)
						ending_.push_back(si); // never enters the status
					else
						status_.insert(si);
				}
				for (segment_index const si : passing_) status_.insert(si);
				// neighbor discovery around the (possibly empty) new block;
				// the element under the block IS the segment below the event
				auto const [block_begin, block_end] = status_.equal_range(probe{});
				below_ = block_begin != status_.begin()
						   ? *std::prev(block_begin)
						   : intersection_event::no_below;
				if (block_begin == block_end) {
					if (block_begin != status_.begin() && block_end != status_.end())
						enqueue_crossing(*std::prev(block_begin), *block_end);
				} else {
					if (block_begin != status_.begin())
						enqueue_crossing(*std::prev(block_begin), *block_begin);
					if (block_end != status_.end())
						enqueue_crossing(*std::prev(block_end), *block_end);
				}
				return should_report;
			}

			std::vector<segment<T>> segments_;
			bool reports_singletons_ = false;
			std::map<sweep_point, std::vector<segment_index>> queue_;
			std::set<segment_index, status_less> status_;
			sweep_point current_{};
			segment_index below_ = intersection_event::no_below;
			bool is_after_event_ = false;
			std::vector<segment_index> starting_, ending_, passing_;

			friend struct status_less;
		};
	}

	/**
	 * @brief Lazy single-pass range of ALL intersection points among a set
	 *        of segments — Bentley-Ottmann with the de Berg et al. handling
	 *        of degeneracies. Events arrive in SWEEP ORDER (lexicographic by
	 *        point); each carries the exact partition of the segments
	 *        through it (see @ref atl::geometry::intersection_event). Early termination is
	 *        the point of the view: "any intersection at all?" pays only for
	 *        the events before the first hit.
	 *
	 *        HANDLED degeneracies: vertical/horizontal segments, k >= 3
	 *        segments through one point (one event, all listed), endpoints
	 *        on interiors, shared endpoints, degenerate point segments, and
	 *        COLLINEAR OVERLAPS — an overlap's continuum of common points is
	 *        reported at its SET-CHANGE points (the overlap endpoints, where
	 *        a segment starts or stops sharing the line) and wherever a
	 *        third segment crosses through it; the continuum interior is not
	 *        enumerated. Identical duplicate segments report at both shared
	 *        endpoints.
	 *
	 * @tparam R Sized random-access segment range.
	 * @pre Fewer than 2^32 segments (asserted). int32 accepts its full range.
	 * @pre For integral coordinates: `|x|, |y| < 2^32` (asserted) — every
	 *      ordering decision is then exact (int256 rational comparisons).
	 *      Floating inputs use rounded comparisons throughout: best-effort
	 *      ordering, ulp-close events may merge or misorder.
	 * @complexity O((n + I) log(n + 1)) time, O(n + I) space, I = total segment/event incidences.
	 * Overlaps can make I quadratic even with only linearly many distinct events.
	 * @note Not constexpr (uses runtime ordered containers). No recursion per segment.
	 */
	template<segment_range R>
	class segment_intersections_view {
		using T = segment_coordinate_t<R>;

	public:
		/**
		 * @brief The segments are copied into the sweep (moved, for an rvalue
		 * `std::vector<segment<T>>`); @p segments need not outlive the view.
		 * @note Not constexpr (runtime ordered containers).
		 */
		explicit segment_intersections_view(R &&segments)
			: sweep_(std::forward<R>(segments)) {}

		/** @brief Input iterator consuming one shared sweep; copies are not independent. */
		class iterator {
		public:
			/** @brief Dereferenced event type. */
			using value_type = intersection_event;
			/** @brief Iterator difference type. */
			using difference_type = std::ptrdiff_t;
			/** @brief Single-pass iterator category. */
			using iterator_concept = std::input_iterator_tag;

			/** @brief Constructs an exhausted iterator. */
			iterator() = default;

			/** @brief Advances the supplied engine to its first reportable event. */
			explicit iterator(detail::segment_sweep<T> *sweep) : sweep_(sweep) {
				++*this;
			}

			/** @brief Current event; spans expire on the next increment. @pre Not exhausted. */
			[[nodiscard]] intersection_event operator*() const { return sweep_->current_event(); }

			/** @brief Consumes the current event. @pre Not exhausted. */
			iterator &operator++() {
				if (!sweep_->advance()) sweep_ = nullptr;
				return *this;
			}

			/** @brief Consumes the current event without retaining a copy. */
			void operator++(int) { ++*this; }

			/** @brief Whether the engine is exhausted. */
			[[nodiscard]] friend bool operator==(iterator const &it, std::default_sentinel_t) {
				return it.sweep_ == nullptr;
			}

		private:
			detail::segment_sweep<T> *sweep_ = nullptr;
		};

		/** @brief Starts consuming the sweep; call once. @note Not constexpr (ordered runtime state). */
		iterator begin() { return iterator{&sweep_}; }

		/** @brief Sentinel for exhaustion. @note Not constexpr (runtime range interface). */
		std::default_sentinel_t end() const { return {}; }

	private:
		detail::segment_sweep<T> sweep_;
	};

	/** @brief Deduces the input range, preserving its reference category. */
	template<segment_range R>
	segment_intersections_view(R &&) -> segment_intersections_view<R>;

	/** @brief An owned copy of one @ref atl::geometry::intersection_event (same fields, vectors for spans). */
	struct materialized_intersection {
		point<float64> at;					 ///< the common point
		std::vector<segment_index> starting; ///< Segments starting here.
		std::vector<segment_index> ending;	 ///< Segments ending here.
		std::vector<segment_index> passing;	 ///< Segments passing through here.
		segment_index below;				 ///< the segment strictly below, or intersection_event::no_below
	};

	/**
	 * @brief Eager form of @ref atl::geometry::segment_intersections_view, with every event
	 *        materialized (spans copied into owned vectors), in sweep order.
	 * @tparam R Sized random-access segment range.
	 * @param segments Closed input segments.
	 * @return Owned event records in sweep order.
	 * @pre As @ref atl::geometry::segment_intersections_view.
	 * @complexity As @ref atl::geometry::segment_intersections_view, plus the size of the
	 *             copied spans.
	 * @note Not constexpr (runtime segment sweep).
	 */
	template<segment_range R>
	[[nodiscard]] std::vector<materialized_intersection> segment_intersections(R &&segments) {
		std::vector<materialized_intersection> out;
		segment_intersections_view view(std::forward<R>(segments));
		for (auto const event : view)
			out.push_back({event.at,
						   {event.starting.begin(), event.starting.end()},
						   {event.ending.begin(), event.ending.end()},
						   {event.passing.begin(), event.passing.end()},
						   event.below});
		return out;
	}

	/** @brief Result of @ref atl::geometry::simplicity_of. */
	struct simplicity {
		bool is_simple;			 ///< the vertex sequence is a simple polygon
		point<float64> conflict; ///< a boundary self-intersection point
								 ///< (zero for fewer than three vertices; meaningless when `is_simple`)
	};

	/**
	 * @brief Is the vertex sequence @p poly a SIMPLE polygon — at least 3
	 *        vertices, no repeated vertices, and edges that intersect only at
	 *        the shared endpoints of cyclically adjacent edges? Collinear
	 *        (straight) vertices are allowed, matching the library-wide
	 *        precondition this function verifies. Either winding accepted.
	 *
	 * One pass of @ref atl::geometry::segment_intersections_view over the n boundary edges,
	 * stopping at the FIRST offending event, so the simple case is
	 * O(n log n) (exactly one event per vertex) and non-simple inputs never
	 * pay for their full event set. An event is in order exactly when no
	 * edge passes through its interior, exactly two distinct edges meet,
	 * they are cyclically adjacent, and the meeting point IS their shared
	 * vertex. Crossings and touches fail the passing test; repeated
	 * vertices, zero-length edges and pinch points fail the two-edge count;
	 * overlaps of adjacent edges fail one of the three however they fold
	 * (a partial backtrack puts the long edge in `passing` at the short
	 * one's far endpoint; a FULL mutual overlap `[a, b]`, `[b, a]` forces
	 * the repeated vertex `v(i) == v(i+2)`, whose event the cyclic
	 * structure fills with a third incident edge — or, at n == 3, with the
	 * zero-length closing edge). The shared-vertex test pins the intended
	 * semantics directly rather than resting on that enumeration.
	 *
	 * `conflict` reports the offending event's point: exact for integral
	 * inputs when it is an input vertex, the rounded crossing point
	 * otherwise. Inherits the sweep's bounds: `|x|, |y| < 2^32` for 64-bit
	 * integral coordinates (asserted) and best-effort ordering for floating
	 * inputs.
	 * @note Not constexpr (runtime segment sweep). No recursion per vertex.
	 * @complexity O(n log(n + 1)) for simple input; early exit on a conflict.
	 */
	template<vertex_range R>
	[[nodiscard]] simplicity simplicity_of(R &&poly) {
		using T = vertex_coordinate_t<R>;
		atl_assert(std::ranges::size(poly) < (uint64(1) << 32));
		uint32 const n = uint32(std::ranges::size(poly));
		auto const first = std::ranges::begin(poly);
		if (n < 3) return {false, {}};
		std::vector<segment<T>> edges;
		edges.reserve(n);
		for (uint32 i = 0; i < n; ++i)
			edges.push_back({first[i], first[(i + 1) % n]});
		segment_intersections_view view(std::move(edges));
		for (auto const event : view) {
			if (!event.passing.empty()) return {false, event.at};
			// the distinct edges meeting here (a zero-length edge sits in
			// both spans; three distinct already decides, so cap the scan)
			segment_index ids[3]{};
			int count = 0;
			auto const add = [&](segment_index const s) {
				for (int k = 0; k < count; ++k)
					if (ids[k] == s) return;
				if (count < 3) ids[count++] = s;
			};
			for (auto const s : event.starting) add(s);
			for (auto const s : event.ending) add(s);
			if (count != 2) return {false, event.at};
			auto const [lo, hi] = std::minmax(ids[0], ids[1]);
			bool const wraps = lo == 0 && hi == n - 1;
			if (hi != lo + 1 && !wraps) return {false, event.at};
			// edges lo and lo+1 share poly[lo + 1]; the wrap pair shares
			// poly[0]
			point<T> const shared = first[wraps ? 0 : hi];
			if (event.at != to_float64(shared))
				return {false, event.at};
		}
		return {true, {}};
	}
}
