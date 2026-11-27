#pragma once

/** @file
 * @brief Shared halfplane intersection implementation without adaptive floating predicates.
 */
#include <atl/geometry/halfplane_core.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/predicates_core.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/radix_sort.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::geometry {
	/** @brief Shape of an intersection of closed half-planes. */
	enum class region_kind {
		empty,	  ///< no common point
		point,	  ///< exactly one common point
		segment,  ///< a segment of positive length
		polygon,  ///< a bounded region of positive area
		unbounded ///< an unbounded region: a wedge, strip, half-plane, ray, line, ... or the plane
	};

	/**
	 * @brief The result of `halfplane_intersection`: the kind (exact under the exact policy) and
	 *        the exact boundary as INPUT INDICES, plus the corners as rounded
	 *        `double` constructions.
	 *
	 * - `polygon`: `boundary` lists the edges counterclockwise, one per edge
	 *   of positive length, starting with the edge whose direction comes
	 *   first in the `[0, 2pi)` sweep from the `+x` axis; `vertices[i] =
	 *   boundary[i - 1] ∩ boundary[i]` (cyclically), so `vertices[i]` is
	 *   where edge `i` starts.
	 * - `unbounded`: `boundary` is the boundary chain walked with the region
	 *   on the left, from the edge that comes in from infinity to the one
	 *   that leaves; `vertices[i] = boundary[i] ∩ boundary[i + 1]`, the
	 *   finite corners (`boundary.size() - 1` of them). Exceptions: the
	 *   whole plane (no input) has neither; a strip (or its zero-width limit,
	 *   a line) has two anti-parallel edges and no vertex; a ray has its two
	 *   coincident anti-parallel edges and one vertex, its apex.
	 * - `segment`: the two anti-parallel constraints carrying it,
	 *   `vertices[i]` the start of edge `i` (the segment runs from
	 *   `vertices[0]` to `vertices[1]` along `boundary[0]`'s direction).
	 * - `point`: two non-parallel constraints through it; `vertices[0] =
	 *   boundary[0] ∩ boundary[1]`.
	 * - `empty`: both empty.
	 *
	 * Several input half-planes that are the same set (`x <= 1` twice, or
	 * `2x <= 2`) report the smallest index among them.
	 */
	struct halfplane_region {
		/** @brief The shape classification; exact under the exact policy. */
		region_kind kind{region_kind::empty};
		/** @brief Input indices of the boundary constraints, in the order the kind's layout above gives. */
		std::vector<uint32> boundary;
		/** @brief The corners as rounded `double` constructions, aligned with `boundary` as above. */
		std::vector<point<float64>> vertices;
	};

	/**
	 * @brief The sign policies `halfplane_intersection` accepts:
	 *        `exact` (the default: every decision exact, a filtered
	 *        expansion for floating input) and `fast` (the rounded
	 *        determinant for floating input). Integral input is exact under
	 *        both.
	 */
	template<typename Policy>
	concept halfplane_intersection_policy = std::same_as<Policy, exact> || std::same_as<Policy, fast>;

	namespace detail {
		template<halfplane_coefficient Coefficient>
		struct halfplane_constraint {
			Coefficient p{}, q{}, r{};
			uint32 index{};
		};

		template<typename T>
		struct halfplane_intersection_arithmetic {
			static_assert(sizeof(T) == 0, "Include <atl/geometry/halfplane_intersection.hpp> before intersecting floating half-planes");
		};

		/** @brief EXACT sign of `a b - c d` (`int128` products: `|a b|, |c d| < 2^126`). */
		[[nodiscard]] constexpr int product_difference_sign(int64 const a, int64 const b, int64 const c, int64 const d) {
			return sign_of(int128(a) * b - int128(c) * d);
		}

		/**
		 * @brief Sign of `p_a q_b - p_b q_a`: the cross product of the two
		 *        boundary DIRECTIONS `(-q, p)` (positive: `b` turns
		 *        counterclockwise from `a`). Exact: `int64` products under
		 *        the `2^31` bound, the monotone-rounding test for doubles.
		 */
		template<halfplane_coefficient Coefficient>
		[[nodiscard]] constexpr int direction_cross_sign(halfplane_constraint<Coefficient> const &a,
														 halfplane_constraint<Coefficient> const &b) {
			if constexpr (std::same_as<Coefficient, int64>)
				return sign_of(a.p * b.q - b.p * a.q);
			else
				return halfplane_intersection_arithmetic<Coefficient>::product_difference_sign(a.p, b.q, b.p, a.q);
		}

		/**
		 * @brief Whether the direction `(-q, p)` lies in `[0, pi)` of the
		 *        `[0, 2pi)` sweep from the `+x` axis (coefficient signs are
		 *        exact).
		 */
		template<halfplane_coefficient Coefficient>
		[[nodiscard]] constexpr bool is_first_half(halfplane_constraint<Coefficient> const &a) {
			return a.p > 0 || (a.p == 0 && a.q < 0);
		}

		/** @brief Strict `[0, 2pi)` sweep order of the boundary directions. */
		template<halfplane_coefficient Coefficient>
		[[nodiscard]] constexpr bool sweeps_before(halfplane_constraint<Coefficient> const &a,
												   halfplane_constraint<Coefficient> const &b) {
			bool const is_a_first = is_first_half(a), is_b_first = is_first_half(b);
			if (is_a_first != is_b_first) return is_a_first;
			return direction_cross_sign(a, b) > 0;
		}

		template<halfplane_coefficient Coefficient>
		[[nodiscard]] constexpr Coefficient magnitude_of(Coefficient const v) {
			return v < 0 ? -v : v;
		}

		/**
		 * @brief For two PARALLEL constraints (normals `n_b = k n_a`, `k != 0`):
		 *        the exact sign of `r_a |n_b| + s r_b |n_a|` up to the
		 *        positive factor `1 / |n_a|`. With `s = -1` and the same
		 *        direction it compares the offsets (negative: `a` is the
		 *        tighter); with `s = +1` and opposite directions it is the
		 *        strip's width sign (negative: empty). Computed on one nonzero
		 *        normal component, which scales by the same `|k|`.
		 */
		template<int Sign, halfplane_coefficient Coefficient>
			requires(Sign == 1 || Sign == -1)
		[[nodiscard]] constexpr int parallel_offset_sign(halfplane_constraint<Coefficient> const &a,
														 halfplane_constraint<Coefficient> const &b) {
			bool const uses_p = a.p != 0;
			Coefficient const a_scale = magnitude_of(uses_p ? a.p : a.q);
			Coefficient const b_scale = magnitude_of(uses_p ? b.p : b.q);
			if constexpr (std::same_as<Coefficient, int64>)
				return product_difference_sign(a.r, b_scale, Coefficient(-Sign) * b.r, a_scale);
			else
				return halfplane_intersection_arithmetic<Coefficient>::product_difference_sign(a.r, b_scale, Coefficient(-Sign) * b.r, a_scale);
		}

		/**
		 * @brief EXACT sign of `d_jk + d_ki + d_ij`, the coefficient of the
		 *        symbolic offset perturbation (see `halfplane_crossing_side`).
		 */
		template<halfplane_coefficient Coefficient>
		[[nodiscard]] constexpr int perturbation_sign(halfplane_constraint<Coefficient> const &i,
													  halfplane_constraint<Coefficient> const &j,
													  halfplane_constraint<Coefficient> const &k) {
			if constexpr (std::same_as<Coefficient, int64>) {
				return sign_of(int128(j.p * k.q - k.p * j.q) + int128(k.p * i.q - i.p * k.q)
							   + int128(i.p * j.q - j.p * i.q));
			} else {
				return halfplane_intersection_arithmetic<Coefficient>::perturbation_sign(i, j, k);
			}
		}

		/**
		 * @brief EXACT sign of `r_i d_jk + r_j d_ki + r_k d_ij` — the 3x3
		 *        determinant of the three constraints' `(p, q, r)` rows.
		 *        Integral: `int64` minors and `int128` terms, exact under the
		 *        coefficient bounds (`halfplane_normal_bound`). Floating:
		 *        a filter on accurately rounded minors
		 *        (`halfplane_determinant_bound`), then the exact expansion.
		 */
		template<halfplane_coefficient Coefficient>
		[[nodiscard]] constexpr int halfplane_determinant_sign(halfplane_constraint<Coefficient> const &i,
															   halfplane_constraint<Coefficient> const &j,
															   halfplane_constraint<Coefficient> const &k) {
			if constexpr (std::same_as<Coefficient, int64>) {
				int64 const minor_jk = j.p * k.q - k.p * j.q;
				int64 const minor_ki = k.p * i.q - i.p * k.q;
				int64 const minor_ij = i.p * j.q - j.p * i.q;
				return sign_of(int128(i.r) * minor_jk + int128(j.r) * minor_ki + int128(k.r) * minor_ij);
			} else {
				return halfplane_intersection_arithmetic<Coefficient>::halfplane_determinant_sign(i, j, k);
			}
		}

		/**
		 * @brief The library determinant-sign policy: `halfplane_determinant_sign`
		 *        (exact `int128`, or the filtered expansion for doubles).
		 *        Policies are the knob the experimental predicate variants turn.
		 */
		struct filtered_determinant {
			template<halfplane_coefficient Coefficient>
			[[nodiscard]] static constexpr int sign(halfplane_constraint<Coefficient> const &i,
													halfplane_constraint<Coefficient> const &j,
													halfplane_constraint<Coefficient> const &k) {
				return halfplane_determinant_sign(i, j, k);
			}
		};

		/**
		 * @brief The `fast` determinant-sign policy: the sign of the
		 *        rounded `double` determinant, no filter and no expansion, so
		 *        a near-degenerate triple may be misclassified; integral input
		 *        stays exact (`halfplane_determinant_sign`).
		 */
		struct rounded_determinant {
			template<halfplane_coefficient Coefficient>
			[[nodiscard]] static constexpr int sign(halfplane_constraint<Coefficient> const &i,
													halfplane_constraint<Coefficient> const &j,
													halfplane_constraint<Coefficient> const &k) {
				if constexpr (std::same_as<Coefficient, float64>) {
					float64 const value = i.r * (j.p * k.q - k.p * j.q) + j.r * (k.p * i.q - i.p * k.q)
										+ k.r * (i.p * j.q - j.p * i.q);
					return int(value > 0) - int(value < 0);
				} else
					return halfplane_determinant_sign(i, j, k);
			}
		};

		/**
		 * @brief Whether @p Determinant decides every sign exactly on
		 *        @p Coefficient input: all but `rounded_determinant` on
		 *        `double`. The debug-only checks of `region_of_sorted`
		 *        trust only exact signs, so they never abort on valid input
		 *        that the rounded signs misclassify.
		 */
		template<typename Determinant, halfplane_coefficient Coefficient>
		inline constexpr bool is_exact_determinant_v =
			!std::same_as<Determinant, rounded_determinant> || std::same_as<Coefficient, int64>;

		template<halfplane_intersection_policy Policy>
		using determinant_of_t = std::conditional_t<std::same_as<Policy, exact>, filtered_determinant, rounded_determinant>;

		/** @brief A type whose static `sign(i, j, k)` is the sign of the 3x3 coefficient determinant. */
		template<typename Policy, typename Coefficient>
		concept determinant_policy =
			halfplane_coefficient<Coefficient>
			&& requires(halfplane_constraint<Coefficient> const &c) {
				   { Policy::sign(c, c, c) } -> std::same_as<int>;
			   };

		/**
		 * @brief EXACT side of the crossing point of @p i and @p j relative
		 *        to @p k: `+1` strictly inside `k`, `0` on its boundary, `-1`
		 *        outside. The sign of `r_k - p_k x - q_k y` at the crossing
		 *        is the determinant's sign times the sign of `d_ij`.
		 *
		 * @p Perturbation symbolically moves every offset `r` by
		 * `Perturbation * epsilon` (`-1` shrinks every half-plane, `+1` grows
		 * it, `0` keeps the input): the determinant becomes `D +
		 * Perturbation * epsilon * (d_jk + d_ki + d_ij)`, so a zero `D` falls
		 * back to the sign of that sum. Shrinking leaves a nonempty region
		 * iff the input has interior; growing turns a point or segment into
		 * a thin polygon whose limit is it.
		 *
		 * The fallback sum can be zero too: then the three normals' tips are
		 * collinear, `k`'s row is an affine combination of `i`'s and `j`'s
		 * (`D == 0` makes it a linear one, the zero sum makes the weights
		 * sum to 1), so `k`'s boundary passes through the crossing under
		 * every shift and the tie is genuine. The side stays `0`, and the
		 * callers' boundary rule settles it as for unperturbed input
		 * (`is_redundant` drops the candidate: a zero-length edge).
		 * @pre `i`, `j` not parallel.
		 */
		template<int Perturbation, typename Determinant = filtered_determinant, halfplane_coefficient Coefficient>
			requires(Perturbation >= -1 && Perturbation <= 1) && determinant_policy<Determinant, Coefficient>
		[[nodiscard]] constexpr int halfplane_crossing_side(halfplane_constraint<Coefficient> const &i,
															halfplane_constraint<Coefficient> const &j,
															halfplane_constraint<Coefficient> const &k) {
			int side = Determinant::sign(i, j, k);
			if constexpr (Perturbation != 0)
				if (side == 0) side = Perturbation * perturbation_sign(i, j, k);
			return direction_cross_sign(i, j) > 0 ? side : -side;
		}

		/**
		 * @brief The crossing point of the boundaries of two non-parallel
		 *        constraints, rounded: `(r_i q_j - r_j q_i, p_i r_j - p_j r_i)
		 *        / d_ij` from exact `int128` numerators and denominator
		 *        converted once, or for doubles from three
		 *        `accurate_product_difference` values (relative error `2
		 *        epsilon` each): every coordinate is within a few ulps of the
		 *        rational.
		 * @pre `i`, `j` not parallel.
		 */
		template<halfplane_coefficient Coefficient>
		[[nodiscard]] constexpr point<float64> halfplane_crossing_point(halfplane_constraint<Coefficient> const &i,
																		halfplane_constraint<Coefficient> const &j) {
			if constexpr (std::same_as<Coefficient, int64>) {
				float64 const d = float64(i.p * j.q - j.p * i.q);
				return {float64(int128(i.r) * j.q - int128(j.r) * i.q) / d,
						float64(int128(i.p) * j.r - int128(j.p) * i.r) / d};
			} else {
				float64 const d = halfplane_intersection_arithmetic<Coefficient>::difference(i.p, j.q, j.p, i.q);
				return {halfplane_intersection_arithmetic<Coefficient>::difference(i.r, j.q, j.r, i.q) / d,
						halfplane_intersection_arithmetic<Coefficient>::difference(i.p, j.r, j.p, i.r) / d};
			}
		}

		/**
		 * @brief A deque that only ever grows at the back: a vector with a
		 *        moving head cursor. Every constraint is pushed at most once,
		 *        so the capacity reserved up front is never exceeded and no
		 *        operation reallocates.
		 */
		template<std::copyable Element>
		class cursor_deque {
		public:
			explicit constexpr cursor_deque(std::size_t const capacity) { items_.reserve(capacity); }

			constexpr void push_back(Element const &e) { items_.push_back(e); }

			constexpr void pop_back() { items_.pop_back(); }

			constexpr void pop_front() { ++head_; }

			constexpr void clear() {
				items_.clear();
				head_ = 0;
			}

			constexpr std::size_t size() const { return items_.size() - head_; }

			[[nodiscard]] constexpr Element const &operator[](std::size_t const i) const { return items_[head_ + i]; }

			[[nodiscard]] constexpr Element const &front() const { return items_[head_]; }

			[[nodiscard]] constexpr Element const &back() const { return items_.back(); }

		private:
			std::vector<Element> items_;
			std::size_t head_ = 0;
		};

		/** @brief What the bounded pass needs from its deque (`cursor_deque`, `std::deque`). */
		template<typename Buffer, typename Element>
		concept constraint_deque = requires(Buffer &b, Buffer const &cb, Element const &e, std::size_t i) {
			b.push_back(e);
			b.pop_back();
			b.pop_front();
			b.clear();
			{ cb.size() } -> std::convertible_to<std::size_t>;
			{ cb[i] } -> std::convertible_to<Element const &>;
			{ cb.front() } -> std::convertible_to<Element const &>;
			{ cb.back() } -> std::convertible_to<Element const &>;
		};

		/**
		 * @brief Is @p candidate redundant given its surviving neighbor
		 *        @p keeper and the @p incoming constraint (zero-length edges
		 *        count as redundant)? Non-parallel neighbors: their crossing
		 *        is not strictly inside `incoming`. Anti-parallel neighbors
		 *        (same directions were merged) have no crossing: the candidate
		 *        is implied iff the keeper-incoming corner satisfies it.
		 */
		template<int Perturbation, typename Determinant, halfplane_coefficient Coefficient>
			requires(Perturbation >= -1 && Perturbation <= 1) && determinant_policy<Determinant, Coefficient>
		[[nodiscard]] constexpr bool is_redundant(halfplane_constraint<Coefficient> const &keeper,
												  halfplane_constraint<Coefficient> const &candidate,
												  halfplane_constraint<Coefficient> const &incoming) {
			if (direction_cross_sign(keeper, candidate) == 0)
				return halfplane_crossing_side<Perturbation, Determinant>(keeper, incoming, candidate) >= 0;
			return halfplane_crossing_side<Perturbation, Determinant>(keeper, candidate, incoming) <= 0;
		}

		/**
		 * @brief The final cleanup of a deque pass: the last constraints
		 *        against the first ones and back, until nothing changes.
		 */
		template<int Perturbation, typename Determinant, halfplane_coefficient Coefficient,
				 constraint_deque<halfplane_constraint<Coefficient>> Buffer>
			requires(Perturbation >= -1 && Perturbation <= 1) && determinant_policy<Determinant, Coefficient>
		constexpr void close_deque(Buffer &active) {
			for (bool has_changed = true; has_changed;) {
				has_changed = false;
				while (active.size() >= 3
					   && is_redundant<Perturbation, Determinant>(active[active.size() - 2], active.back(),
																  active.front())) {
					active.pop_back();
					has_changed = true;
				}
				while (active.size() >= 3 && is_redundant<Perturbation, Determinant>(active[1], active[0], active.back())) {
					active.pop_front();
					has_changed = true;
				}
			}
		}

		/**
		 * @brief The deque pass over constraints sorted by direction, ALL
		 *        consecutive directions less than `pi` apart (a bounded
		 *        region or none): the classic pops of the core in
		 *        convex_polygon_operations.hpp, on coefficients, under the
		 *        symbolic @p Perturbation, with zero-length edges popped too.
		 *
		 * While the region is nonempty the deque is exactly the boundary of
		 * the processed constraints' intersection. Until the processed
		 * directions span `pi` that region is unbounded and its first edge
		 * runs in from infinity inside every later constraint, so no front
		 * pop can happen: those steps test the back only. From then on the
		 * region is closed off on every side, so a constraint that leaves
		 * fewer than two survivors has cut it away entirely: that is the
		 * empty answer, found the moment it happens.
		 */
		struct deque_pass {
			/**
			 * @return Whether the region is nonempty (with `>= 3` edges left
			 *         in @p active, in sweep order).
			 */
			template<int Perturbation, typename Determinant, halfplane_coefficient Coefficient,
					 constraint_deque<halfplane_constraint<Coefficient>> Buffer>
				requires(Perturbation >= -1 && Perturbation <= 1) && determinant_policy<Determinant, Coefficient>
			[[nodiscard]] static constexpr bool run(std::vector<halfplane_constraint<Coefficient>> const &sorted, Buffer &active) {
				active.clear();
				std::size_t const m = sorted.size();
				std::size_t t = 0;
				for (; t < m && (t == 0 || direction_cross_sign(sorted[0], sorted[t]) > 0); ++t) {
					while (active.size() >= 2
						   && is_redundant<Perturbation, Determinant>(active[active.size() - 2], active.back(), sorted[t]))
						active.pop_back();
					active.push_back(sorted[t]);
				}
				for (; t < m; ++t) {
					auto const &c = sorted[t];
					while (active.size() >= 2
						   && is_redundant<Perturbation, Determinant>(active[active.size() - 2], active.back(), c))
						active.pop_back();
					while (active.size() >= 2 && is_redundant<Perturbation, Determinant>(active[1], active[0], c))
						active.pop_front();
					if (active.size() < 2) return false;
					active.push_back(c);
				}
				close_deque<Perturbation, Determinant, Coefficient>(active);
				return active.size() >= 3;
			}
		};

		/**
		 * @brief A type whose static `run<Perturbation, Determinant>(sorted,
		 *        buffer)` is a bounded pass: checked for both perturbations
		 *        `region_of_sorted` runs (`-1`, then `+1`), with the
		 *        @p Determinant and the @p Buffer it actually passes.
		 */
		template<typename Pass, typename Determinant, typename Coefficient, typename Buffer>
		concept bounded_pass_policy =
			determinant_policy<Determinant, Coefficient> && constraint_deque<Buffer, halfplane_constraint<Coefficient>>
			&& requires(std::vector<halfplane_constraint<Coefficient>> const &sorted, Buffer &active) {
				   { Pass::template run<-1, Determinant>(sorted, active) } -> std::same_as<bool>;
				   { Pass::template run<1, Determinant>(sorted, active) } -> std::same_as<bool>;
			   };

		/**
		 * @brief Certificate of a bounded pass (checked under `ATL_ASSERT`):
		 *        the survivors form a convex polygon — consecutive directions
		 *        turn counterclockwise by less than `pi`, every edge has
		 *        positive length — and every constraint contains its
		 *        extreme vertex: the corner of the two survivors whose
		 *        directions bracket its own. O(n).
		 */
		template<int Perturbation, typename Determinant, halfplane_coefficient Coefficient,
				 constraint_deque<halfplane_constraint<Coefficient>> Buffer>
			requires(Perturbation >= -1 && Perturbation <= 1) && determinant_policy<Determinant, Coefficient>
		[[nodiscard]] constexpr bool certifies_bounded(std::vector<halfplane_constraint<Coefficient>> const &sorted,
													   Buffer const &active) {
			std::size_t const k = active.size();
			if (k < 3) return false;
			for (std::size_t t = 0; t < k; ++t) {
				auto const &before = active[t == 0 ? k - 1 : t - 1], &at = active[t], &after = active[(t + 1) % k];
				if (direction_cross_sign(at, after) <= 0) return false;
				if (halfplane_crossing_side<Perturbation, Determinant>(before, at, after) <= 0) return false;
			}
			std::size_t next = 0; // first survivor not yet passed in sweep order
			for (auto const &c : sorted) {
				if (next < k && c.index == active[next].index) {
					++next;
					continue;
				}
				auto const &low = active[next == 0 ? k - 1 : next - 1], &high = active[next == k ? 0 : next];
				if (halfplane_crossing_side<Perturbation, Determinant>(low, high, c) < 0) return false;
			}
			return next == k;
		}

		/**
		 * @brief Keep one constraint per direction from a direction-sorted
		 *        list: the tightest, the smallest input index among equally
		 *        tight ones.
		 */
		template<halfplane_coefficient Coefficient>
		constexpr void merge_directions(std::vector<halfplane_constraint<Coefficient>> &constraints) {
			using constraint = halfplane_constraint<Coefficient>;
			std::size_t kept = 0;
			for (std::size_t i = 0; i < constraints.size(); ++i) {
				constraint const &c = constraints[i];
				if (kept > 0 && is_first_half(constraints[kept - 1]) == is_first_half(c)
					&& direction_cross_sign(constraints[kept - 1], c) == 0) {
					constraint &held = constraints[kept - 1];
					int const tightness = parallel_offset_sign<-1>(c, held);
					if (tightness < 0 || (tightness == 0 && c.index < held.index)) held = c;
				} else
					constraints[kept++] = c;
			}
			constraints.resize(kept);
		}

		/**
		 * @brief Sort into the `[0, 2pi)` direction sweep by the exact
		 *        comparator alone, then `merge_directions`.
		 * @complexity O(n log n) exact comparisons
		 */
		template<halfplane_coefficient Coefficient>
		constexpr void sort_and_merge_by_comparator(std::vector<halfplane_constraint<Coefficient>> &constraints) {
			using constraint = halfplane_constraint<Coefficient>;
			std::ranges::sort(constraints, [](constraint const &a, constraint const &b) { return sweeps_before(a, b); });
			merge_directions(constraints);
		}

		/**
		 * @brief A 32-bit sort key MONOTONE in the direction's `[0, 2pi)`
		 *        sweep position: the half in the top bit, then the
		 *        order-preserving bits of the slope `q / p` (`-inf` when `p ==
		 *        0`) rounded to `float`, less their lowest bit. Within a half
		 *        the direction's angle increases with `q / p`, and every
		 *        rounding step (the `double` quotient, the `float` narrowing,
		 *        the dropped bit) is monotone, so a smaller key always means an
		 *        earlier direction. Equal keys are left to the exact comparator;
		 *        at worst (every direction within a few `float` ulps) that is
		 *        the comparator sort.
		 */
		template<halfplane_coefficient Coefficient>
		[[nodiscard]] constexpr uint32 direction_key(halfplane_constraint<Coefficient> const &c) {
			float32 const slope = c.p == 0 ? -std::numeric_limits<float32>::infinity()
										   : float32(float64(c.q) / float64(c.p)) + 0.0f; // no -0
			uint32 bits = std::bit_cast<uint32>(slope);
			bits = bits >> 31 != 0 ? ~bits : bits | uint32(1) << 31;
			return uint32(!is_first_half(c)) << 31 | bits >> 1;
		}

		/**
		 * @brief Reorder @p constraints by keys sorted with a monotone
		 *        direction key, settle runs of equal keys with the exact
		 *        comparator, then `merge_directions`.
		 */
		template<halfplane_coefficient Coefficient, atl::unsigned_integral Key>
			requires std::same_as<Key, std::remove_cv_t<Key>>
		constexpr void settle_by_keys(std::vector<halfplane_constraint<Coefficient>> &constraints,
									  std::vector<atl::detail::keyed_position<Key>> const &keys) {
			using constraint = halfplane_constraint<Coefficient>;
			std::size_t const n = constraints.size();
			std::vector<constraint> sorted(n);
			for (std::size_t i = 0; i < n; ++i) sorted[i] = constraints[keys[i].position];
			auto const exact_order = [](constraint const &a, constraint const &b) { return sweeps_before(a, b); };
			for (std::size_t run = 0; run < n;) {
				std::size_t end = run + 1;
				while (end < n && keys[end].key == keys[run].key) ++end;
				if (end - run > 1)
					std::ranges::sort(sorted.begin() + std::ptrdiff_t(run), sorted.begin() + std::ptrdiff_t(end), exact_order);
				run = end;
			}
			constraints = std::move(sorted);
			merge_directions(constraints);
		}

		/**
		 * @brief Sort by `direction_key` — skipped when the input is
		 *        already in order, a comparison sort below 2048 constraints,
		 *        an 11-bit LSD radix sort (three passes,
		 *        `atl::detail::radix_sort`) above — then
		 *        `settle_by_keys`.
		 * @pre Size fits uint32 and ptrdiff_t (asserted).
		 * @complexity O(n) for the keys, plus O(k log k) exact comparisons
		 *             per run of k equal keys
		 */
		template<halfplane_coefficient Coefficient>
		constexpr void sort_and_merge_by_key(std::vector<halfplane_constraint<Coefficient>> &constraints) {
			std::size_t const n = constraints.size();
			atl_assert(n <= (std::numeric_limits<uint32>::max)());
			atl_assert(n <= std::size_t((std::numeric_limits<std::ptrdiff_t>::max)()));
			using keyed = atl::detail::keyed_position<uint32>;
			std::vector<keyed> keys(n);
			for (std::size_t i = 0; i < n; ++i) keys[i] = {direction_key(constraints[i]), uint32(i)};
			auto const by_key = [](keyed const &a, keyed const &b) { return a.key < b.key; };
			if (!std::ranges::is_sorted(keys, by_key)) {
				if (n < 2048)
					std::ranges::sort(keys, by_key);
				else
					atl::detail::radix_sort<11>(keys);
			}
			settle_by_keys(constraints, keys);
		}

		/**
		 * @brief Monotone keys in at most eight ascending runs avoid a full key
		 * sort: a circular rotation is rotated back, otherwise runs are merged
		 * pairwise. More runs use comparison sorting below 2048 keys, 11-bit
		 * radix above. Equal keys always receive exact angular settlement,
		 * so the tightest/smallest-index direction representative is unchanged.
		 * @pre Size fits uint32 and ptrdiff_t (asserted).
		 * @complexity O(n log runs) key merging on up to eight runs, plus
		 * O(k log k) exact comparisons per equal-key group of size k.
		 */
		template<halfplane_coefficient Coefficient>
		constexpr void sort_and_merge_rotated_runs(std::vector<halfplane_constraint<Coefficient>> &constraints) {
			using keyed = atl::detail::keyed_position<uint32>;
			std::size_t const n = constraints.size();
			atl_assert(n <= (std::numeric_limits<uint32>::max)());
			atl_assert(n <= std::size_t((std::numeric_limits<std::ptrdiff_t>::max)()));
			std::vector<keyed> keys(n);
			for (std::size_t i = 0; i < n; ++i) keys[i] = {direction_key(constraints[i]), uint32(i)};
			auto const by_key = [](keyed const &a, keyed const &b) { return a.key < b.key; };
			std::vector<std::size_t> run_starts{0};
			for (std::size_t i = 1; i < n && run_starts.size() <= 8; ++i)
				if (keys[i].key < keys[i - 1].key) run_starts.push_back(i);
			if (run_starts.size() > 8) {
				if (n < 2048)
					std::ranges::sort(keys, by_key);
				else
					atl::detail::radix_sort<11>(keys);
			} else if (run_starts.size() == 2 && keys.back().key <= keys.front().key)
				std::ranges::rotate(keys, keys.begin() + std::ptrdiff_t(run_starts[1]));
			else if (run_starts.size() > 1) {
				run_starts.push_back(n);
				std::vector<keyed> scratch(n);
				while (run_starts.size() > 2) {
					std::vector<std::size_t> merged_starts;
					std::size_t r = 0;
					for (; r + 2 < run_starts.size(); r += 2) {
						auto const begin = keys.begin();
						std::ranges::merge(begin + std::ptrdiff_t(run_starts[r]), begin + std::ptrdiff_t(run_starts[r + 1]),
										   begin + std::ptrdiff_t(run_starts[r + 1]),
										   begin + std::ptrdiff_t(run_starts[r + 2]),
										   scratch.begin() + std::ptrdiff_t(run_starts[r]), by_key);
						merged_starts.push_back(run_starts[r]);
					}
					if (r + 1 < run_starts.size()) { // an odd run out
						std::ranges::copy(keys.begin() + std::ptrdiff_t(run_starts[r]), keys.end(),
										  scratch.begin() + std::ptrdiff_t(run_starts[r]));
						merged_starts.push_back(run_starts[r]);
					}
					merged_starts.push_back(n);
					std::ranges::swap(keys, scratch);
					run_starts = std::move(merged_starts);
				}
			}
			settle_by_keys(constraints, keys);
		}

		/**
		 * @brief The region of constraints sorted and merged by
		 *        `sort_and_merge_by_key`.
		 *
		 * A cyclic gap of at least `pi` between consecutive directions means
		 * the normals fit in a closed half-plane, so a nonempty region is
		 * unbounded; its boundary is one chain over the directions read from
		 * just after the gap, built by a single stack (no front pops). A gap
		 * of exactly `pi` closes the chain between two anti-parallel lines,
		 * whose strip may be empty or of zero width (a ray). Without such a
		 * gap the region is bounded: the shrunk pass decides whether it has
		 * interior, the grown pass whether a point or segment remains.
		 */
		template<typename Determinant, typename Pass, halfplane_coefficient Coefficient,
				 constraint_deque<halfplane_constraint<Coefficient>> Buffer>
			requires bounded_pass_policy<Pass, Determinant, Coefficient, Buffer>
		[[nodiscard]] constexpr halfplane_region region_of_sorted(std::vector<halfplane_constraint<Coefficient>> const &h,
																  Buffer &active) {
			using constraint = halfplane_constraint<Coefficient>;
			halfplane_region out;
			std::size_t const m = h.size();
			if (m == 0) {
				out.kind = region_kind::unbounded;
				return out;
			}
			std::size_t gap = m;
			if (m == 1)
				gap = 0;
			else
				for (std::size_t g = 0; g < m && gap == m; ++g)
					if (direction_cross_sign(h[g], h[g + 1 == m ? 0 : g + 1]) <= 0) gap = g;

			if (gap == m) {
				if (Pass::template run<-1, Determinant>(h, active)) {
					if constexpr (is_exact_determinant_v<Determinant, Coefficient>) atl_assert((certifies_bounded<-1, Determinant>(h, active)));
					// drop the edges of zero length in the unperturbed input:
					// their two corners coincide
					std::size_t const k = active.size();
					std::vector<std::size_t> kept;
					kept.reserve(k);
					for (std::size_t t = 0; t < k; ++t) {
						auto const &before = active[t == 0 ? k - 1 : t - 1], &after = active[t + 1 == k ? 0 : t + 1];
						if (halfplane_crossing_side<0, Determinant>(before, active[t], after) != 0) kept.push_back(t);
					}
					out.kind = region_kind::polygon;
					out.boundary.reserve(kept.size());
					out.vertices.reserve(kept.size());
					for (std::size_t b = 0; b < kept.size(); ++b) {
						auto const &at = active[kept[b]];
						out.boundary.push_back(at.index);
						out.vertices.push_back(
							halfplane_crossing_point(active[kept[b == 0 ? kept.size() - 1 : b - 1]], at));
					}
					return out;
				}
				if (!Pass::template run<1, Determinant>(h, active)) return out; // empty
				if constexpr (is_exact_determinant_v<Determinant, Coefficient>) atl_assert((certifies_bounded<1, Determinant>(h, active)));
				// a thin polygon around a point or a segment: its edges of
				// positive length in the limit are the segment's two sides
				std::size_t const k = active.size();
				std::size_t sides[2] = {k, k};
				std::size_t found = 0;
				for (std::size_t t = 0; t < k; ++t) {
					auto const &before = active[t == 0 ? k - 1 : t - 1], &after = active[t + 1 == k ? 0 : t + 1];
					if (halfplane_crossing_side<0, Determinant>(before, active[t], after) != 0) {
						// rounded signs may misclassify the count: keep the first two
						if constexpr (is_exact_determinant_v<Determinant, Coefficient>) atl_assert((found < 2));
						if (found < 2) sides[found] = t;
						++found;
					}
				}
				if constexpr (is_exact_determinant_v<Determinant, Coefficient>) atl_assert((found == 0 || found == 2));
				if (found < 2) {
					out.kind = region_kind::point;
					out.boundary = {active[0].index, active[1].index};
					out.vertices = {halfplane_crossing_point(active[0], active[1])};
					return out;
				}
				out.kind = region_kind::segment;
				std::size_t const e = sides[0];
				auto const &before = active[e == 0 ? k - 1 : e - 1], &after = active[e + 1 == k ? 0 : e + 1];
				out.boundary = {active[e].index, active[sides[1]].index};
				out.vertices = {halfplane_crossing_point(before, active[e]),
								halfplane_crossing_point(active[e], after)};
				return out;
			}

			out.kind = region_kind::unbounded;
			if (m == 1) {
				out.boundary = {h[0].index};
				return out;
			}
			if (m == 2 && direction_cross_sign(h[0], h[1]) == 0) { // strip or line
				if (parallel_offset_sign<1>(h[0], h[1]) < 0)
					out.kind = region_kind::empty;
				else
					out.boundary = {h[0].index, h[1].index};
				return out;
			}
			// the chain over the directions read from just after the gap
			auto const at = [&](std::size_t const t) -> constraint const & {
				std::size_t const s = gap + 1 + t;
				return h[s >= m ? s - m : s];
			};
			constraint const &first = at(0), &last = at(m - 1);
			if (direction_cross_sign(last, first) == 0) { // ends anti-parallel
				int const width = parallel_offset_sign<1>(first, last);
				if (width < 0) {
					out.kind = region_kind::empty;
					return out;
				}
				if (width == 0) {
					// a ray along `first`, from the crossing that comes first
					// along its direction: every middle constraint keeps the
					// part of the line before its own crossing
					std::size_t apex = 1;
					for (std::size_t t = 2; t + 1 < m; ++t)
						if (halfplane_crossing_side<0, Determinant>(first, at(t), at(apex)) > 0) apex = t;
					out.boundary = {first.index, last.index};
					out.vertices = {halfplane_crossing_point(first, at(apex))};
					return out;
				}
			}
			std::vector<constraint> chain;
			chain.reserve(m);
			for (std::size_t t = 0; t < m; ++t) {
				constraint const &c = at(t);
				while (chain.size() >= 2 && halfplane_crossing_side<0, Determinant>(chain[chain.size() - 2], chain.back(), c) <= 0)
					chain.pop_back();
				chain.push_back(c);
			}
			out.boundary.reserve(chain.size());
			out.vertices.reserve(chain.size() - 1);
			for (std::size_t t = 0; t < chain.size(); ++t) {
				out.boundary.push_back(chain[t].index);
				if (t > 0) out.vertices.push_back(halfplane_crossing_point(chain[t - 1], chain[t]));
			}
			return out;
		}

		/**
		 * @brief The input as constraints, bounds asserted.
		 * @pre Fewer than `2^32` half-planes; each within the bounds of
		 *      `halfplane`::at_most (both asserted).
		 */
		template<halfplane_range R>
		[[nodiscard]] constexpr std::vector<halfplane_constraint<halfplane_coefficient_t<halfplane_coordinate_t<R>>>>
		constraints_of(R const &halfplanes) {
			using Coefficient = halfplane_coefficient_t<halfplane_coordinate_t<R>>;
			auto const input_size = std::ranges::size(halfplanes);
			atl_assert(input_size <= (std::numeric_limits<uint32>::max)());
			atl_assert(input_size <= std::size_t((std::numeric_limits<std::ptrdiff_t>::max)()));
			std::size_t const n = static_cast<std::size_t>(input_size);
			std::vector<halfplane_constraint<Coefficient>> constraints;
			constraints.reserve(n);
			uint32 index = 0;
			for (auto &&value : halfplanes) {
				auto const h = materialize_halfplane<halfplane_coordinate_t<R>>(std::forward<decltype(value)>(value));
				atl_assert(is_valid_halfplane(h.p, h.q, h.r));
				constraints.push_back({h.p, h.q, h.r, index++});
			}
			return constraints;
		}
	}

	/**
	 * @brief The `[0, 2pi)` sweep order of the boundary directions (the
	 *        direction of `p x + q y <= r` is `(-q, p)`, the interior on its
	 *        left), from the `+x` axis counterclockwise: the order
	 *        `halfplane_intersection` sorts its input into. Exact (signs
	 *        of `2x2` coefficient determinants, no `atan2`); half-planes
	 *        with the same direction are equivalent.
	 *
	 * Sorting once pays off when the same directions are intersected many
	 * times with other offsets. The largest inscribed circle, say, binary
	 * searches its radius `rho`: sort once with
	 * `std::ranges::sort(h, halfplane_sweep_less{})`, then each step shifts
	 * every offset to `r - rho sqrt(p^2 + q^2)` (the directions, hence the
	 * order, stay) and calls `halfplane_intersection` on the shifted
	 * copy, which finds its input in sweep order with one O(n) check and
	 * skips the full key sort; equal-key groups still need exact settlement.
	 * @pre Both half-planes within the bounds of `halfplane`::at_most
	 *      (asserted).
	 */
	struct halfplane_sweep_less {
		/** @brief Compare boundary directions; equal directions remain equivalent.
		 * @tparam T Coordinate type.
		 * @param a First constraint.
		 * @param b Second constraint.
		 * @return Whether a precedes b in the sweep.
		 */
		template<coordinate T>
		[[nodiscard]] constexpr bool operator()(halfplane<T> const &a, halfplane<T> const &b) const {
			using constraint = detail::halfplane_constraint<halfplane_coefficient_t<T>>;
			atl_assert(detail::is_valid_halfplane(a.p, a.q, a.r) && detail::is_valid_halfplane(b.p, b.q, b.r));
			return detail::sweeps_before(constraint{a.p, a.q, a.r, 0}, constraint{b.p, b.q, b.r, 0});
		}
	};

	/**
	 * @brief The intersection of closed half-planes, classified EXACTLY:
	 *        empty, a point, a segment, a polygon, or unbounded, with the
	 *        boundary as input indices (see `halfplane_region` for the
	 *        layout per kind). Corners are rounded `double` constructions.
	 *
	 * Sort by direction (exact, no `atan2`), keep the tightest constraint
	 * per direction, then: an unbounded chain by one monotone stack, or a
	 * bounded region by the classic deque under a symbolic shrink (interior
	 * or not) and, only when that is empty, a symbolic grow (point, segment
	 * or empty). Every decision is the sign of a 3x3 determinant of
	 * coefficients: exact in `int128` for integral input; for floating
	 * input a filtered expansion under `exact` (the default), the
	 * rounded `double` sign under `fast`. Input already in
	 * `halfplane_sweep_less` order skips the sort. Integral coordinates also
	 * rotate or merge up to eight ascending key runs; more runs keep the full sort.
	 *
	 * `fast` trades exactness on near-degenerate floating input (nearly
	 * concurrent or nearly parallel boundaries may get a wrong kind or a
	 * missing or spurious edge) for speed; the direction sort stays exact.
	 * Measured on the experimental `rounded_determinant` reference row
	 * (benchmarks/geometry/halfplane_intersection.cpp, record of
	 * 2026-10-24, Xeon Platinum 8280, 1e6 doubles, plain `-O2`, ms): 162 vs
	 * 192 on tangents of a circle, 151 vs 182 around the origin, 127 vs
	 * 140 empty, 119 vs 149 presorted. Under `ATL_ASSERT` the certificate
	 * of the bounded pass is checked only where the signs are exact (every
	 * policy on integral input, `exact` on floating input): a debug
	 * build never aborts on valid input that `fast` misclassifies.
	 * @pre Fewer than `2^32` half-planes; integral coefficients `|p|, |q| <
	 *      2^31`, `|r| < 2^62`, `(p, q) != (0, 0)`; floating: finite, no
	 *      overflow or underflow in products of three coefficients (all
	 *      asserted but the last).
	 * @param halfplanes Constraints; not modified.
	 * @tparam R Const-readable half-plane range.
	 * @tparam Policy Exact or fast determinant policy.
	 * @return Classification, input boundary indices and rounded corners.
	 * @pre Size also fits ptrdiff_t (asserted). Floating pairwise and triple
	 * products must be finite and normal (zero factors allowed); intermediate
	 * sums and constructed vertices must remain finite. Round-to-nearest and
	 * no unsafe floating reassociation are required; there is no rescaling.
	 * @complexity O(n log n) time, O(n) memory; O(log n) sort stack, other passes iterative.
	 */
	template<halfplane_range R, halfplane_intersection_policy Policy = exact>
	[[nodiscard]] constexpr halfplane_region halfplane_intersection(R const &halfplanes, Policy const = {}) {
		using Coefficient = halfplane_coefficient_t<halfplane_coordinate_t<R>>;
		auto constraints = detail::constraints_of(halfplanes);
		if constexpr (std::integral<halfplane_coordinate_t<R>>)
			detail::sort_and_merge_rotated_runs(constraints);
		else
			detail::sort_and_merge_by_key(constraints);
		detail::cursor_deque<detail::halfplane_constraint<Coefficient>> active(constraints.size());
		return detail::region_of_sorted<detail::determinant_of_t<Policy>, detail::deque_pass>(constraints, active);
	}
}
