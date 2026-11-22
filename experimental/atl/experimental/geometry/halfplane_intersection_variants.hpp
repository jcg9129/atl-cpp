#pragma once
#include <atl/geometry/halfplane.hpp>
#include <atl/geometry/halfplane_intersection.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/radix_sort.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <deque>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Contenders raced against the library atl::geometry::halfplane_intersection
 *        (benchmarks/geometry/halfplane_intersection.cpp). Each is one knob
 *        of the library composition turned: the determinant predicate for
 *        `double` input, the bounded pass, the direction sort, the deque
 *        buffer. Exact compositions return the library classification and boundary
 *        (tests/geometry/halfplane_intersection.cpp runs them against the
 *        same oracle).
 *
 * Numbers below were measured using benchmarks/geometry/halfplane_intersection.cpp
 * (2026-10-24, Intel Xeon Platinum 8280, x86-64),
 * as milliseconds at n = 1e6, plain `-O2` / `-march=native`.
 */

namespace atl::experimental {
	namespace detail {
		using geometry::detail::halfplane_coefficient;
		using geometry::detail::halfplane_constraint;

		/**
		 * @brief Determinant policy without the static filter: the exact
		 *        expansion on every call for `double` input (integral input
		 *        is `int128` either way).
		 *        LOST everywhere (1e6, ms, -O2 / native): tangents 541 / 425 vs the
		 *        library 192 / 157, around the origin 520 / 403 vs 182 / 151,
		 *        empty 275 / 224 vs 140 / 128, presorted 509 / 390 vs 149 / 119.
		 */
		struct always_exact_determinant {
			template<halfplane_coefficient Coefficient>
			[[nodiscard]] static constexpr int sign(halfplane_constraint<Coefficient> const &i,
													halfplane_constraint<Coefficient> const &j,
													halfplane_constraint<Coefficient> const &k) {
				if constexpr (std::same_as<Coefficient, float64>)
					return geometry::detail::halfplane_determinant_sign_exact(i, j, k);
				else
					return geometry::detail::halfplane_determinant_sign(i, j, k);
			}
		};

		/**
		 * @brief A product-permanent filter for `double` input: the rounded
		 *        determinant against the permanent of the PRODUCTS inside the
		 *        minors (Shewchuk's stage-A shape), then the exact expansion.
		 *        Nearly parallel constraints cancel inside the minors, so on
		 *        tangents of one circle it falls through to the expansion on
		 *        most calls; the library filter bounds against the accurately
		 *        rounded minors instead.
		 *        LOST on the near-parallel shapes (1e6, ms, -O2 / native): tangents
		 *        450 / 363 vs 192 / 157, presorted 408 / 323 vs 149 / 119; it
		 *        wins slightly where the filter rarely fails at plain -O2
		 *        (around the origin 168 vs 182, empty 134 vs 140), where its
		 *        plain products beat two `std::fma` library calls per minor.
		 */
		struct product_filter_determinant {
			template<halfplane_coefficient Coefficient>
			[[nodiscard]] static constexpr int sign(halfplane_constraint<Coefficient> const &i,
													halfplane_constraint<Coefficient> const &j,
													halfplane_constraint<Coefficient> const &k) {
				if constexpr (std::same_as<Coefficient, float64>) {
					using geometry::detail::magnitude_of;
					float64 const jk_left = j.p * k.q, jk_right = k.p * j.q;
					float64 const ki_left = k.p * i.q, ki_right = i.p * k.q;
					float64 const ij_left = i.p * j.q, ij_right = j.p * i.q;
					float64 const value =
						i.r * (jk_left - jk_right) + j.r * (ki_left - ki_right) + k.r * (ij_left - ij_right);
					float64 const permanent = magnitude_of(i.r) * (magnitude_of(jk_left) + magnitude_of(jk_right))
											+ magnitude_of(j.r) * (magnitude_of(ki_left) + magnitude_of(ki_right))
											+ magnitude_of(k.r) * (magnitude_of(ij_left) + magnitude_of(ij_right));
					float64 const bound = geometry::detail::halfplane_determinant_bound * permanent;
					if (value > bound || -value > bound) return int(value > 0) - int(value < 0);
					return geometry::detail::halfplane_determinant_sign_exact(i, j, k);
				} else
					return geometry::detail::halfplane_determinant_sign(i, j, k);
			}
		};

		/**
		 * @brief The library filter with its minors from a Veltkamp/Dekker
		 *        split expansion instead of Kahan's two `std::fma`, for
		 *        x86-64 at plain `-O2`, where `std::fma` is a library call. It
		 *        LOST there (geometry-halfplane_intersection, plain -O2 record of
		 *        2026-10-24, Xeon 8280, 1e6 doubles: 282 vs 192 ms tangents, 270
		 *        vs 182 around the origin, 180 vs 140 empty, 240 vs 149
		 *        presorted; the split's ~40 flops per minor cost more than two
		 *        calls).
		 * @note Runtime stores preserve the split arithmetic under FMA contraction;
		 * the cost of those stores needs a new measurement.
		 */
		struct dekker_determinant {
			/** @brief Preserve each split operation's binary64 rounding under contraction and x87. */
			[[nodiscard]] static constexpr float64 split_round(float64 const value) {
				if consteval {
					return value;
				} else {
					float64 const volatile stored = value;
					return stored;
				}
			}

			/** @brief `a b` as its rounding plus the exact error in @p err, by splitting. */
			[[nodiscard]] static constexpr float64 split_product(float64 const a, float64 const b, float64 &err) {
				constexpr float64 splitter = 0x1p27 + 1.0;
				auto const split = [](float64 const v, float64 &low) {
					float64 const c = split_round(splitter * v);
					float64 const high = split_round(c - split_round(c - v));
					low = split_round(v - high);
					return high;
				};
				float64 const product = split_round(a * b);
				float64 a_low = 0, b_low = 0;
				float64 const a_high = split(a, a_low), b_high = split(b, b_low);
				err = split_round(split_round(split_round(split_round(split_round(a_high * b_high) - product) + split_round(a_high * b_low)) + split_round(a_low * b_high)) + split_round(a_low * b_low));
				return product;
			}

			/** @brief `a b - c d` summed from its exact split expansion. */
			[[nodiscard]] static constexpr float64 minor(float64 const a, float64 const b, float64 const c, float64 const d) {
				float64 left_err = 0, right_err = 0, parts[4] = {};
				float64 const left = split_product(a, b, left_err), right = split_product(c, d, right_err);
				geometry::detail::two_two_diff(left, left_err, right, right_err, parts[3], parts[2], parts[1], parts[0]);
				return ((parts[0] + parts[1]) + parts[2]) + parts[3];
			}

			template<halfplane_coefficient Coefficient>
			[[nodiscard]] static constexpr int sign(halfplane_constraint<Coefficient> const &i,
													halfplane_constraint<Coefficient> const &j,
													halfplane_constraint<Coefficient> const &k) {
				if constexpr (std::same_as<Coefficient, float64>) {
					using geometry::detail::magnitude_of;
					float64 const term_i = i.r * minor(j.p, k.q, k.p, j.q);
					float64 const term_j = j.r * minor(k.p, i.q, i.p, k.q);
					float64 const term_k = k.r * minor(i.p, j.q, j.p, i.q);
					float64 const value = term_i + term_j + term_k;
					float64 const bound = geometry::detail::halfplane_determinant_bound
										* (magnitude_of(term_i) + magnitude_of(term_j) + magnitude_of(term_k));
					if (value > bound || -value > bound) return int(value > 0) - int(value < 0);
					return geometry::detail::halfplane_determinant_sign_exact(i, j, k);
				} else
					return geometry::detail::halfplane_determinant_sign(i, j, k);
			}
		};

		/**
		 * @brief The REFERENCE row, not a contender: the rounded `double`
		 *        determinant's sign, which can misclassify near-degenerate
		 *        triples (integral input stays exact). Library as the
		 *        geometry::fast policy of geometry::halfplane_intersection;
		 *        the row prices the exactness of the others (1e6, ms, -O2 /
		 *        native): tangents 162 / 151 vs the library 192 / 157, around
		 *        the origin 151 / 144 vs 182 / 151, empty 127 / 123 vs 140 /
		 *        128, presorted 119 / 110 vs 149 / 119.
		 */
		using geometry::detail::rounded_determinant;

		/**
		 * @brief A deque pass with the front
		 *        tested for every constraint (the library pass skips it while
		 *        the processed directions span less than `pi`).
		 *        Testing every front gives no consistent improvement (1e6, ms, -O2 / native): int64
		 *        tangents 170 / 166 vs 163 / 161, around the origin 149 / 147
		 *        vs 147 / 145, empty 128 / 125 vs 126 / 125, presorted 126 /
		 *        128 vs 122 / 125.
		 */
		struct deque_pass_testing_every_front {
			template<int Perturbation, typename Determinant, halfplane_coefficient Coefficient,
					 geometry::detail::constraint_deque<halfplane_constraint<Coefficient>> Buffer>
				requires(Perturbation >= -1 && Perturbation <= 1) && geometry::detail::determinant_policy<Determinant, Coefficient>
			[[nodiscard]] static constexpr bool run(std::vector<halfplane_constraint<Coefficient>> const &sorted, Buffer &active) {
				using geometry::detail::is_redundant;
				active.clear();
				bool is_closed = false;
				for (std::size_t t = 0; t < sorted.size(); ++t) {
					auto const &c = sorted[t];
					is_closed = is_closed || (t > 0 && geometry::detail::direction_cross_sign(sorted[0], c) <= 0);
					while (active.size() >= 2
						   && is_redundant<Perturbation, Determinant>(active[active.size() - 2], active.back(), c))
						active.pop_back();
					while (active.size() >= 2 && is_redundant<Perturbation, Determinant>(active[1], active[0], c))
						active.pop_front();
					if (is_closed && active.size() < 2) return false;
					active.push_back(c);
				}
				geometry::detail::close_deque<Perturbation, Determinant, Coefficient>(active);
				return active.size() >= 3;
			}
		};

		/**
		 * @brief The envelope split: the directions of the first
		 *        half turn (`[theta_0, theta_0 + pi)`) form one monotone chain
		 *        built by a stack straight into the deque, the second half its
		 *        own chain by a second stack; only the second chain's
		 *        survivors then run through the deque with front pops. A
		 *        constraint redundant within its half is redundant overall,
		 *        so the deque sees the same region.
		 *        LOST or tied (1e6, ms, -O2 / native): int64 tangents 170 / 168 vs 163 /
		 *        161, around the origin 143 / 140 vs 147 / 145 (its one win,
		 *        ~3%), empty 134 / 135 vs 126 / 125, presorted 128 / 132 vs 122
		 *        / 125; double tangents 201 / 162 vs 192 / 157. The second stack
		 *        saves the front tests of redundant constraints but repays it
		 *        with a second pass on non-redundant ones.
		 */
		struct envelope_pass {
			template<int Perturbation, typename Determinant, halfplane_coefficient Coefficient,
					 geometry::detail::constraint_deque<halfplane_constraint<Coefficient>> Buffer>
				requires(Perturbation >= -1 && Perturbation <= 1) && geometry::detail::determinant_policy<Determinant, Coefficient>
			[[nodiscard]] static constexpr bool run(std::vector<halfplane_constraint<Coefficient>> const &sorted, Buffer &active) {
				using geometry::detail::is_redundant;
				active.clear();
				std::size_t const m = sorted.size();
				std::size_t t = 0;
				for (; t < m && (t == 0 || geometry::detail::direction_cross_sign(sorted[0], sorted[t]) > 0); ++t) {
					while (active.size() >= 2
						   && is_redundant<Perturbation, Determinant>(active[active.size() - 2], active.back(), sorted[t]))
						active.pop_back();
					active.push_back(sorted[t]);
				}
				std::vector<halfplane_constraint<Coefficient>> second;
				second.reserve(m - t);
				for (; t < m; ++t) {
					while (second.size() >= 2
						   && is_redundant<Perturbation, Determinant>(second[second.size() - 2], second.back(), sorted[t]))
						second.pop_back();
					second.push_back(sorted[t]);
				}
				for (auto const &c : second) {
					while (active.size() >= 2
						   && is_redundant<Perturbation, Determinant>(active[active.size() - 2], active.back(), c))
						active.pop_back();
					while (active.size() >= 2 && is_redundant<Perturbation, Determinant>(active[1], active[0], c))
						active.pop_front();
					if (active.size() < 2) return false;
					active.push_back(c);
				}
				geometry::detail::close_deque<Perturbation, Determinant, Coefficient>(active);
				return active.size() >= 3;
			}
		};

		/**
		 * @brief The 64-bit form of geometry::detail::direction_key: the slope
		 *        kept as a `double` (fewer equal keys, twice the bytes to sort).
		 *        LOST to the 32-bit key (1e6, ms, -O2 / native), int64: std::sort 219 / 219
		 *        vs 218 / 225 (32-bit, std::sort) on tangents; 16-bit radix 231
		 *        / 230 vs 168 / 164 (32-bit, 16-bit radix), the four passes over
		 *        16-byte items cost more than the rarer key ties save.
		 */
		template<halfplane_coefficient Coefficient>
		[[nodiscard]] constexpr uint64 direction_key_64(halfplane_constraint<Coefficient> const &c) {
			float64 const slope = c.p == 0 ? -std::numeric_limits<float64>::infinity()
										   : float64(c.q) / float64(c.p) + 0.0; // no -0
			uint64 bits = std::bit_cast<uint64>(slope);
			bits = bits >> 63 != 0 ? ~bits : bits | uint64(1) << 63;
			return uint64(!geometry::detail::is_first_half(c)) << 63 | bits >> 1;
		}

		/**
		 * @brief Key sorts other than the library one: @p Key picks the 32- or
		 *        64-bit key, @p DigitBits the radix width (0: std::ranges::sort).
		 *        Library (32-bit key, 11-bit radix) vs these (1e6, ms, -O2 / native), int64
		 *        tangents: 163 / 161 vs 32-bit std::sort 218 / 225, 32-bit
		 *        16-bit radix 168 / 164 (a tie elsewhere: 140 / 139 vs 147 / 145
		 *        around the origin), comparator sort 233 / 234. Presorted input
		 *        skips the sort: 122 / 125 vs radix 16 143 / 144.
		 */
		template<std::unsigned_integral Key, int DigitBits, halfplane_coefficient Coefficient>
			requires(std::same_as<Key, uint32> || std::same_as<Key, uint64>) && (DigitBits == 0 || (DigitBits > 0 && DigitBits <= 16))
		constexpr void sort_and_merge_keyed(std::vector<halfplane_constraint<Coefficient>> &constraints) {
			using atl::detail::keyed_position;
			std::size_t const n = constraints.size();
			atl_assert(n <= (std::numeric_limits<uint32>::max)());
			atl_assert(n <= std::size_t((std::numeric_limits<std::ptrdiff_t>::max)()));
			std::vector<keyed_position<Key>> keys(n);
			for (std::size_t i = 0; i < n; ++i) {
				if constexpr (sizeof(Key) == 8)
					keys[i] = {direction_key_64(constraints[i]), uint32(i)};
				else
					keys[i] = {geometry::detail::direction_key(constraints[i]), uint32(i)};
			}
			if constexpr (DigitBits == 0)
				std::ranges::sort(keys, [](keyed_position<Key> const &a, keyed_position<Key> const &b) { return a.key < b.key; });
			else
				atl::detail::radix_sort<DigitBits>(keys);
			geometry::detail::settle_by_keys(constraints, keys);
		}

		template<bool UsesStdDeque, halfplane_coefficient Coefficient>
		using variant_buffer_t = std::conditional_t<UsesStdDeque, std::deque<halfplane_constraint<Coefficient>>,
													geometry::detail::cursor_deque<halfplane_constraint<Coefficient>>>;
	}

	namespace detail {
		/**
		 * @brief The corner of two adjacent deque constraints `i, j` (in deque
		 *        order) for `int64` input: `d_ij = p_i q_j - p_j q_i` and the
		 *        crossing's numerators `x = r_i q_j - r_j q_i`, `y = p_i r_j -
		 *        p_j r_i` (the crossing is `(x, y) / d_ij`). Under the
		 *        coefficient bounds `|d| < 2^63` and `|x|, |y| < 2^94`.
		 */
		struct cached_corner {
			int64 d;
			int128 x, y;
		};

		[[nodiscard]] constexpr cached_corner corner_of(halfplane_constraint<int64> const &i, halfplane_constraint<int64> const &j) {
			return {i.p * j.q - j.p * i.q, int128(i.r) * j.q - int128(j.r) * i.q, int128(i.p) * j.r - int128(j.p) * i.r};
		}

		/**
		 * @brief geometry::detail::is_redundant for an adjacent deque pair
		 *        `first, second` with its `cached_corner`, @p is_front
		 *        saying which is the keeper (`first` at the back of the deque,
		 *        `second` at the front). The determinant `r_i d_jk + r_j d_ki +
		 *        r_k d_ij` regrouped is `r_k d_ij - p_k x - q_k y`: three
		 *        `int128` multiplies instead of nine. Swapping `i` and `j`
		 *        negates the determinant, `d_ij` and the perturbation sum
		 *        alike, so the side needs no keeper order; anti-parallel pairs
		 *        (`d_ij == 0`) take the library test.
		 */
		template<int Perturbation, typename Determinant>
			requires(Perturbation >= -1 && Perturbation <= 1) && geometry::detail::determinant_policy<Determinant, int64>
		[[nodiscard]] constexpr bool is_redundant_cached(halfplane_constraint<int64> const &first,
														 halfplane_constraint<int64> const &second, cached_corner const &corner,
														 halfplane_constraint<int64> const &incoming, bool const is_front) {
			if (corner.d == 0)
				return is_front ? geometry::detail::is_redundant<Perturbation, Determinant>(second, first, incoming)
								: geometry::detail::is_redundant<Perturbation, Determinant>(first, second, incoming);
			int128 const value = int128(incoming.r) * corner.d - incoming.p * corner.x - incoming.q * corner.y;
			int side = int(value > 0) - int(value < 0);
			if constexpr (Perturbation != 0)
				if (side == 0) side = Perturbation * geometry::detail::perturbation_sign(first, second, incoming);
			return (corner.d > 0 ? side : -side) <= 0;
		}

		/**
		 * @brief geometry::detail::deque_pass with each adjacent pair's
		 *        `cached_corner` kept in a parallel deque (`corners[t]`
		 *        joins `active[t]` and `active[t + 1]`): a pair's corner is
		 *        computed once when the pair forms, so each redundancy test
		 *        costs the three multiplies of `is_redundant_cached`
		 *        instead of a fresh 3x3 determinant. The extra cache buffer
		 *        costs more on presorted input than those saved products pay.
		 *        `int64` input only (all its determinant policies
		 *        are exact, so the cached sign is theirs); `double` input runs
		 *        the library pass unchanged. Lost: presorted 1e6 int64
		 *        103 / 106 ms vs 80.6 / 83.5 (+28% / +27%), record
		 *        geometry-halfplane_intersection 2026-11-01, Xeon 8280,
		 *        -O2 / native. Combined with rotated runs it also loses:
		 *        104 / 107 ms vs 80.6 / 83.5 on the same shape.
		 */
		struct corner_cached_pass {
			/**
			 * @brief Run the corner-cached bounded pass.
			 * @return Whether the region is nonempty (with `>= 3` edges left
			 *         in @p active, in sweep order).
			 */
			template<int Perturbation, typename Determinant, halfplane_coefficient Coefficient,
					 geometry::detail::constraint_deque<halfplane_constraint<Coefficient>> Buffer>
				requires(Perturbation >= -1 && Perturbation <= 1) && geometry::detail::determinant_policy<Determinant, Coefficient>
			[[nodiscard]] static constexpr bool run(std::vector<halfplane_constraint<Coefficient>> const &sorted, Buffer &active) {
				if constexpr (!std::same_as<Coefficient, int64>)
					return geometry::detail::deque_pass::run<Perturbation, Determinant>(sorted, active);
				else {
					active.clear();
					std::size_t const m = sorted.size();
					geometry::detail::cursor_deque<cached_corner> corners(m);
					auto const pop_back = [&] {
						active.pop_back();
						corners.pop_back();
					};
					auto const pop_front = [&] {
						active.pop_front();
						corners.pop_front();
					};
					auto const push_back = [&](halfplane_constraint<int64> const &c) {
						if (active.size() > 0) corners.push_back(corner_of(active.back(), c));
						active.push_back(c);
					};
					auto const is_back_redundant = [&](halfplane_constraint<int64> const &c) {
						return is_redundant_cached<Perturbation, Determinant>(active[active.size() - 2], active.back(),
																			  corners.back(), c, false);
					};
					auto const is_front_redundant = [&](halfplane_constraint<int64> const &c) {
						return is_redundant_cached<Perturbation, Determinant>(active[0], active[1], corners.front(), c,
																			  true);
					};
					std::size_t t = 0;
					for (; t < m && (t == 0 || geometry::detail::direction_cross_sign(sorted[0], sorted[t]) > 0); ++t) {
						while (active.size() >= 2 && is_back_redundant(sorted[t])) pop_back();
						push_back(sorted[t]);
					}
					for (; t < m; ++t) {
						auto const &c = sorted[t];
						while (active.size() >= 2 && is_back_redundant(c)) pop_back();
						while (active.size() >= 2 && is_front_redundant(c)) pop_front();
						if (active.size() < 2) return false;
						push_back(c);
					}
					for (bool has_changed = true; has_changed;) {
						has_changed = false;
						while (active.size() >= 3 && is_back_redundant(active.front())) {
							pop_back();
							has_changed = true;
						}
						while (active.size() >= 3 && is_front_redundant(active.back())) {
							pop_front();
							has_changed = true;
						}
					}
					return active.size() >= 3;
				}
			}
		};
	}

	/**
	 * @brief Direction-sort choices for compositions; rotated-run dispatch
	 * avoids full key sorting on up to eight ascending runs (see the guide).
	 * Key sort versus run dispatch (2026-11-01, Xeon 8280, -O2 / native, ms):
	 * rotated run 1e6 int64 97.2 / 99.5 versus 82.3 / 83.7; hull edges
	 * 28.8 / 29.2 versus 25.4 / 26.3.
	 */
	enum class halfplane_sort {
		key,			 ///<  32-bit key, 11-bit radix (comparison below 2048)
		comparator,		 ///< the exact comparator on whole constraints
		key_std_sort,	 ///< 32-bit key, std::ranges::sort
		key_radix_16,	 ///< 32-bit key, 16-bit radix (two passes)
		key_64_std_sort, ///< 64-bit key (the slope as a double), std::ranges::sort
		key_64_radix_16, ///< 64-bit key, 16-bit radix (four passes)
		rotated_runs	 ///< library dispatch; retained for the combined corner-cache loser
	};

	/**
	 * @brief Experimental compositions of the determinant, pass and sort: @p Determinant
	 *        (`geometry::detail::filtered_determinant` is the library default), @p Pass
	 *        (`geometry::detail::deque_pass` is the library default), @p Sort, and the deque
	 *        buffer (`geometry::detail::cursor_deque` is the library default; @p UsesStdDeque
	 *        swaps in `std::deque`, a segmented buffer: LOST
	 *        (1e6, ms, -O2 / native), int64 tangents 184 / 183 vs 163 / 161, presorted
	 *        141 / 145 vs 122 / 125, within noise on the other two).
	 *        Constant-evaluable except with @p UsesStdDeque (`std::deque`
	 *        is not constexpr).
	 * @tparam Determinant Static coefficient determinant-sign policy.
	 * @tparam Pass Bounded deque-pass policy.
	 * @tparam Sort Direction-sort method.
	 * @tparam UsesStdDeque Whether to use standard deque storage.
	 * @tparam R Const-readable half-plane range.
	 * @param halfplanes Input constraints, not modified.
	 * @return The region using the library result layout.
	 * @pre Coefficient, size and floating arithmetic bounds as halfplane_intersection.
	 * @complexity O(n log n) time and O(n) memory.
	 */
	template<typename Determinant, typename Pass, halfplane_sort Sort, bool UsesStdDeque = false,
			 geometry::halfplane_range R>
		requires geometry::detail::bounded_pass_policy<
			Pass, Determinant, geometry::halfplane_coefficient_t<geometry::halfplane_coordinate_t<R>>,
			detail::variant_buffer_t<UsesStdDeque, geometry::halfplane_coefficient_t<geometry::halfplane_coordinate_t<R>>>>
	[[nodiscard]] constexpr geometry::halfplane_region halfplane_intersection_by_composition(R const &halfplanes) {
		using Coefficient = geometry::halfplane_coefficient_t<geometry::halfplane_coordinate_t<R>>;
		auto constraints = geometry::detail::constraints_of(halfplanes);
		if constexpr (Sort == halfplane_sort::key)
			geometry::detail::sort_and_merge_by_key(constraints);
		else if constexpr (Sort == halfplane_sort::comparator)
			geometry::detail::sort_and_merge_by_comparator(constraints);
		else if constexpr (Sort == halfplane_sort::key_std_sort)
			detail::sort_and_merge_keyed<uint32, 0>(constraints);
		else if constexpr (Sort == halfplane_sort::key_radix_16)
			detail::sort_and_merge_keyed<uint32, 16>(constraints);
		else if constexpr (Sort == halfplane_sort::key_64_std_sort)
			detail::sort_and_merge_keyed<uint64, 0>(constraints);
		else if constexpr (Sort == halfplane_sort::key_64_radix_16)
			detail::sort_and_merge_keyed<uint64, 16>(constraints);
		else if constexpr (std::integral<geometry::halfplane_coordinate_t<R>>)
			geometry::detail::sort_and_merge_rotated_runs(constraints);
		else
			geometry::detail::sort_and_merge_by_key(constraints);
		if constexpr (UsesStdDeque) {
			detail::variant_buffer_t<true, Coefficient> active;
			return geometry::detail::region_of_sorted<Determinant, Pass>(constraints, active);
		} else {
			detail::variant_buffer_t<false, Coefficient> active(constraints.size());
			return geometry::detail::region_of_sorted<Determinant, Pass>(constraints, active);
		}
	}
}
