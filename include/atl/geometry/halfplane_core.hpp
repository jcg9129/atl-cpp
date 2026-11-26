#pragma once

/** @file
 * @brief Shared halfplane implementation without adaptive floating predicates.
 */
#include <atl/geometry/line_core.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon_core.hpp>
#include <atl/geometry/predicates_core.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>

namespace atl::geometry {
	/**
	 * @brief Type of the coefficients of a @ref atl::geometry::halfplane over coordinate @p T:
	 *        `int64` for every integral coordinate (the asserted bounds below
	 *        keep each coefficient inside it), `float64` for floating ones.
	 * @tparam T Input coordinate type.
	 */
	template<coordinate T>
	using halfplane_coefficient_t = std::conditional_t<std::floating_point<T>, float64, int64>;

	namespace detail {
		template<typename Coefficient>
		concept halfplane_coefficient = std::same_as<Coefficient, int64> || std::same_as<Coefficient, float64>;

		/**
		 * @name Integral coefficient bounds
		 * @brief `|p|, |q| < 2^31` and `|r| < 2^62`. Every exact decision of
		 *        @ref atl::geometry::halfplane_intersection is the sign of a 3x3
		 *        determinant of three constraints' coefficients, evaluated
		 *        as `r_i d_jk + r_j d_ki + r_k d_ij` with the 2x2 normal
		 *        determinants `d_jk = p_j q_k - p_k q_j`: each `|d| <= 2 (2^31
		 *        - 1)^2 < 2^63` fits `int64`, each product `|r d| < 2^125`, the
		 *        sum `< 3 * 2^125 < 2^127` fits `int128`. Points `|x|, |y| <
		 *        2^30` give `|p|, |q| <= 2^31 - 2` and `|r| < 2^61` through
		 *        @ref atl::geometry::halfplane::left_of, inside both bounds.
		 * @{
		 */
		inline constexpr int64 halfplane_normal_bound = int64(1) << 31;
		inline constexpr int64 halfplane_offset_bound = int64(1) << 62;
		inline constexpr int64 halfplane_point_bound = int64(1) << 30;

		/** @} */

		/**
		 * @brief Whether the coefficients describe a proper half-plane inside
		 *        the coefficient bounds: `(p, q) != (0, 0)`, and for integral
		 *        coefficients the bounds above; for floating ones finite values.
		 */
		template<halfplane_coefficient Coefficient>
		[[nodiscard]] constexpr bool is_valid_halfplane(Coefficient const p, Coefficient const q, Coefficient const r) {
			if (p == Coefficient(0) && q == Coefficient(0)) return false;
			if constexpr (std::same_as<Coefficient, int64>) {
				return -halfplane_normal_bound < p && p < halfplane_normal_bound
					&& -halfplane_normal_bound < q && q < halfplane_normal_bound
					&& -halfplane_offset_bound < r && r < halfplane_offset_bound;
			} else {
				constexpr float64 largest = (std::numeric_limits<float64>::max)();
				auto const finite = [&](float64 const v) { return -largest <= v && v <= largest; };
				return finite(p) && finite(q) && finite(r);
			}
		}

		template<typename T>
		struct halfplane_arithmetic {
			static_assert(sizeof(T) == 0, "Include <atl/geometry/halfplane.hpp> before using floating half-planes");
		};

	}

	/**
	 * @brief The CLOSED half-plane `p x + q y <= r` — the input of
	 *        @ref atl::geometry::halfplane_intersection. Aggregate over its coefficients;
	 *        the two factories check the coefficient bounds, as do
	 *        the intersection and containment algorithms at entry (an aggregate can be filled directly).
	 *        `==` is representational: `2x <= 2` and `x <= 1` are the same
	 *        set but different values.
	 *
	 * Integral coordinates store `int64` coefficients with `|p|, |q| <
	 * 2^31`, `|r| < 2^62` (the determinant bounds explain why
	 * these keep every decision exact in `int128`). Floating coordinates
	 * store `double` coefficients and every decision is exact WITH RESPECT
	 * TO THEM.
	 * @tparam T Coordinate type; coefficients widen independently of T.
	 */
	template<coordinate T>
	struct halfplane {
		/** @brief Input coordinate type. */
		using coordinate_type = T;
		/** @brief Stored coefficient type, independent of the input integer width. */
		using coefficient_type = halfplane_coefficient_t<T>;

		/** @brief The normal's `x` component: the coefficient of `x`. */
		coefficient_type p{};
		/** @brief The normal's `y` component: the coefficient of `y`. */
		coefficient_type q{};
		/** @brief The offset: the right-hand side of `p x + q y <= r`. */
		coefficient_type r{};

		/**
		 * @brief The half-plane on the LEFT of the directed line `a -> b`,
		 *        boundary included: `p = b.y - a.y`, `q = a.x - b.x`,
		 *        `r = a.x b.y - a.y b.x`. Exact for integral coordinates;
		 *        for floating ones the coefficients are rounded constructions
		 *        (exact for integer-valued coordinates of magnitude below `2^26`).
		 * @param a Start of the directed boundary.
		 * @param b Distinct second point on the boundary.
		 * @return The closed left half-plane.
		 * @complexity O(1) time and space.
		 * @pre Floating products and coefficient constructions must remain finite
		 * and normal (zero-factor products allowed).
		 * @pre `a != b`; integral: `|x|, |y| < 2^30` (both asserted).
		 */
		[[nodiscard]] static constexpr halfplane left_of(point<T> const a, point<T> const b) {
			atl_assert(a != b);
			if constexpr (std::floating_point<T>) {
				float64 const ax = a.x, ay = a.y, bx = b.x, by = b.y;
				// Compensate the second product; the coefficient remains a construction.
				float64 err = 0;
				float64 const ay_bx = detail::halfplane_arithmetic<T>::product(ay, bx, err);
				float64 const ax_by = ax * by - ay_bx;
				halfplane const h{by - ay, ax - bx, ax_by - err};
				atl_assert(detail::is_valid_halfplane(h.p, h.q, h.r));
				return h;
			} else {
				[[maybe_unused]] auto const in_bounds = [](point<T> const v) {
					return -detail::halfplane_point_bound < int64(v.x) && int64(v.x) < detail::halfplane_point_bound
						&& -detail::halfplane_point_bound < int64(v.y) && int64(v.y) < detail::halfplane_point_bound;
				};
				atl_assert(in_bounds(a) && in_bounds(b));
				int64 const ax = a.x, ay = a.y, bx = b.x, by = b.y;
				return {by - ay, ax - bx, ax * by - ay * bx};
			}
		}

		/**
		 * @brief The half-plane `p x + q y <= r`.
		 * @param p Normal's x coefficient.
		 * @param q Normal's y coefficient.
		 * @param r Offset.
		 * @return The validated coefficient aggregate.
		 * @complexity O(1) time and space.
		 * @pre `(p, q) != (0, 0)`; integral coordinates: `|p|, |q| < 2^31`,
		 *      `|r| < 2^62`; floating: finite values (all asserted).
		 */
		[[nodiscard]] static constexpr halfplane at_most(coefficient_type const p, coefficient_type const q,
														 coefficient_type const r) {
			atl_assert(detail::is_valid_halfplane(p, q, r));
			return {p, q, r};
		}

		/**
		 * @brief Where @p v sits: inside (`p x + q y < r`), on the boundary,
		 *        or outside — EXACT (an `int128` residual for integral
		 *        coordinates, any `int64` point; a filtered expansion for
		 *        floating ones). Same as the free
		 *        `contains(h, v)` under the default @ref atl::geometry::exact policy.
		 * @param v Query point.
		 * @return Inside, boundary or outside according to the residual sign.
		 * @complexity O(1) time and space.
		 * @pre The coefficient bounds of @ref atl::geometry::halfplane::at_most (asserted); floating:
		 *      no overflow or underflow in the products `p x`, `q y`; all expansion
		 *      intermediates and `|r| + |p x| + |q y|` must be finite
		 *      (the absolute sum and rounded residual are asserted).
		 */
		[[nodiscard]] constexpr containment contains(point<T> const v) const {
			atl_assert(detail::is_valid_halfplane(p, q, r));
			if constexpr (std::floating_point<T>)
				return containment(detail::halfplane_arithmetic<T>::residual(p, q, r, float64(v.x), float64(v.y)));
			else
				return containment(detail::sign_of(int128(r) - int128(p) * v.x - int128(q) * v.y));
		}

		/**
		 * @brief The boundary line `p x + q y = r`, directed so the interior
		 *        is on its LEFT — a `double` construction (an integral
		 *        half-plane's boundary need not pass through any integer
		 *        point). In the convention of @ref atl::geometry::line_from_coefficients
		 *        (`A x + B y + C = 0`, positive side on the left) the
		 *        coefficients are `(A, B, C) = (-p, -q, r)`: the interior
		 *        `p x + q y <= r` is `A x + B y + C >= 0`.
		 * @return A constructed directed supporting line.
		 * @complexity O(1) time and space.
		 * @pre Nonzero normal (asserted by line construction); floating coefficient
		 * products must satisfy line_from_coefficients' finite normal-product bounds.
		 * Unlike intersection, boundary construction accepts full-range int64
		 * coefficients: negation widens first. Endpoints must remain finite.
		 */
		[[nodiscard]] constexpr line<float64> boundary() const {
			if constexpr (std::floating_point<T>)
				return line_from_coefficients(std::array{-p, -q, r});
			else
				return line_from_coefficients(std::array{-int128(p), -int128(q), int128(r)});
		}

		/** @brief Equality of stored coefficients, without geometric normalization. */
		[[nodiscard]] friend constexpr bool operator==(halfplane const &, halfplane const &) = default;
	};

	namespace detail {
		template<typename H>
		struct is_halfplane : std::false_type {};

		template<coordinate T>
		struct is_halfplane<halfplane<T>> : std::true_type {};

		template<typename H, typename Reference>
		concept readable_halfplane = is_halfplane<H>::value
								  && (std::convertible_to<Reference, H> || requires(Reference const &h) {
										 { h.p } -> std::convertible_to<typename H::coefficient_type>;
										 { h.q } -> std::convertible_to<typename H::coefficient_type>;
										 { h.r } -> std::convertible_to<typename H::coefficient_type>;
									 });

		// Keep member-readable proxies as well as proxies convertible to the value type.
		template<coordinate T, typename Reference>
			requires readable_halfplane<halfplane<T>, Reference>
		[[nodiscard]] constexpr halfplane<T> materialize_halfplane(Reference &&h) {
			if constexpr (std::convertible_to<Reference, halfplane<T>>)
				return std::forward<Reference>(h);
			else {
				using C = typename halfplane<T>::coefficient_type;
				return {C(h.p), C(h.q), C(h.r)};
			}
		}

	}

	/**
	 * @brief Where @p v sits relative to the closed half-plane @p h, under
	 *        the sign policy @p p — the free form every geometry shape has
	 *        (`contains(polygon, q, p)`).
	 *
	 * @ref atl::geometry::exact (the default) is @ref atl::geometry::halfplane::contains : an `int128`
	 * residual for integral coordinates, a filtered expansion for floating
	 * ones. Any other policy decides with its `sign` on the residual `r - p
	 * x - q y`: in `int128` for integral coordinates (so @ref atl::geometry::fast stays
	 * exact there), rounded to `double` for floating ones.
	 * @tparam T Coordinate type.
	 * @tparam Policy Sign policy accepting the widened residual (int128 for integral T).
	 * @param h Half-plane.
	 * @param v Query point.
	 * @param p Sign policy.
	 * @return Inside, boundary or outside.
	 * @complexity O(1) time and space.
	 * @pre The coefficient bounds of @ref atl::geometry::halfplane::at_most (asserted);
	 *      floating: no overflow or underflow in the products `p x`, `q y`.
	 *      Under exact, expansion intermediates and `|r| + |p x| + |q y|`
	 *      must remain finite (absolute sum and rounded residual asserted).
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
		requires requires(Policy const &policy, std::conditional_t<std::floating_point<T>, float64, int128> residual) {
			{ policy.sign(residual) } -> std::same_as<int>;
		}
	[[nodiscard]] constexpr containment contains(halfplane<T> const &h, point<T> const v, Policy const p = {}) {
		if constexpr (std::same_as<Policy, exact>)
			return h.contains(v);
		else {
			atl_assert(detail::is_valid_halfplane(h.p, h.q, h.r));
			if constexpr (std::floating_point<T>)
				return containment(p.sign(h.r - h.p * float64(v.x) - h.q * float64(v.y)));
			else
				return containment(p.sign(int128(h.r) - int128(h.p) * v.x - int128(h.q) * v.y));
		}
	}

	/** @brief A sized random-access range of @ref atl::geometry::halfplane values.
	 * @tparam R Range readable through const, with convertible or member-readable references.
	 */
	template<typename R>
	concept halfplane_range = std::ranges::random_access_range<R> && std::ranges::sized_range<R>
						   && detail::is_halfplane<std::remove_cvref_t<std::ranges::range_value_t<R>>>::value
						   && std::ranges::random_access_range<R const> && std::ranges::sized_range<R const>
						   && detail::readable_halfplane<std::ranges::range_value_t<R>, std::ranges::range_reference_t<R const>>;

	/** @brief Coordinate type of the half-planes of a @ref atl::geometry::halfplane_range.
	 * @tparam R Half-plane range.
	 */
	template<halfplane_range R>
	using halfplane_coordinate_t = typename std::remove_cvref_t<std::ranges::range_value_t<R>>::coordinate_type;
}
