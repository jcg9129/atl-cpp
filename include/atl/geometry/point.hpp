#pragma once
#include <atl/config.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/traits/widen_mul.hpp>
#include <atl/utils/assert.hpp>
#include <cmath>
#include <compare>
#include <concepts>
#include <limits>
#include <ranges>
#include <type_traits>

namespace atl::geometry {
	namespace detail {
		template<typename T>
		inline constexpr bool is_character = std::same_as<std::remove_cv_t<T>, char>
										  || std::same_as<std::remove_cv_t<T>, signed char> || std::same_as<std::remove_cv_t<T>, unsigned char>
										  || std::same_as<std::remove_cv_t<T>, wchar_t> || std::same_as<std::remove_cv_t<T>, char8_t>
										  || std::same_as<std::remove_cv_t<T>, char16_t> || std::same_as<std::remove_cv_t<T>, char32_t>;
	}

	/**
	 * @brief A type usable as a 2D coordinate: a signed integer of at most 64
	 *        bits excluding character types, or `float`/`double`.
	 *
	 * Unsigned integers are excluded because every geometric quantity built
	 * from coordinates (cross products, orientations, differences) is signed.
	 * `long double` is excluded because its format differs by target (80-bit
	 * x87 on x86 Linux, binary128 on AArch64 Linux, plain `double` on MSVC and
	 * Apple), so no exactness argument would be portable; the exact `double`
	 * predicates make it unnecessary. 128-bit integers are excluded: their
	 * products have no wider type.
	 */
	template<typename T>
	concept coordinate =
		(std::signed_integral<T> && sizeof(T) <= 8 && !detail::is_character<T>)
		|| std::same_as<T, float> || std::same_as<T, double>;

	namespace detail {
		/**
		 * @brief Same-size fixed-width alias for any signed integral: `long` and
		 * `long long` are distinct 8-byte types but only one of them is
		 * `int64`, and the widening trait keys on the aliases.
		 */
		template<std::signed_integral T>
		using normalized_int = std::conditional_t<
			sizeof(T) == 1, int8,
			std::conditional_t<sizeof(T) == 2, int16, std::conditional_t<sizeof(T) == 4, int32, int64>>>;

		/** @brief Result type of one product of raw coordinates (see @ref atl::geometry::product_t). */
		template<coordinate T>
		struct product_type {
			using type = traits::widen_mul_type_t<normalized_int<T>>;
		};

		/**
		 * @brief `float` products widen to `double`, where each product is
		 *        exact (24-bit mantissas give 48-bit products).
		 */
		template<>
		struct product_type<float> {
			using type = float64;
		};

		/** @brief `double` products stay `double`: each product is rounded. */
		template<>
		struct product_type<double> {
			using type = float64;
		};
	}

	/**
	 * @brief Type holding one product of raw coordinates of type @p T:
	 *        the integer of twice the width for integral coordinates
	 *        (`int16 -> int32`, `int32 -> int64`,
	 *        `int64 -> int128`), `double` for floating ones.
	 *
	 * For integral @p T every value of this type computed by the two-point
	 * @ref atl::geometry::cross and by @ref atl::geometry::dot is exact. For `float` each single product is
	 * exact in `double`, but a sum or difference of two of them is rounded
	 * once; for `double` every product and the final sum are rounded. Use the
	 * predicates for certified signs.
	 */
	template<coordinate T>
	using product_t = typename detail::product_type<T>::type;

	/**
	 * @brief Type holding a product of coordinate DIFFERENCES: one more
	 *        widening step than @ref atl::geometry::product_t, since a difference needs one
	 *        extra bit (`int16 -> int64`, `int32` and
	 *        `int64 -> int128`, `double` for floating coordinates). This is
	 *        the result type of the three-point @ref atl::geometry::cross and of
	 *        @ref atl::geometry::squared_distance.
	 *
	 * For `int64` coordinates no wider type than `int128` exists, so those
	 * operations carry the magnitude precondition `|x|, |y| < 2^62`
	 * (asserted). Narrower integral coordinates have no bound.
	 */
	template<coordinate T>
	using difference_product_t =
		std::conditional_t<std::floating_point<T>, float64, traits::widen_mul_type_t<product_t<T>>>;

	/**
	 * @brief A 2D point, equally a 2D vector (the library does not
	 *        distinguish). Aggregate; `point{x, y}` deduces `T`.
	 *
	 * Comparison is lexicographic (`x`, then `y`), the sweep order. The
	 * arithmetic operators work in @p T and overflow integral coordinates
	 * near the type's limits like any other @p T arithmetic; the predicates
	 * and the widened value functions take the original points and widen
	 * internally, so prefer `orientation(o, a, b)` over `cross(a - o, b - o)`.
	 */
	template<coordinate T>
	struct point {
		/** @brief The coordinate type @p T. */
		using coordinate_type = T;

		T x{}; ///< Abscissa.
		T y{}; ///< Ordinate.

		/** @brief Lexicographic order: by `x`, then by `y`. */
		friend constexpr auto operator<=>(point const &, point const &) = default;

		/** @brief Adds @p o coordinatewise, in @p T. */
		constexpr point &operator+=(point const o) {
			x = T(x + o.x);
			y = T(y + o.y);
			return *this;
		}

		/** @brief Subtracts @p o coordinatewise, in @p T. */
		constexpr point &operator-=(point const o) {
			x = T(x - o.x);
			y = T(y - o.y);
			return *this;
		}

		/** @brief Scales both coordinates by @p k, in @p T. */
		constexpr point &operator*=(T const k) {
			x = T(x * k);
			y = T(y * k);
			return *this;
		}

		/** @brief Coordinatewise sum, in @p T. */
		[[nodiscard]] friend constexpr point operator+(point a, point const b) { return a += b; }

		/** @brief Coordinatewise difference, in @p T. */
		[[nodiscard]] friend constexpr point operator-(point a, point const b) { return a -= b; }

		/** @brief Both coordinates negated, in @p T. */
		[[nodiscard]] friend constexpr point operator-(point const a) { return {T(-a.x), T(-a.y)}; }

		/** @brief Both coordinates scaled by @p k, in @p T. */
		[[nodiscard]] friend constexpr point operator*(point a, T const k) { return a *= k; }

		/** @brief Both coordinates scaled by @p k, in @p T. */
		[[nodiscard]] friend constexpr point operator*(T const k, point a) { return a *= k; }
	};

	/**
	 * @brief @p v rotated 90 degrees counterclockwise, `(-y, x)`, in @p T: the
	 * direction whose extreme point maximizes `cross(v, .)`.
	 */
	template<coordinate T>
	[[nodiscard]] constexpr point<T> perp(point<T> const v) {
		return {T(-v.y), T(v.x)};
	}

	namespace detail {
		/** @brief Whether @p P is some `point<T>` (see @ref atl::geometry::vertex_range). */
		template<typename P>
		struct is_point : std::false_type {};

		template<coordinate T>
		struct is_point<point<T>> : std::true_type {};
	}

	/**
	 * @brief A sized random-access range of `point<T>`: how the library
	 *        spells "polygon" (vertices in order, closing edge implicit,
	 *        either orientation) and any other plain point sequence. A
	 *        `std::vector<point<T>>` qualifies; there is no polygon class
	 *        because a vertex list carries no invariant to protect.
	 */
	template<typename R>
	concept vertex_range =
		std::ranges::random_access_range<R> && std::ranges::sized_range<R>
		&& detail::is_point<std::remove_cvref_t<std::ranges::range_value_t<R>>>::value;

	/**
	 * @brief Index of a point within a @ref atl::geometry::vertex_range, the element type of
	 * index-valued results such as @ref atl::geometry::polar_angle_order (positional, unlike
	 * `graph::vertex_id`, which names a node identity).
	 */
	using vertex_index = uint32;

	/** @brief Index of a triangle within a triangulation result. */
	using triangle_index = uint32;

	/** @brief Coordinate type of the points of a @ref atl::geometry::vertex_range. */
	template<vertex_range R>
	using vertex_coordinate_t =
		typename std::remove_cvref_t<std::ranges::range_value_t<R>>::coordinate_type;

	namespace detail {
		/** @brief -1, 0 or +1 by comparison with zero. */
		template<typename V>
			requires std::constructible_from<V, int> && std::totally_ordered<V>
		constexpr int sign_of(V const v) {
			return int(v > V(0)) - int(v < V(0));
		}

		/**
		 * @brief For 64-bit coordinates, whose difference products only just
		 * fit `int128`: every coordinate must satisfy `|c| < 2^62`.
		 */
		template<coordinate T, std::same_as<point<T>>... Pts>
		constexpr void check_difference_bounds([[maybe_unused]] point<T> const p,
											   [[maybe_unused]] Pts const... pts) {
			if constexpr (std::signed_integral<T> && sizeof(T) == 8) {
				[[maybe_unused]] constexpr T bound = T(1) << 62;
				[[maybe_unused]] auto const ok = [&](point<T> const q) {
					return -bound < q.x && q.x < bound && -bound < q.y && q.y < bound;
				};
				atl_assert(ok(p) && (ok(pts) && ...));
			}
		}

		/** @brief Whether a product is finite and normal, or is zero from a zero factor. */
		constexpr bool normal_product(float64 const a, float64 const b) {
			float64 const v = a * b;
			float64 const magnitude = v < 0 ? -v : v;
			return (a == 0 || b == 0) ? v == 0
									  : (std::numeric_limits<float64>::min)() <= magnitude && magnitude <= (std::numeric_limits<float64>::max)();
		}

		/** @brief Cheap entry check for floating coordinate squares; zero coordinates are allowed. */
		template<coordinate T>
		constexpr void check_floating_magnitude([[maybe_unused]] point<T> const p) {
			if constexpr (std::floating_point<T>) {
				atl_assert(normal_product(float64(p.x), float64(p.x)));
				atl_assert(normal_product(float64(p.y), float64(p.y)));
			}
		}

		/** @brief `a - b` computed and stored in the wider type @p D. */
		template<typename D, coordinate T>
		constexpr D widened_difference(T const a, T const b) {
			return D(D(a) - D(b));
		}
	}

	/**
	 * @brief Cross product `a.x * b.y - a.y * b.x` in @ref atl::geometry::product_t — exact for
	 *        integral coordinates; for `float` the two products are exact and
	 *        their difference is rounded once, so the sign is still exact; for
	 *        `double` a rounded construction with no sign guarantee (for a
	 *        certified sign use @ref atl::geometry::orientation).
	 *
	 * Total for every integral type: the extreme value, `2^(2w-1) - 2^(w-1)`
	 * for `w`-bit coordinates (`2^127 - 2^63` for `int64`), fits.
	 * @complexity O(1)
	 */
	template<coordinate T>
	[[nodiscard]] constexpr product_t<T> cross(point<T> const a, point<T> const b) {
		using P = product_t<T>;
		return P(P(a.x) * P(b.y) - P(a.y) * P(b.x));
	}

	/**
	 * @brief Dot product `a.x * b.x + a.y * b.y` in @ref atl::geometry::product_t, with the
	 *        exactness tiers of the two-point @ref atl::geometry::cross.
	 * @pre For integral coordinates: not all four coordinates equal the
	 *      type's minimum (the one input whose result, `2^(2w-1)` for
	 *      `w`-bit coordinates, overflows @ref atl::geometry::product_t; asserted).
	 * @complexity O(1)
	 */
	template<coordinate T>
	[[nodiscard]] constexpr product_t<T> dot(point<T> const a, point<T> const b) {
		using P = product_t<T>;
		if constexpr (std::signed_integral<T>) {
			[[maybe_unused]] constexpr T min = (std::numeric_limits<T>::min)();
			atl_assert(a.x != min || a.y != min || b.x != min || b.y != min);
		}
		return P(P(a.x) * P(b.x) + P(a.y) * P(b.y));
	}

	/**
	 * @brief Cross product of `a - o` and `b - o`, twice the signed area of
	 *        the triangle `o, a, b`, in @ref atl::geometry::difference_product_t. Integral
	 *        results are exact: the differences and products are widened.
	 *        For floating coordinates a rounded construction (differences,
	 *        products and the final difference are each rounded, so it can
	 *        cancel catastrophically); its sign near zero is what
	 *        @ref atl::geometry::orientation certifies.
	 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted).
	 * @complexity O(1)
	 */
	template<coordinate T>
	[[nodiscard]] constexpr difference_product_t<T> cross(point<T> const o, point<T> const a, point<T> const b) {
		if constexpr (std::floating_point<T>) {
			using F = float64;
			return (F(a.x) - F(o.x)) * (F(b.y) - F(o.y)) - (F(a.y) - F(o.y)) * (F(b.x) - F(o.x));
		} else {
			using D = product_t<T>;
			using P = difference_product_t<T>;
			detail::check_difference_bounds(o, a, b);
			D const acx = detail::widened_difference<D>(a.x, o.x), acy = detail::widened_difference<D>(a.y, o.y);
			D const bcx = detail::widened_difference<D>(b.x, o.x), bcy = detail::widened_difference<D>(b.y, o.y);
			return P(P(acx) * P(bcy) - P(acy) * P(bcx));
		}
	}

	/**
	 * @brief Squared euclidean length of @p v in @ref atl::geometry::product_t, exact for
	 * integral coordinates.
	 * @pre As @ref atl::geometry::dot — for integral coordinates, not both coordinates equal
	 *      the type's minimum (asserted).
	 * @complexity O(1)
	 */
	template<coordinate T>
	[[nodiscard]] constexpr product_t<T> squared_length(point<T> const v) {
		return dot(v, v);
	}

	/**
	 * @brief Squared euclidean distance between @p a and @p b in
	 *        @ref atl::geometry::difference_product_t, computed on widened differences:
	 *        exact for integral coordinates.
	 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted).
	 * @complexity O(1)
	 */
	template<coordinate T>
	[[nodiscard]] constexpr difference_product_t<T> squared_distance(point<T> const a, point<T> const b) {
		if constexpr (std::floating_point<T>) {
			using F = float64;
			F const dx = F(a.x) - F(b.x), dy = F(a.y) - F(b.y);
			return dx * dx + dy * dy;
		} else {
			using D = product_t<T>;
			using P = difference_product_t<T>;
			detail::check_difference_bounds(a, b);
			D const dx = detail::widened_difference<D>(a.x, b.x), dy = detail::widened_difference<D>(a.y, b.y);
			return P(P(dx) * P(dx) + P(dy) * P(dy));
		}
	}

	namespace detail {
		/**
		 * @brief The vector `to - from` as `double`s, differenced BEFORE
		 *        rounding: integral coordinates subtract exactly in the
		 *        widened type and round once, so two far-away points that
		 *        are close together keep their offset. Converting each
		 *        coordinate first loses everything below the magnitude's
		 *        ulp (512 at `2^61`) and can cancel the difference to zero.
		 *
		 * The entry point for every `double` construction that starts from
		 * a coordinate difference.
		 */
		template<coordinate T>
		constexpr point<float64> float64_difference(point<T> const from, point<T> const to) {
			if constexpr (std::floating_point<T>) {
				return {float64(to.x) - float64(from.x), float64(to.y) - float64(from.y)};
			} else {
				using D = product_t<T>;
				return {float64(widened_difference<D>(to.x, from.x)), float64(widened_difference<D>(to.y, from.y))};
			}
		}
	}

	/**
	 * @brief @p p as a `point<float64>`, each coordinate converted on its own.
	 *
	 * Exact for `float` and `double` coordinates and for integral ones with
	 * `|c| <= 2^53`; larger integers round once per coordinate, making the
	 * result a construction. To convert a coordinate difference, subtract in
	 * the widened type first.
	 * @complexity O(1)
	 */
	template<coordinate T>
	[[nodiscard]] constexpr point<float64> to_float64(point<T> const p) {
		return {float64(p.x), float64(p.y)};
	}

	/**
	 * @brief Euclidean length of @p v as `double`, a construction (rounded):
	 * compare @ref atl::geometry::squared_length values instead in exact decisions.
	 * @pre As @ref atl::geometry::squared_length.
	 * @note Not constexpr (`std::sqrt`).
	 */
	template<coordinate T>
	[[nodiscard]] float64 length(point<T> const v) {
		return std::sqrt(float64(squared_length(v)));
	}

	/**
	 * @brief Euclidean distance between @p a and @p b as `double`, a
	 * construction: compare @ref atl::geometry::squared_distance values in exact decisions.
	 * @pre As @ref atl::geometry::squared_distance.
	 * @note Not constexpr (`std::sqrt`).
	 */
	template<coordinate T>
	[[nodiscard]] float64 distance(point<T> const a, point<T> const b) {
		return std::sqrt(float64(squared_distance(a, b)));
	}
}
