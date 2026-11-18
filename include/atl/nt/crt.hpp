#pragma once

/**
 * @file
 * @brief Linear congruences and the Chinese remainder theorem: solve
 *        `a·x ≡ c (mod m)`, intersect congruences with arbitrary (not necessarily
 *        coprime) moduli, and reconstruct a residue modulo a target from
 *        pairwise-coprime residues (Garner's mixed-radix algorithm).
 *
 * Reference: "Applications of modular linear equations" (2003).
 */

#include <atl/nt/gcd.hpp>
#include <atl/nt/modular.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <ranges>
#include <vector>

namespace atl {
	/**
	 * @brief The residue class `x ≡ residue (mod modulus)`, with
	 *        `0 <= residue < modulus`. `{0, 1}` is the class of every integer.
	 * @tparam U Unsigned integral residue and modulus type, including atl::uint128.
	 */
	template<unsigned_integral U>
	struct congruence {
		U residue; ///< Canonical residue, below modulus.
		U modulus; ///< Positive modulus.

		/** @brief Compare both canonical fields. */
		[[nodiscard]] constexpr bool operator==(congruence const &) const = default;
	};

	namespace detail {
		template<typename T>
		inline constexpr bool is_congruence_v = false;

		template<unsigned_integral U>
		inline constexpr bool is_congruence_v<congruence<U>> = true;
	}

	/**
	 * @brief An input range of `congruence<U>` (for one unsigned `U`), the systems
	 *        @ref atl::chinese_remainder and @ref atl::chinese_remainder_mod take.
	 * @tparam R Input range with values `congruence<U>` and references convertible to values.
	 */
	template<typename R>
	concept congruence_range =
		std::ranges::input_range<R> && detail::is_congruence_v<std::ranges::range_value_t<R>>
		&& std::convertible_to<std::ranges::range_reference_t<R>, std::ranges::range_value_t<R>>;

	namespace detail {
		/** @brief Asserts that @p c is a well-formed class: modulus >= 1, residue below it. */
		template<unsigned_integral U>
		constexpr void assert_congruence(congruence<U> const &c) {
			atl_assert(c.modulus >= 1);
			atl_assert(c.residue < c.modulus);
		}
	}

	/**
	 * @brief Solves `a·x ≡ c (mod m)`.
	 *
	 * With `g = gcd(a, m)` it is solvable iff `g` divides `c`, and then the solutions
	 * form one class modulo `m / g`: `x ≡ (c/g)·(a/g)⁻¹ (mod m/g)`.
	 *
	 * @tparam Coefficient Integral coefficient type, including signed and wide integers.
	 * @tparam Constant Integral right-hand-side type.
	 * @tparam Modulus Unsigned integral modulus type.
	 * @param a Coefficient (any integer; reduced mod m).
	 * @param c Right-hand side (any integer; reduced mod m).
	 * @param m Modulus.
	 * @return The solution class `{x0, m / g}` with `0 <= x0 < m / g`, or
	 *         `std::nullopt` when `g` does not divide `c`. `a ≡ c ≡ 0` gives `{0, 1}`.
	 * @pre `m >= 1` (asserted).
	 *
	 * @complexity O(log m)
	 */
	template<integral Coefficient, integral Constant, unsigned_integral Modulus>
	[[nodiscard]] constexpr std::optional<congruence<Modulus>> solve_linear_congruence(
		Coefficient a, Constant c, Modulus m) {
		atl_assert(m >= 1);
		Modulus const ra = reduce_mod(a, m), rc = reduce_mod(c, m);
		auto const [g, x, _] = extended_gcd(ra, m);
		if (rc % g != 0) return std::nullopt;
		Modulus const step = m / g;
		// ra·x ≡ g (mod m), so (ra/g)·x ≡ 1 (mod m/g)
		return congruence<Modulus>{mul_mod_unchecked(Modulus(rc / g % step), reduce_mod(x, step), step), step};
	}

	/**
	 * @brief Intersects two residue classes whose moduli need not be coprime.
	 *
	 * `x ≡ r1 (mod m1)` and `x ≡ r2 (mod m2)` hold together iff
	 * `r1 ≡ r2 (mod gcd(m1, m2))`, and then exactly on one class modulo
	 * `lcm(m1, m2)`. Every intermediate stays below that lcm, so nothing wider than
	 * `U` is needed beyond one modular product.
	 *
	 * @tparam U Unsigned integral residue type.
	 * @param first First canonical residue class.
	 * @param second Second canonical residue class.
	 * @return The class modulo `lcm(m1, m2)`, or `std::nullopt` when the two are
	 *         inconsistent.
	 * @pre Both are well-formed (`modulus >= 1`, `residue < modulus`; asserted), and
	 *      `lcm(m1, m2)` fits `U` (asserted). Use `congruence<uint128>` when a
	 *      combined modulus can pass 2^64.
	 *
	 * @complexity O(log min(m1, m2))
	 */
	template<unsigned_integral U>
	[[nodiscard]] constexpr std::optional<congruence<U>> chinese_remainder(
		congruence<U> const &first, congruence<U> const &second) {
		detail::assert_congruence(first);
		detail::assert_congruence(second);
		auto const [g, inverse_first, _] = extended_gcd(first.modulus, second.modulus);
		U const difference = sub_mod_unchecked(second.residue, U(first.residue % second.modulus), second.modulus);
		if (difference % g != 0) return std::nullopt;
		U const step = second.modulus / g;
		atl_assert(first.modulus <= (std::numeric_limits<U>::max)() / step);
		// first.modulus·inverse_first ≡ g (mod second.modulus), so the multiple of
		// first.modulus to add is (difference/g)·inverse_first mod step
		U const k = mul_mod_unchecked(U(difference / g), reduce_mod(inverse_first, step), step);
		return congruence<U>{first.residue + first.modulus * k, first.modulus * step};
	}

	/**
	 * @brief Intersects a list of residue classes whose moduli need not be coprime.
	 *
	 * Folds the pairwise @ref atl::chinese_remainder left to right, stopping at the first
	 * inconsistency.
	 *
	 * @tparam R Input range whose values are `congruence<U>`, with references convertible to values.
	 * @param congruences Classes to intersect, consumed once.
	 * @return The class modulo the lcm of all the moduli (`{0, 1}` for an empty
	 *         list), or `std::nullopt` when the system is inconsistent.
	 * @pre Every congruence is well-formed and every running lcm fits `U` (asserted).
	 *
	 * @complexity O(k log M) word steps and O(1) storage, M the largest input modulus.
	 */
	template<congruence_range R>
	[[nodiscard]] constexpr std::optional<std::ranges::range_value_t<R>> chinese_remainder(R &&congruences) {
		using U = decltype(std::ranges::range_value_t<R>::modulus);
		congruence<U> combined{0, 1};
		for (congruence<U> const &c : congruences) {
			auto const next = chinese_remainder(combined, c);
			if (!next) return std::nullopt;
			combined = *next;
		}
		return combined;
	}

	/**
	 * @brief `x mod target`, where `x` is the unique solution in `[0, Π mᵢ)` of the
	 *        system `x ≡ rᵢ (mod mᵢ)` with pairwise-coprime moduli (Garner's
	 *        algorithm).
	 *
	 * Writes `x = v₀ + v₁·m₀ + v₂·m₀m₁ + …` with mixed-radix digits `vᵢ < mᵢ`, each
	 * found modulo its own `mᵢ`, then evaluates that sum modulo @p target. The
	 * product of the moduli is never formed, so it may be arbitrarily large: this is
	 * the recombination step of multi-prime convolution under an arbitrary modulus.
	 *
	 * @tparam R Input range of `congruence<U>`, with references convertible to values.
	 * @param congruences The residue classes `{rᵢ, mᵢ}`.
	 * @param target      The modulus of the answer.
	 * @return `x mod target` (0 for an empty list).
	 * @pre Every congruence is well-formed, the moduli are pairwise coprime, and
	 *      `target >= 1` (all asserted; coprimality through the inverses it takes).
	 *
	 * @complexity O(k²) modular products plus k inverses and O(k) storage, for k classes.
	 */
	template<congruence_range R>
	[[nodiscard]] constexpr auto chinese_remainder_mod(
		R &&congruences, decltype(std::ranges::range_value_t<R>::modulus) target) {
		using U = decltype(std::ranges::range_value_t<R>::modulus);
		atl_assert(target >= 1);
		std::vector<congruence<U>> system;
		if constexpr (std::ranges::sized_range<R>) system.reserve(std::ranges::size(congruences));
		for (congruence<U> const &c : congruences) {
			detail::assert_congruence(c);
			system.push_back(c);
		}
		std::size_t const k = system.size();
		// slot j < k tracks modulus mⱼ, slot k the target: the partial sum
		// Σ_{i<done} vᵢ·m₀…m_{i-1} and the product m₀…m_{done-1}, both mod that slot
		std::vector<U> partial(k + 1, 0), product(k + 1);
		for (std::size_t j = 0; j < k; ++j) product[j] = U(1 % system[j].modulus);
		product[k] = U(1 % target);
		for (std::size_t i = 0; i < k; ++i) {
			U const m = system[i].modulus;
			U const digit =
				mul_mod_unchecked(sub_mod_unchecked(system[i].residue, partial[i], m), inverse_mod(product[i], m), m);
			for (std::size_t j = i + 1; j <= k; ++j) {
				U const mj = j < k ? system[j].modulus : target;
				partial[j] = add_mod_unchecked(partial[j], mul_mod_unchecked(U(digit % mj), product[j], mj), mj);
				product[j] = mul_mod_unchecked(product[j], U(m % mj), mj);
			}
		}
		return partial[k];
	}
}
