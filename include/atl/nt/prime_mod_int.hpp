#pragma once

#include <atl/nt/basic.hpp>
#include <atl/types/concepts.hpp>
#include <cstdint>
#include <limits>
#include <ranges>

namespace atl {
	/**
	 * @brief A static modular integer whose modulus is a prime that fits in 64 bits.
	 *
	 * Satisfied by an object type that models @ref atl::static_mod_int (so `Coefficient::mod()` is a constant
	 * expression) whose modulus passes `atl::is_prime`. Primality is a compile-time check on that static modulus:
	 * it runs while the constraint is checked, never at run time. Dynamic moduli and composite static moduli do
	 * not model it. The type is taken as written: references and cv-qualified types do not model it (they are not
	 * `std::regular`), so constrain on the decayed type when one may arrive.
	 */
	template<typename Coefficient>
	concept static_prime_mod_int = static_mod_int<Coefficient>
								&& (Coefficient::mod() <= (std::numeric_limits<std::uint64_t>::max)())
								&& is_prime(std::uint64_t(Coefficient::mod()));

	/**
	 * @brief An input range of static prime-modulus integers that can be iterated through a const reference.
	 *
	 * `Range` may be a reference or cv-qualified type: the check is made on `Range const` (a reference stays a
	 * reference to the same range), and the element type is `std::ranges::range_value_t<Range const>`, which is
	 * cv-unqualified, so ranges of `const` coefficients qualify too. Single-pass ranges qualify; the range need not
	 * be sized or forward.
	 */
	template<typename Range>
	concept prime_mod_int_range = std::ranges::input_range<Range const>
							   && static_prime_mod_int<std::ranges::range_value_t<Range const>>;

	/**
	 * @brief The coefficient type of a series range: `std::ranges::range_value_t<Range const>`.
	 *
	 * The functions that take a @ref atl::prime_mod_int_range return coefficients of this type, and two input
	 * ranges are compatible when their coefficient types are the same.
	 * @tparam Range An input range that can also be iterated through a const reference; references and
	 *         cv-qualified ranges are accepted.
	 */
	template<std::ranges::input_range Range>
	using series_coefficient_t = std::ranges::range_value_t<Range const>;
}
