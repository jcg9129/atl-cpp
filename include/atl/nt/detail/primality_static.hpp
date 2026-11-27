#pragma once

#include <atl/nt/detail/primality_base.hpp>
#include <atl/nt/montgomery_static.hpp>
#include <atl/types/alias.hpp>
#include <limits>
#include <type_traits>

namespace atl::detail {
	/** @brief Deterministic static-modulus primality without a runtime reducer. */
	template<auto Mod>
	constexpr bool static_is_prime() {
		constexpr uint64 n = uint64(Mod);
		if constexpr (n < 2) {
			return false;
		} else {
			using U = std::conditional_t<(n <= (std::numeric_limits<uint32>::max)()), uint32, uint64>;
			return is_prime_impl(n, [](uint64) { return atl::modulus_engine<U(n)>{}; });
		}
	}
}
