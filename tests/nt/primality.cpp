#include <atl/nt/detail/primality_static.hpp>
#include <atl/nt/primality.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <utility>

namespace {
	template<atl::uint64 Mod, bool Expected>
	void check_prime() {
		CAPTURE(Mod);
		constexpr bool static_result = atl::detail::static_is_prime<Mod>();
		CHECK(static_result == Expected);
		CHECK(atl::is_prime(Mod) == static_result);
		if constexpr (Mod <= 0xffffffffull) {
			CHECK(atl::detail::static_is_prime<atl::uint32(Mod)>() == static_result);
		}
	}
}

TEST_CASE("static and runtime primality agree on deterministic witness and word boundaries", "[primality]") {
	[]<atl::uint64... Values>(std::integer_sequence<atl::uint64, Values...>) {
		(check_prime<Values, true>(), ...);
	}(std::integer_sequence<atl::uint64,
							2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 61, 73, 97, 193, 257,
							998244353, 1000000007, 4294967291ull, 4294967311ull, 2305843009213693951ull,
							4179340454199820289ull, 18446744069414584321ull, 18446744073709551557ull>{});
	[]<atl::uint64... Values>(std::integer_sequence<atl::uint64, Values...>) {
		(check_prime<Values, false>(), ...);
	}(std::integer_sequence<atl::uint64,
							0, 1, 4, 9, 25, 49, 121, 169, 289, 361, 529, 841, 961, 1369,
							561, 1105, 1729, 2465, 2821, 6601, 41041, 825265, 321197185,
							2047, 1373653, 25326001, 3215031751ull, 4759123140ull, 4759123141ull, 4759123142ull,
							2152302898747ull, 3474749660383ull, 341550071728321ull, 3825123056546413051ull,
							4294967295ull, 4294967296ull, 4294967297ull, 2305843009213693953ull,
							18446744069414584320ull, 18446744069414584322ull, 18446744073709551556ull,
							18446744073709551558ull, 18446744073709551614ull, 18446744073709551615ull>{});
}
