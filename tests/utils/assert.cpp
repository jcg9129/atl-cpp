#include <atl/utils/assert.hpp>
#include <atl/utils/prefetch.hpp>
#include <catch2/catch.hpp>

namespace {
	constexpr bool check_assert() {
		int value = 0;
		if (true)
			atl_assert(value == 0);
		else
			++value;
#ifndef ATL_ASSERT
		atl_assert(++value == 1);
#endif
		atl::prefetch(&value);
		return value == 0;
	}

	static_assert(check_assert());
#ifdef ATL_FORCE_FALLBACKS
	static_assert(atl::detail::uses_prefetch_fallback);
#endif
}

TEST_CASE("assert: constexpr statements and unevaluated disabled conditions", "[utils]") {
	REQUIRE(check_assert());
}
