#include <atl/utils/assert.hpp>
#include <atl/utils/prefetch.hpp>
#include <catch2/catch.hpp>
#include "../expect_abort.hpp"

namespace {
	constexpr bool check_assert() {
		int value = 0;
		if constexpr (true)
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

TEST_CASE("aborts: returning bodies fail the death check", "[utils]") {
	REQUIRE_FALSE(atl_test::aborts([] {}));
}

TEST_CASE("aborts: successive calls, [brackets], \"quotes\", \\ and * in a name", "[utils]") {
	REQUIRE(atl_test::aborts([] { std::abort(); }));
	REQUIRE(atl_test::aborts([] {
		int value = 0;
		atl_assert(value != 0);
	}));
	REQUIRE_FALSE(atl_test::aborts([] {}));
}

TEST_CASE("aborts: indices restart for another test case", "[utils]") {
	SECTION("first traversal") {
		REQUIRE(atl_test::aborts([] { std::abort(); }));
	}
	SECTION("second traversal") {
		REQUIRE(atl_test::aborts([] { std::abort(); }));
	}
}
