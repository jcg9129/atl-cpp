// Temporary MSVC constant-evaluation probe for ds-2 (removed before merge).
#include <catch2/catch.hpp>

namespace msvc_probe {
	// A caller's local passed by reference into the next recursive call.
	constexpr int sum_by_reference(int depth, int const &carried) {
		if (depth == 0) return carried;
		int const next = carried + 1;
		return sum_by_reference(depth - 1, next) + sum_by_reference(depth - 1, next);
	}

	// The same, by value.
	constexpr int sum_by_value(int depth, int carried) {
		if (depth == 0) return carried;
		int const next = carried + 1;
		return sum_by_value(depth - 1, next) + sum_by_value(depth - 1, next);
	}
}

static_assert(msvc_probe::sum_by_value(3, 0) == 24);
#if !defined(MSVC_PROBE_SKIP_REFERENCE)
static_assert(msvc_probe::sum_by_reference(3, 0) == 24);
#endif

TEST_CASE("msvc probe recursion", "[msvc_probe]") { SUCCEED(); }
