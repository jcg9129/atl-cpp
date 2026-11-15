#include <atl/types/forward_like.hpp>
#include <catch2/catch.hpp>
#include <type_traits>
#include <utility>

// atl::forward_like must match the C++23 std::forward_like semantics on every
// (value-category, constness) combination of the model type T. These are pure
// type identities, so static_assert is the whole test — and it runs under both
// branches of the shim (define ATL_NO_STD_FORWARD_LIKE to force the fallback).
#ifdef ATL_FORCE_FALLBACKS
static_assert(ATL_HAS_INT128 == 0);
static_assert(atl::detail::uses_forward_like_fallback);
#endif

namespace {
	[[maybe_unused]] int obj = 0;

	// non-const lvalue model: value category from T, no const added
	static_assert(std::is_same_v<decltype(atl::forward_like<int &>(obj)), int &>);
	static_assert(std::is_same_v<decltype(atl::forward_like<int &>(std::move(obj))), int &>);
	// const lvalue model: adds const, stays lvalue
	static_assert(std::is_same_v<decltype(atl::forward_like<int const &>(obj)), int const &>);
	// non-const rvalue model: value category rvalue
	static_assert(std::is_same_v<decltype(atl::forward_like<int>(obj)), int &&>);
	static_assert(std::is_same_v<decltype(atl::forward_like<int &&>(obj)), int &&>);
	// const rvalue model: const rvalue
	static_assert(std::is_same_v<decltype(atl::forward_like<int const>(obj)), int const &&>);
	static_assert(std::is_same_v<decltype(atl::forward_like<int const &&>(obj)), int const &&>);
	// constness of the argument itself never removes constness the model lacks
	int const cobj = 0;
	static_assert(std::is_same_v<decltype(atl::forward_like<int &&>(cobj)), int const &&>);
}

TEST_CASE("forward_like: forwards the model's value category and constness", "[types]") {
	// a runtime touchpoint so the TU is a real (not header-only) test case
	int x = 41;
	auto &&r = atl::forward_like<int const &>(x);
	static_assert(std::is_same_v<decltype(r), int const &>);
	REQUIRE(&r == &x);
	REQUIRE(r == 41);
}
