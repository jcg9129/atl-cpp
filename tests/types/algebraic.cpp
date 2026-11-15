#include <atl/types/algebraic.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <functional>

// The callable concepts check well-formedness on the documented value
// category: const for the operations the structures call from const members,
// non-const for update_function.
namespace {
	using add = decltype([](int a, int b) { return a + b; });
	using counting_add = decltype([n = 0](int a, int b) mutable { return n += a + b; });
	using to_pointer = decltype([](int, int) { return "x"; });

	static_assert(atl::binary_operation<std::plus<>, int>);
	static_assert(atl::binary_operation<add, int>);
	static_assert(atl::binary_operation<add, int, long>);
	static_assert(!atl::binary_operation<counting_add, int>); // mutable
	static_assert(!atl::binary_operation<to_pointer, int>);

	static_assert(atl::inverse_operation<std::negate<>, int>);
	static_assert(!atl::inverse_operation<std::plus<>, int>); // binary, not unary

	using add_tag = decltype([](long f, int s) { return int(f + s); });
	static_assert(atl::action<add_tag, long, int>);
	static_assert(!atl::action<add_tag, int *, int>);

	using absorb = decltype([](int const &f, int &s) { s += f; return true; });
	using absorb_int = decltype([](int const &, int &s) { return s; });
	static_assert(atl::fallible_action<absorb, int, int>);
	static_assert(!atl::fallible_action<absorb_int, int, int>); // exactly bool

	using twice = decltype([](int x) { return 2 * x; });
	using stateful = decltype([k = 0](int x) mutable { return x + ++k; });
	static_assert(atl::update_function<twice, int>);
	static_assert(atl::update_function<stateful, int>); // invoked as an lvalue
	static_assert(!atl::update_function<stateful const, int>);
	static_assert(!atl::update_function<twice, int *>);

	using by_width = decltype([](std::size_t w) { return int(w); });
	using by_bounds = decltype([](std::size_t lo, std::size_t hi) { return int(hi - lo); });
	using mutable_fill = decltype([k = 0](std::size_t w) mutable { return int(w) + k++; });
	static_assert(atl::fill_function<by_width, int>);
	static_assert(atl::fill_function<by_bounds, int>);
	static_assert(!atl::fill_function<mutable_fill, int>); // called from const members

	static_assert(atl::tag_order<void> && atl::tag_order<atl::commutative_tags_t>);
	static_assert(!atl::tag_order<atl::commutative_fold_t>);
	static_assert(atl::fold_order<void> && atl::fold_order<atl::commutative_fold_t>);
	static_assert(!atl::fold_order<int>);
}

TEST_CASE("algebraic concepts: the documented callable contracts", "[types]") {
	REQUIRE(std::invoke(add{}, 2, 3) == 5);
	int s = 2;
	REQUIRE(absorb{}(3, s));
	REQUIRE(s == 5);
}
