#include <atl/ds/static_vector.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <compare>
#include <concepts>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"
#include "move_probe.hpp"

TEST_CASE("static_vector basics", "[static_vector]") {
	atl::static_vector<int, 4> v;
	REQUIRE(v.empty());
	v.push_back(1);
	v.push_back(2);
	REQUIRE(v.size() == 2);
	REQUIRE(v[0] == 1);
	REQUIRE(v[1] == 2);
	int sum = 0;
	for (int x : v) sum += x;
	REQUIRE(sum == 3);
}

TEST_CASE("static_vector: emplace_back and push_back(&&) avoid copies", "[static_vector]") {
	using atl_test::move_probe;
	atl::static_vector<move_probe, 8> v;

	move_probe::reset();
	v.emplace_back(move_probe::sentinel);
	REQUIRE(v[0].v == move_probe::sentinel);
	REQUIRE(move_probe::sentinel_copies == 0);

	move_probe::reset();
	v.push_back(move_probe(move_probe::sentinel));
	REQUIRE(v[1].v == move_probe::sentinel);
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(move_probe::sentinel_moves >= 1);

	move_probe::reset();
	move_probe lvalue(move_probe::sentinel);
	v.push_back(lvalue);
	REQUIRE(v[2].v == move_probe::sentinel);
	REQUIRE(move_probe::sentinel_copies >= 1);
}

TEST_CASE("static_vector: std-container parity", "[static_vector]") {
	using vec = atl::static_vector<std::string, 4>;
	static_assert(std::same_as<vec::value_type, std::string> && std::same_as<vec::iterator, std::string *>
				  && std::same_as<vec::const_reference, std::string const &> && vec::capacity() == 4);
	static_assert(noexcept(std::declval<vec const &>().size()) && noexcept(std::declval<vec const &>().empty()));

	vec v{"a", "b", "c"};
	REQUIRE(v.front() == "a");
	REQUIRE(v.back() == "c");
	REQUIRE(v.data() == &v[0]);
	v.pop_back();
	REQUIRE(v.size() == 2);
	// the popped slot is released, so a later grow reads value-initialized
	v.resize(4);
	REQUIRE(v.size() == 4);
	REQUIRE(v[2].empty());
	REQUIRE(v[3].empty());
	v.resize(1);
	REQUIRE(v == vec{"a"});
	v.resize(3, "z");
	REQUIRE(v == vec{"a", "z", "z"});
	REQUIRE(v != vec{"a", "z"});
	REQUIRE(vec{"a", "z"} < v);
	REQUIRE((vec{"b"} <=> v) == std::strong_ordering::greater);
	v.clear();
	REQUIRE(v.empty());
	REQUIRE(v == vec{});
}

TEST_CASE("static_vector: equality ignores the spare capacity", "[static_vector]") {
	atl::static_vector<int, 4> a{1, 2, 3}, b{1, 2};
	a.pop_back();
	REQUIRE(a == b);
	b.push_back(7);
	b.pop_back();
	REQUIRE(a == b);
}

TEST_CASE("static_vector: range constructor follows the forwarding rule", "[static_vector]") {
	using atl_test::move_probe;
	using vec = atl::static_vector<move_probe, 4>;
	std::vector<move_probe> src(3, move_probe(move_probe::sentinel));

	move_probe::reset();
	auto const from_span = vec(std::span(src)); // borrowed rvalue: copied, the source stays intact
	REQUIRE(move_probe::sentinel_copies == 3);
	REQUIRE(std::ranges::all_of(src, [](move_probe const &p) { return p.v == move_probe::sentinel; }));
	REQUIRE(from_span.size() == 3);

	move_probe::reset();
	auto const from_filter = vec(src | std::views::filter([](move_probe const &) { return true; })); // view: copied
	REQUIRE(move_probe::sentinel_copies == 3);
	REQUIRE(from_filter.size() == 3);
	REQUIRE(std::ranges::all_of(src, [](move_probe const &p) { return p.v == move_probe::sentinel; }));

	move_probe::reset();
	auto const moved = vec(std::move(src)); // owning rvalue: moved
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(moved.size() == 3);
	REQUIRE(moved[2].v == move_probe::sentinel);
}

TEST_CASE("static_vector: precondition violations abort", "[static_vector]") {
	REQUIRE(atl_test::aborts([] {
		atl::static_vector<int, 2> v;
		v.pop_back();
	}));
	REQUIRE(atl_test::aborts([] {
		atl::static_vector<int, 2> v;
		v.resize(3);
	}));
	REQUIRE(atl_test::aborts([] {
		atl::static_vector<int, 2> v;
		(void)v.back();
	}));
	REQUIRE(atl_test::aborts([] { atl::static_vector<int, 2> v(std::vector<int>{1, 2, 3}); }));
}

TEST_CASE("static_vector: value-yielding views move their temporary elements", "[static_vector]") {
	using atl_test::move_probe;
	auto values = std::views::iota(0, 3) | std::views::transform([](int) { return move_probe(move_probe::sentinel); });
	move_probe::reset();
	atl::static_vector<move_probe, 3> const vector(values);
	REQUIRE(vector.size() == 3);
	REQUIRE(move_probe::sentinel_copies == 0);
	static_assert([] { atl::static_vector<int, 0> empty; return empty.begin() == empty.end(); }());
}
