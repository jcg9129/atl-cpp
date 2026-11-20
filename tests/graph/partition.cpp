#include <atl/graph/partition.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;
using atl::uint32;

TEST_CASE("vertex_partition: grouped matches groups on random partitions", "[graph]") {
	std::mt19937 rng(3);
	for (int iter = 0; iter < 500; ++iter) {
		std::size_t const n = rng() % 40, count = n == 0 ? 0 : 1 + rng() % n;
		std::vector<uint32> component(n);
		for (std::size_t v = 0; v < n; ++v) component[v] = v < count ? uint32(v) : uint32(rng() % count);
		std::ranges::shuffle(component, rng);
		g::vertex_partition const p{component, count};
		auto const nested = p.groups();
		auto const flat = p.grouped();
		REQUIRE(flat.count() == count);
		REQUIRE(flat.members.size() == n);
		REQUIRE(flat.offsets.back() == n);
		for (std::size_t c = 0; c < count; ++c)
			REQUIRE(std::vector<g::vertex_id>(flat[c].begin(), flat[c].end()) == nested[c]);
	}
}

TEST_CASE("vertex_groups: empty and out-of-range", "[graph]") {
	g::vertex_groups const none;
	REQUIRE(none.count() == 0);
	REQUIRE(g::vertex_partition{{}, 0}.grouped().count() == 0);
	g::vertex_partition const p{{1, 0, 1}, 2};
	auto const flat = p.grouped();
	REQUIRE(flat[0].size() == 1);
	REQUIRE(flat[1].size() == 2);
	CHECK(atl_test::aborts([&] { (void)flat[2]; }));
}

TEST_CASE("vertex_partition: out-of-range component id aborts", "[graph]") {
	CHECK(atl_test::aborts([] { g::vertex_partition const p{{0, 2, 1}, 2}; (void)p; }));
	CHECK(atl_test::aborts([] { g::vertex_partition const p{{0}, 0}; (void)p; }));
}

TEST_CASE("vertex_partition: unused ids and empty slices", "[graph]") {
	g::vertex_partition const empty{{}, 3};
	auto const flat = empty.grouped();
	REQUIRE(empty.groups().size() == 3);
	for (std::size_t group = 0; group < flat.count(); ++group) REQUIRE(flat[group].empty());
	g::vertex_partition const partition{{0, 2}, 3};
	REQUIRE(partition.groups()[1].empty());
	REQUIRE(partition.grouped()[1].empty());
	REQUIRE(atl_test::aborts([&] { (void)partition.same(0, 2); }));
}
