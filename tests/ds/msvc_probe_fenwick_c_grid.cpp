// Temporary MSVC ARM64 probe for ds-2 (removed before merge).
#include <atl/ds/binary_indexed_tree.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <cstdio>

TEST_CASE("msvc probe fenwick grid", "[msvc_probe]") {
	auto grid = atl::make_binary_indexed_tree(0, 3, 4);
	std::printf("probe c: sizes=%zu,%zu\n", grid.sizes()[0], grid.sizes()[1]);
	std::fflush(stdout);
	grid.update(7, 1, 2);
	REQUIRE(grid.prefix_fold(2, 3) == 7);
}
