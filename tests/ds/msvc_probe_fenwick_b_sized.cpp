// Temporary MSVC ARM64 probe for ds-2 (removed before merge).
#include <atl/ds/binary_indexed_tree.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <cstdio>

TEST_CASE("msvc probe fenwick sized", "[msvc_probe]") {
	atl::binary_indexed_tree<int> tree(5);
	std::printf("probe b: size=%zu\n", tree.size());
	std::fflush(stdout);
	tree.update(2, 1);
	REQUIRE(tree.prefix_fold(2) == 2);
}
