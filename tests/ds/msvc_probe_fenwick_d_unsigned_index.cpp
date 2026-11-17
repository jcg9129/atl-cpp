// Temporary MSVC ARM64 probe for ds-2 (removed before merge).
#include <atl/ds/binary_indexed_tree.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <cstdio>

TEST_CASE("msvc probe fenwick unsigned index", "[msvc_probe]") {
	atl::binary_indexed_tree<int> tree(5);
	tree.update(2, std::size_t(1));
	std::printf("probe d: ok\n");
	std::fflush(stdout);
	REQUIRE(tree.prefix_fold(std::size_t(2)) == 2);
}
