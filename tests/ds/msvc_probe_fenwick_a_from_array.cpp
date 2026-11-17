// Temporary MSVC ARM64 probe for ds-2 (removed before merge).
#include <atl/ds/binary_indexed_tree.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <cstdio>

TEST_CASE("msvc probe fenwick from array", "[msvc_probe]") {
	auto tree = atl::make_binary_indexed_tree_from(std::array{3, 1, 4, 1, 5});
	std::printf("probe a: size=%zu\n", tree.size());
	std::fflush(stdout);
	std::printf("probe a: prefix_fold(3)=%d\n", tree.prefix_fold(3));
	std::fflush(stdout);
	tree.update(2, 1);
	std::printf("probe a: after update prefix_fold(3)=%d\n", tree.prefix_fold(3));
	std::fflush(stdout);
	REQUIRE(tree.prefix_fold(3) == 10);
}
