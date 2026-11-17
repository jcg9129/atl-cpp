// Temporary MSVC ARM64 probe for ds-2 (removed before merge).
#include <atl/ds/binary_indexed_tree.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdio>
#include <functional>

namespace {
	void show(char const *what, std::array<std::size_t, 2> const &s) {
		std::printf("probe %s: %zu,%zu\n", what, s[0], s[1]);
		std::fflush(stdout);
	}
}

TEST_CASE("msvc probe fenwick sizes", "[msvc_probe]") {
	auto const extents = atl::detail::checked_extents(3, 4);
	show("checked_extents", extents);
	std::array<std::size_t, 2> const direct_sizes{3, 4};
	atl::binary_indexed_tree<int, 2> direct(direct_sizes);
	show("direct int", direct.sizes());
	atl::binary_indexed_tree<long long, 2> direct_ll(direct_sizes);
	show("direct long long", direct_ll.sizes());
	atl::binary_indexed_tree<int, 2, std::multiplies<>> direct_mul(direct_sizes, 1);
	show("direct multiplies", direct_mul.sizes());
	auto factory = atl::make_binary_indexed_tree(0, 3, 4);
	show("factory", factory.sizes());
	auto copy = direct;
	show("copy", copy.sizes());
	std::printf("probe sizeof: tree=%zu array=%zu\n", sizeof(direct), sizeof(direct_sizes));
	std::fflush(stdout);
	REQUIRE(direct.sizes() == direct_sizes);
	REQUIRE(factory.sizes() == direct_sizes);
}
