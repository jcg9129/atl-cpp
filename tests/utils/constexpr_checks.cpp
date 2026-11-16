#include <atl/types/alias.hpp>
#include <atl/utils/counting_sort.hpp>
#include <atl/utils/floor_midpoint.hpp>
#include <atl/utils/prefetch.hpp>
#include <atl/utils/radix_sort.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

using atl::uint32;

static_assert([] {
	std::array<uint32, 5> const keys{3, 1, 3, 0, 1};
	std::array<uint32, 5> offsets{}, order{};
	atl::detail::bucket_by_key(4, [&](std::size_t i) { return keys[i]; }, std::span<uint32>(offsets), std::span<uint32>(order));
	return offsets == std::array<uint32, 5>{0, 1, 3, 3, 5} && order == std::array<uint32, 5>{3, 1, 4, 0, 2};
}());

static_assert(atl::detail::floor_midpoint(-7LL, 4LL) == -2);
static_assert(atl::detail::floor_midpoint(std::numeric_limits<long long>::min(), std::numeric_limits<long long>::max()) == -1);

static constexpr bool shared_radix_sort() {
	using keyed = atl::detail::keyed_position<atl::uint32>;
	std::vector<keyed> keys = {{0x30000u, 0}, {5u, 1}, {0x30000u, 2}, {0u, 3}, {0xffffffffu, 4}};
	atl::detail::radix_sort<11>(keys);
	std::vector<atl::uint32> order;
	for (auto const &k : keys) order.push_back(k.position);
	return order == std::vector<atl::uint32>{3, 1, 0, 2, 4};
}

static_assert(shared_radix_sort());

static_assert([] {
	int value = 0;
	atl::prefetch<true>(&value);
	return value == 0;
}());

TEST_CASE("utility algorithms are constant-evaluable", "[utils][constexpr]") {
	STATIC_REQUIRE(atl::detail::floor_midpoint(-7LL, 4LL) == -2);
}
