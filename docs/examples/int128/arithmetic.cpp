#include <atl/types/alias.hpp>
#include <atl/types/int128_format.hpp>
#include <format>
#include <iostream>

int main() {
	atl::uint128 const value = (atl::uint128{1} << 100) + 17;
	std::cout << std::format("{}\n", atl::detail::software_uint128(value));
	std::cout << std::format("{} {}\n", atl::detail::software_int128(atl::int128{-42} / 5),
							 atl::detail::software_int128(atl::int128{-42} % 5));
	std::cout << std::format("{:+#012x}\n", atl::detail::software_int128{-42});
}
