#include <atl/types/alias.hpp>
#include <atl/types/int128_format.hpp>
#include <format>
#include <iostream>

int main() {
	atl::uint128 const value = (atl::uint128{1} << 100) + 17;
	std::cout << std::format("{}\n", value);
	std::cout << std::format("{} {}\n", atl::int128{-42} / 5, atl::int128{-42} % 5);
	std::cout << std::format("{:+#012x}\n", atl::int128{-42});
}
