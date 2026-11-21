#include <atl/strings/matching.hpp>
#include <iostream>
#include <string_view>

int main() {
	std::string_view const text = "abababa", pattern = "aba";
	auto const borders = atl::border_lengths(pattern);
	for (auto position : atl::occurrences(text, pattern, borders)) std::cout << position << '\n';
	std::cout << atl::common_prefix_lengths(text)[2] << '\n';
}
