#include <atl/strings/lexicographic_order.hpp>
#include <array>
#include <iostream>
#include <string_view>

int main() {
	std::array const words{std::string_view("ab"), std::string_view("a"), std::string_view("ab")};
	char const *separator = "";
	for (auto index : atl::lexicographic_order(words)) {
		std::cout << separator << index;
		separator = " ";
	}
	std::cout << '\n';
}
