#include <atl/strings/suffix_array.hpp>
#include <iostream>
#include <string_view>

int main() {
	atl::suffix_array const suffixes(std::string_view("banana"));
	char const *separator = "";
	for (auto position : suffixes.positions()) {
		std::cout << separator << position;
		separator = " ";
	}
	std::cout << '\n';
	separator = "";
	for (auto length : suffixes.lcp_lengths()) {
		std::cout << separator << length;
		separator = " ";
	}
	std::cout << '\n';
}
