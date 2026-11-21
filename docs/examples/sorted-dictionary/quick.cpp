#include <atl/strings/sorted_dictionary.hpp>
#include <array>
#include <iostream>
#include <string_view>

int main() {
	std::array const words{std::string_view("ab"), std::string_view("abc"), std::string_view("b")};
	atl::sorted_dictionary const dictionary(words);
	for (std::string_view query : {"ab", "c", ""}) {
		auto const interval = dictionary.prefix_interval(query);
		std::cout << interval.begin << ' ' << interval.end << '\n';
	}
}
