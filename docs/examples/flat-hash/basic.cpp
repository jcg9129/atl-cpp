#include <atl/ds/flat_hash_map.hpp>
#include <atl/ds/flat_hash_set.hpp>
#include <iostream>
#include <string>
#include <string_view>

int main() {
	atl::flat_hash_map<std::string, int> counts;
	for (std::string const word : {"red", "blue", "red"}) ++counts[word];
	std::cout << counts.at(std::string_view("red")) << '\n';
	atl::flat_hash_set<int> seen;
	std::cout << seen.insert(7) << ' ' << seen.insert(7) << '\n';
}
