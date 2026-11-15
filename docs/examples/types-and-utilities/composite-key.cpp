#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/hash.hpp>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

template<atl::integral T>
constexpr T twice(T value) {
	return value + value;
}

int main() {
	using key = std::pair<atl::uint32, std::string>;
	std::unordered_map<key, int, atl::hash<key>> scores;
	scores.emplace(key{7, "blue"}, twice(21));
	std::cout << scores.at(key{7, "blue"}) << '\n';
}
