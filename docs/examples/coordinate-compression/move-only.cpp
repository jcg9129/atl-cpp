#include <atl/ds/coordinate_compression.hpp>
#include <array>
#include <functional>
#include <iostream>
#include <utility>

struct ordering {
	ordering() = default;
	ordering(ordering &&) = default;
	ordering(ordering const &) = delete;

	bool operator()(int a, int b) const { return a < b; }
};

int main() {
	auto result = atl::compress(std::array{30, 10, 30}, std::identity{}, ordering{});
	std::cout << result.universe.rank(30) << ' ' << result.universe.rank_upper(30) << '\n';
	std::cout << result.ranks[0] << ' ' << result.ranks[1] << ' ' << result.ranks[2] << '\n';
}
