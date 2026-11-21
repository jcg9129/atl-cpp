#include <atl/ds/aggregate_deque.hpp>
#include <atl/ds/aggregate_queue.hpp>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>

// An affine map x -> a * x + b; composing them is associative but not commutative.
struct affine {
	std::int64_t a, b;
};

int main() {
	// A window of the last three maps. The fold applies them oldest first: for maps f, g, h it is x -> h(g(f(x))).
	auto const then = [](affine first, affine second) { return affine{second.a * first.a, second.a * first.b + second.b}; };
	atl::aggregate_queue window(affine{1, 0}, then);
	std::array<affine, 5> const maps{{{2, 1}, {1, 3}, {3, 0}, {1, -4}, {2, 2}}};
	for (affine const &map : maps) {
		window.push_back(map);
		if (window.size() > 3) window.pop_front();
		affine const composed = window.fold();
		std::cout << "window of " << window.size() << ": a = " << composed.a << ", b = " << composed.b << '\n';
	}

	atl::aggregate_deque<std::string, std::plus<>> words("");
	words.push_back("cd");
	words.push_front("ab");
	words.emplace_back(2, 'e');
	std::cout << words.fold() << '\n';
	words.pop_front();
	words.pop_back();
	std::cout << words.fold() << ' ' << words.size() << '\n';
}
