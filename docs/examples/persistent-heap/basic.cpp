#include <atl/ds/persistent_heap.hpp>
#include <iostream>
#include <vector>

int main() {
	atl::persistent_heap<int> base;
	auto a = base.push_range(std::vector{3, 7, 1});
	auto b = a.pop().push(5);
	std::cout << a.top() << ' ' << b.top() << ' ' << a.size() << '\n';
	atl::persistent_heap_forest<int> forest;
	auto const x = forest.push(forest.empty_heap, 4);
	auto const y = forest.push(x, 8);
	std::cout << forest.top(x) << ' ' << forest.top(y) << '\n';
}
