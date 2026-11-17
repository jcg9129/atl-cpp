#include <atl/ds/meldable_heap_forest.hpp>
#include <functional>
#include <iostream>

int main() {
	atl::meldable_heap_forest<int, std::less<>, int> heaps(2);
	heaps.push(0, 3);
	heaps.push(1, 7);
	heaps.add_to_all(0, 5);
	heaps.meld(1, 0);
	std::cout << heaps.pop(1) << ' ' << heaps.pop(1) << '\n';
	atl::addressable_meldable_heap_forest<int> handles(1);
	auto const id = handles.push(0, 2);
	handles.promote(0, id, 9);
	std::cout << handles.erase(0, id) << '\n';
}
