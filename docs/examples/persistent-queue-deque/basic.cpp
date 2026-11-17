#include <atl/ds/persistent_adapters.hpp>
#include <iostream>

int main() {
	atl::persistent_queue<int> empty;
	auto queue = empty.push(1).push(2);
	auto next = queue.pop();
	std::cout << queue.front() << ' ' << next.front() << '\n';
	atl::persistent_deque<int> base(3);
	auto deque = base.push_back(2).push_front(1).push_back(3);
	auto changed = deque.pop_front().push_back(4);
	std::cout << deque.front() << ' ' << changed.front() << ' ' << changed.back() << '\n';
}
