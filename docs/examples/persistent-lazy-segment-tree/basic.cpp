#include <atl/ds/persistent_lazy_segment_tree.hpp>
#include <iostream>
#include <vector>

struct aggregate {
	long long sum, width;
};

int main() {
	auto op = [](aggregate a, aggregate b) { return aggregate{a.sum + b.sum, a.width + b.width}; };
	auto map = [](long long f, aggregate x) { return aggregate{x.sum + f * x.width, x.width}; };
	auto compose = [](long long f, long long g) { return f + g; };
	atl::persistent_lazy_segment_tree a(atl::commutative_tags, std::vector<aggregate>{{1, 1}, {2, 1}, {3, 1}},
										aggregate{0, 0}, 0LL, op, map, compose);
	auto b = a.apply(0, 2, 5);
	auto c = b.copy_range(a, 1, 3);
	std::cout << a.fold().sum << ' ' << b.fold().sum << ' ' << c.fold().sum << '\n';
}
