#include <atl/ds/dynamic_lazy_segment_tree.hpp>
#include <functional>
#include <iostream>

int main() {
	struct item {
		long long sum, width;
	};

	atl::dynamic_lazy_segment_tree library(1'000'000'000, item{0, 0}, 0LL, [](item a, item b) { return item{a.sum + b.sum, a.width + b.width}; }, [](long long tag, item value) { return item{value.sum + tag * value.width, value.width}; }, std::plus<>{}, [](std::size_t lo, std::size_t hi) { return item{0, static_cast<long long>(hi - lo)}; });
	library.apply(2, 5, 3);
	std::cout << library.fold().sum << ' ' << library.at(3).sum << '\n';
}
