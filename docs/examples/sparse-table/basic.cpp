#include <atl/ds/sparse_table.hpp>
#include <array>
#include <functional>
#include <iostream>

int main() {
	std::array const values{4, 2, 7, 1, 9};
	atl::sparse_table minimum(values, [](int x, int y) { return x < y ? x : y; });
	atl::disjoint_sparse_table sum(values, std::plus<>{});
	std::cout << minimum.fold(1, 4) << ' ' << sum.fold(1, 4) << '\n';
}
