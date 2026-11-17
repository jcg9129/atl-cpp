#include <atl/ds/wavelet_matrix.hpp>
#include <array>
#include <iostream>

int main() {
	std::array const values{5u, 1u, 4u, 1u, 7u, 0u};
	atl::wavelet_matrix matrix(values);
	std::cout << matrix.kth(1, 5, 2) << ' ' << matrix.count_less(1, 5, 4) << ' ' << matrix.count(0, 6, 1, 5) << '\n';
}
