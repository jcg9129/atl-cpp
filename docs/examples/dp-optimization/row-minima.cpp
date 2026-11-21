#include <atl/dp/row_minima.hpp>
#include <atl/types/alias.hpp>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
	// the nearest y to each x, both sorted: A[i][j] = |x_i - y_j| is Monge
	std::vector<atl::int64> const x{1, 4, 9, 10, 20}, y{0, 5, 8, 21};
	auto const distance = [&](atl::uint32 i, atl::uint32 j) { return x[i] > y[j] ? x[i] - y[j] : y[j] - x[i]; };
	auto const smawk = atl::totally_monotone_row_minima(x.size(), y.size(), distance);
	auto const divided = atl::monotone_row_minima(x.size(), y.size(), distance);
	for (std::size_t i = 0; i < x.size(); ++i) std::cout << x[i] << " -> " << y[smawk.column[i]] << " (" << smawk.value[i] << ")\n";
	std::cout << (divided.column == smawk.column ? "both agree" : "they differ") << '\n';
}
