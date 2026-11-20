#include <atl/graph/assignment.hpp>
#include <array>
#include <iostream>

int main() {
	std::array<std::array<int, 3>, 2> const costs{{{5, 1, 9}, {2, 8, 3}}};
	auto const r = atl::graph::assignment<long long>(costs);
	std::cout << r.cost << '\n';
	std::cout << r.row_to_col[0] << ' ' << r.row_to_col[1] << '\n';
	std::cout << (r.col_to_row[2] == atl::graph::none) << '\n';
}
