#include <atl/ds/dynamic_li_chao_tree.hpp>
#include <atl/ds/li_chao_tree.hpp>
#include <atl/ds/linear_function.hpp>
#include <atl/types/alias.hpp>
#include <iostream>
#include <vector>

int main() {
	atl::li_chao_tree<long long> fixed(std::vector<long long>{1, 7, 9});
	fixed.insert(1, 0);
	fixed.insert(-1, 10, 3, 8);
	std::cout << *fixed.query(7) << ' ' << *fixed.query(9) << '\n';
	atl::dynamic_li_chao_tree<atl::int128> wide(-1000, 1000);
	wide.insert(atl::linear_function<atl::int128>{atl::int128(1) << 70, 3});
	std::cout << static_cast<int>(*wide.query(0)) << '\n';
}
