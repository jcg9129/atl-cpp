#include <atl/ds/top_clusters/diameter.hpp>
#include <atl/ds/top_tree_engine.hpp>
#include <iostream>

int main() {
	atl::diameter_top_tree<long long> d(4);
	atl::uint32 e = d.link(0, 1, 5);
	d.link(1, 2, 3);
	d.link(1, 3, 4);
	long long initial_diameter = d.diameter(0);
	d.set(e, 1);
	std::cout << initial_diameter << ' ' << d.diameter(0) << '\n';
}
