#include <atl/dp/convex_piecewise_linear.hpp>
#include <atl/types/alias.hpp>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
	// a server starts anywhere; request i at a_i costs w_i |x - a_i| from position x,
	// and moving between requests costs 2 per unit: dp_i = clip(dp_{i-1}) + w_i |x - a_i|
	std::vector<atl::int64> const a{0, 10, 3, 8, 20};
	std::vector<atl::int64> const w{3, 1, 4, 1, 1};
	atl::convex_piecewise_linear<atl::int64, atl::int64> f;
	for (std::size_t i = 0; i < a.size(); ++i) {
		f.clip_slopes(-2, 2); // the infimal convolution with 2|t|
		f.add_absolute(a[i], w[i]);
	}
	auto const best = f.minimizers();
	std::cout << "least cost: " << f.minimum() << ", final positions [" << *best.lower << ", " << *best.upper << "]\n";

	// consecutive positions at most 3 apart: dp_i(x) = min over |x - y| <= 3 of dp_{i-1}(y) + |x - a_i|
	atl::convex_piecewise_linear<atl::int64> g;
	for (atl::int64 const x : a) {
		g.take_window_minimum(-3, 3);
		g.add_absolute(x);
	}
	std::cout << "least deviation with steps of at most 3: " << g.minimum() << '\n';
}
