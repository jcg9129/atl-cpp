#include <atl/dp/convex_piecewise_linear.hpp>
#include <atl/types/alias.hpp>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
	// the nondecreasing sequence b closest to a in total |b_i - a_i|
	std::vector<atl::int64> const a{3, 1, 4, 1, 5, 9, 2, 6};
	atl::convex_piecewise_linear<atl::int64> f; // unit breakpoints, the default
	std::vector<atl::slope_clip_interval<atl::int64>> ranges;
	for (atl::int64 const x : a) {
		ranges.push_back(f.take_prefix_minimum()); // b_i >= b_{i-1}: the best of all smaller b
		f.add_absolute(x);
	}
	std::cout << "least total change: " << f.minimum() << '\n';
	// top-down: b_{i-1} is the clamp of b_i into the range the prefix minimum reported
	std::vector<atl::int64> b(a.size());
	b.back() = *f.minimizers().lower;
	for (std::size_t i = a.size() - 1; i > 0; --i) b[i - 1] = ranges[i].clamp(b[i]);
	std::cout << "b:";
	for (atl::int64 const x : b) std::cout << ' ' << x;
	std::cout << '\n';

	// the same with weights: Slope = int64 stores (position, weight) breakpoints
	std::vector<atl::int64> const w{1, 5, 1, 1, 2, 1, 3, 1};
	atl::convex_piecewise_linear<atl::int64, atl::int64> g;
	for (std::size_t i = 0; i < a.size(); ++i) {
		g.take_prefix_minimum();
		g.add_absolute(a[i], w[i]);
	}
	std::cout << "least weighted change: " << g.minimum() << '\n';
}
