#include <atl/dp/segmentation.hpp>
#include <atl/types/alias.hpp>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
	// break words into lines: a line holding words [j, i) costs (its length - 10)^2
	std::vector<atl::int64> const words{3, 4, 2, 5, 3, 6, 2, 4};
	std::vector<atl::int64> prefix{0};
	for (atl::int64 const length : words) prefix.push_back(prefix.back() + length);
	auto const lines = atl::optimal_segmentation(words.size(), [&prefix](atl::uint32 j, atl::uint32 i) {
		atl::int64 const slack = prefix[i] - prefix[j] - 10;
		return slack * slack;
	});
	std::cout << "badness " << lines.value[words.size()] << '\n';
	auto const cuts = lines.boundaries();
	for (std::size_t t = 0; t + 1 < cuts.size(); ++t) std::cout << "line [" << cuts[t] << ", " << cuts[t + 1] << ")\n";
}
