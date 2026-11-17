#include <atl/ds/splay_engine.hpp>
#include <functional>
#include <iostream>
#include <vector>

struct sum_rope : atl::sized_splay_links {
	std::vector<long long> values{0}, sums{0};
	std::vector<bool> is_marked{false};

	constexpr atl::uint32 make(long long value) {
		values.push_back(value);
		sums.push_back(value);
		is_marked.push_back(false);
		return add_node();
	}

	constexpr long long const &value(atl::uint32 x) const { return values[x]; }

	constexpr long long const &aggregate(atl::uint32 x) const { return sums[x]; }

	constexpr void pull(atl::uint32 x) {
		pull_size(x);
		sums[x] = sums[child(x, 0)] + values[x] + sums[child(x, 1)];
	}

	constexpr void push(atl::uint32 x) {
		if (!is_marked[x]) return;
		auto const left = child(x, 0), right = child(x, 1);
		set_child(x, 0, right);
		set_child(x, 1, left);
		if (left != 0) is_marked[left] = !is_marked[left];
		if (right != 0) is_marked[right] = !is_marked[right];
		is_marked[x] = false;
	}
};

int main() {
	using engine = atl::splay_engine;
	sum_rope rope;
	atl::uint32 root = 0;
	for (long long value = 1; value <= 5; ++value) root = engine::join(rope, root, rope.make(value));
	auto const [prefix, rest] = engine::split_at(rope, root, 1);
	auto const [middle, suffix] = engine::split_at(rope, rest, 3);
	rope.is_marked[middle] = true;
	root = engine::join(rope, engine::join(rope, prefix, middle), suffix);
	bool is_first = true;
	engine::for_each(rope, root, [&](atl::uint32 x) {
		if (!is_first) std::cout << ' ';
		is_first = false;
		std::cout << rope.values[x];
	});
	std::cout << '\n';
}
