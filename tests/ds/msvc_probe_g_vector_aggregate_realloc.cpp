// Temporary MSVC constant-evaluation probe for ds-2 (removed before merge).
#include <atl/ds/dynamic_lazy_segment_tree.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <vector>

namespace {
	struct sum_width {
		int sum = 0, width = 0;
	};

	struct affine {
		int scale = 1, bias = 0;
		constexpr bool operator==(affine const &) const = default;
	};

	constexpr auto op = [](sum_width x, sum_width y) { return sum_width{x.sum + y.sum, x.width + y.width}; };
	constexpr auto mapping = [](affine f, sum_width x) { return sum_width{f.scale * x.sum + f.bias * x.width, x.width}; };
	constexpr auto compose = [](affine f, affine g) { return affine{f.scale * g.scale, f.scale * g.bias + f.bias}; };
	constexpr auto fill = [](std::size_t width) { return sum_width{int(width), int(width)}; };
	using tree = atl::dynamic_lazy_segment_tree<sum_width, affine, decltype(op), decltype(mapping), decltype(compose), decltype(fill)>;
}

namespace {
	struct node {
		sum_width aggregate;
		affine tag;
		unsigned left = 0, right = 0;
	};
}

static_assert([] {
	affine const identity{};
	std::vector<node> nodes;
	nodes.reserve(2);
	nodes.push_back({sum_width{}, identity, 0, 0});
	for (int i = 0; i < 9; ++i) nodes.push_back({sum_width{i, 1}, identity, 0, 0});
	affine acc = identity;
	for (node const &nd : nodes) acc = compose(acc, nd.tag);
	return acc.scale == 1 && acc.bias == 0;
}());

TEST_CASE("msvc probe g_vector_aggregate_realloc", "[msvc_probe]") { SUCCEED(); }
