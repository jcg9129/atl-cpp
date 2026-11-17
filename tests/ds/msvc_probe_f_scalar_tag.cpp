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

static_assert([] {
	auto sop = [](sum_width x, sum_width y) { return sum_width{x.sum + y.sum, x.width + y.width}; };
	auto smap = [](int f, sum_width x) { return sum_width{x.sum + f * x.width, x.width}; };
	auto scomp = [](int f, int g) { return f + g; };
	atl::dynamic_lazy_segment_tree st(7, sum_width{}, 0, sop, smap, scomp, fill);
	st.apply(1, 6, 1);
	return st.fold().sum == 12 && st.fold(1, 6).sum == 10;
}());

TEST_CASE("msvc probe f_scalar_tag", "[msvc_probe]") { SUCCEED(); }
