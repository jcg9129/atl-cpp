/**
 * @file
 * @brief Hull-union throughput over finished hulls and raw circle clouds.
 * Concatenation sorts the union; materialized chains merge full lexicographic
 * sequences; the copodal walk switches outward supports; streamed chains skip
 * bridge dispatch. The library uses bridges only on separated integral hulls.
 * Rows include overlapping, nested, disjoint, identical, interleaved and disk
 * hulls, plus separated-size sweeps. Each loop measures one complete merge.
 * Intel Xeon Platinum 8358, GCC 15.2.0, RelWithDebInfo, assertions off,
 * 2026-11-02, -O2 / -O2 -march=native: 300000-vertex disjoint circles
 * library 0.186 / 0.168 ms vs streamed chains 2.2 / 2.6 ms.
 * Other shapes are not measured yet in the supplied current record.
 * Full measured rows: docs/convex-hull.md.
 */
#include <atl/experimental/geometry/convex_hull_variants.hpp>
#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
	template<typename Call>
		requires std::invocable<Call &>
	void measure(std::string const &name, Call call) {
		BENCHMARK(std::string(name)) { return std::invoke(call); };
	}
}

using atl::geometry::point;
using pt = point<atl::int64>;

namespace {
	/** @brief Every hull pair the merge benchmarks time, from one seeded generator. */
	struct merge_hulls {
		std::vector<pt> circle_a, circle_b, big_a, big_b, nested, disjoint, random_a, random_b, left_half, right_half, big_left_half, big_right_half, interleaved_a, interleaved_b, raw_a, raw_b;
	};

	/** @brief Builds @ref merge_hulls (deterministic: the same shapes in every test case). */
	merge_hulls make_merge_hulls() {
		merge_hulls h;
		std::mt19937_64 rng(1);
		auto const circle_cloud = [&](std::size_t const count, double const cx, double const radius) {
			std::vector<pt> v(count);
			for (auto &q : v) {
				double const t = double(rng() % (1u << 30)) / double(1u << 30) * 6.283185307179586;
				q = {atl::int64(std::llround(cx + radius * std::cos(t))), atl::int64(std::llround(radius * std::sin(t)))};
			}
			return v;
		};
		auto const disk_cloud = [&](std::size_t const count, double const cx, double const radius) {
			std::vector<pt> v(count);
			for (auto &q : v) {
				double const t = double(rng() % (1u << 30)) / double(1u << 30) * 6.283185307179586;
				double const r = radius * std::sqrt(double(rng() % (1u << 30)) / double(1u << 30));
				q = {atl::int64(std::llround(cx + r * std::cos(t))), atl::int64(std::llround(r * std::sin(t)))};
			}
			return v;
		};
		auto const hull = [](std::vector<pt> const &v) { return atl::geometry::convex_hull(v); };

		// 1e6 rounded points on a radius-1e9 circle keep ~1.57e5 hull vertices;
		// at radius 1e15 every point is a vertex
		h.circle_a = hull(circle_cloud(1'000'000, 0, 1e9));
		h.circle_b = hull(circle_cloud(1'000'000, 5e8, 1e9));
		h.big_a = hull(circle_cloud(1'000'000, 0, 1e15));
		h.big_b = hull(circle_cloud(1'000'000, 5e14, 1e15));
		h.nested = hull(circle_cloud(1'000'000, 0, 5e8));
		h.disjoint = hull(circle_cloud(1'000'000, 3e9, 1e9));
		h.random_a = hull(disk_cloud(1'000'000, 0, 1e9));
		h.random_b = hull(disk_cloud(1'000'000, 5e8, 1e9));
		// raw point sets (not hulls), drawn here to keep the generator's
		// deterministic workload order
		h.raw_a = circle_cloud(1'000'000, 0, 1e9);
		h.raw_b = circle_cloud(1'000'000, 5e8, 1e9);
		// Bridge workloads: one cloud cut at its lexicographic median (the D&C
		// merge step: separated hulls), and one cloud dealt alternately into
		// two (interleaved, near-identical hulls)
		auto const halves = [&](double const radius) {
			auto v = circle_cloud(2'000'000, 0, radius);
			std::ranges::sort(v);
			auto const middle = v.begin() + std::ptrdiff_t(v.size() / 2);
			return std::pair{hull(std::vector<pt>(v.begin(), middle)), hull(std::vector<pt>(middle, v.end()))};
		};
		std::tie(h.left_half, h.right_half) = halves(1e9);
		std::tie(h.big_left_half, h.big_right_half) = halves(1e15);
		std::vector<pt> dealt_a, dealt_b;
		for (auto const &q : circle_cloud(2'000'000, 0, 1e9)) (dealt_a.size() <= dealt_b.size() ? dealt_a : dealt_b).push_back(q);
		h.interleaved_a = hull(dealt_a);
		h.interleaved_b = hull(dealt_b);
		return h;
	}

	/** @brief Stratified circle samples retain near the requested vertex count after integral rounding. */
	template<atl::geometry::coordinate T>
	std::vector<point<T>> sized_circle_cloud(std::size_t const count, double const cx, double const radius, std::mt19937_64 &rng) {
		std::vector<point<T>> cloud(count);
		for (std::size_t i = 0; i < count; ++i) {
			double const jitter = double(rng() % (atl::uint64(1) << 30)) * 0x1p-30 * 0.25;
			double const angle = (double(i) + jitter) / double(count) * 6.283185307179586;
			cloud[i] = {T(std::llround(cx + radius * std::cos(angle))), T(std::llround(radius * std::sin(angle)))};
		}
		return cloud;
	}

	/** @brief Time separated circle halves and disjoint circles at a chosen total hull size. */
	template<atl::geometry::coordinate T>
	void separated_merge_rows(std::size_t const total, double const radius, std::string const &type) {
		std::mt19937_64 rng(23);
		auto cloud = sized_circle_cloud<T>(total, 0, radius, rng);
		std::ranges::sort(cloud);
		auto const middle = cloud.begin() + std::ptrdiff_t(cloud.size() / 2);
		auto const left = atl::geometry::convex_hull(std::vector<point<T>>(cloud.begin(), middle));
		auto const right = atl::geometry::convex_hull(std::vector<point<T>>(middle, cloud.end()));
		auto const circle_a = atl::geometry::convex_hull(sized_circle_cloud<T>(total / 2, 0, radius, rng));
		auto const circle_b = atl::geometry::convex_hull(sized_circle_cloud<T>(total / 2, 3 * radius, radius, rng));

		struct shape {
			char const *name;
			std::vector<point<T>> const &a, &b;
		};

		for (shape const s : {shape{"left/right halves", left, right}, shape{"disjoint circles", circle_a, circle_b}}) {
			REQUIRE((std::ranges::max)(s.a) < (std::ranges::min)(s.b));
			REQUIRE(s.a.size() + s.b.size() >= total * 9 / 10);
			REQUIRE((s.a.size() + s.b.size() >= 1024) == (total >= 1024));
			auto const expected = atl::geometry::merge_convex_hulls(s.a, s.b);
			REQUIRE(atl::experimental::merge_convex_hulls_by_streamed_chains(s.a, s.b) == expected);
			std::vector<point<T>> both(s.a);
			both.insert(both.end(), s.b.begin(), s.b.end());
			REQUIRE(atl::geometry::convex_hull(both) == expected);
			std::string const tag = type + " " + s.name + " (" + std::to_string(s.a.size()) + "+"
								  + std::to_string(s.b.size()) + ") total=" + std::to_string(s.a.size() + s.b.size());
			measure("merge_convex_hulls by streamed chains, " + tag, [&] {
				return atl::experimental::merge_convex_hulls_by_streamed_chains(s.a, s.b).size();
			});
			measure("merge_convex_hulls library, " + tag, [&] {
				return atl::geometry::merge_convex_hulls(s.a, s.b).size();
			});
		}
	}
}

TEST_CASE("convex hull merge benchmarks", "[geometry]") {
	auto const hulls = make_merge_hulls();
	auto const &circle_a = hulls.circle_a, &circle_b = hulls.circle_b, &big_a = hulls.big_a, &big_b = hulls.big_b, &nested = hulls.nested, &disjoint = hulls.disjoint, &random_a = hulls.random_a, &random_b = hulls.random_b, &raw_a = hulls.raw_a, &raw_b = hulls.raw_b;
	auto const hull = [](std::vector<pt> const &v) { return atl::geometry::convex_hull(v); };
	/** @brief Rehulls the concatenated inputs instead of merging finished boundaries. */
	auto const merge_convex_hulls_by_concatenation = [](std::vector<pt> const &a, std::vector<pt> const &b) {
		std::vector<pt> both(a);
		both.insert(both.end(), b.begin(), b.end());
		return atl::geometry::convex_hull(both).size();
	};

	struct shape {
		char const *name;
		std::vector<pt> const &a, &b;
		bool has_rehull_rows;
	};

	for (shape const s : {shape{"circle", circle_a, circle_b, true},
						  shape{"big circle", big_a, big_b, true},
						  shape{"nested circles", circle_a, nested, false},
						  shape{"disjoint circles", circle_a, disjoint, false},
						  shape{"identical circle", circle_a, circle_a, false},
						  shape{"random disk hulls", random_a, random_b, false}}) {
		std::string const tag = std::string(s.name) + " (" + std::to_string(s.a.size()) + "+"
							  + std::to_string(s.b.size()) + ")";
		std::vector<pt> both(s.a);
		both.insert(both.end(), s.b.begin(), s.b.end());
		auto const expected = atl::geometry::convex_hull(both);
		REQUIRE(atl::geometry::merge_convex_hulls(s.a, s.b) == expected);
		REQUIRE(atl::experimental::merge_convex_hulls_by_materialized_chains(s.a, s.b) == expected);
		REQUIRE(atl::experimental::merge_convex_hulls_by_copodal_walk(s.a, s.b) == expected);
		measure("by concatenation, " + tag, [&] { return merge_convex_hulls_by_concatenation(s.a, s.b); });
		measure("library, " + tag, [&] { return atl::geometry::merge_convex_hulls(s.a, s.b).size(); });
		measure("by materialized chains, " + tag, [&] {
			return atl::experimental::merge_convex_hulls_by_materialized_chains(s.a, s.b).size();
		});
		measure("by copodal walk, " + tag, [&] {
			return atl::experimental::merge_convex_hulls_by_copodal_walk(s.a, s.b).size();
		});
		if (!s.has_rehull_rows) continue;
		measure("rehull + merge_convex_hulls, " + tag, [&] {
			return atl::geometry::merge_convex_hulls(hull(s.a), hull(s.b)).size();
		});
		measure("by materialized chains with rehull, " + tag, [&] {
			return atl::experimental::convex_hull_union_by_materialized_chains(s.a, s.b).size();
		});
		measure("by copodal walk with rehull, " + tag, [&] {
			return atl::experimental::convex_hull_union_by_copodal_walk(s.a, s.b).size();
		});
	}

	// raw point sets (not hulls): what a caller holding two clouds can do
	{
		std::vector<pt> both(raw_a);
		both.insert(both.end(), raw_b.begin(), raw_b.end());
		auto const expected = atl::geometry::convex_hull(both);
		REQUIRE(atl::geometry::merge_convex_hulls(hull(raw_a), hull(raw_b)) == expected);
		REQUIRE(atl::experimental::convex_hull_union_by_materialized_chains(raw_a, raw_b) == expected);
		REQUIRE(atl::experimental::convex_hull_union_by_copodal_walk(raw_a, raw_b) == expected);
	}
	measure("by concatenation, raw circle clouds 1e6+1e6", [&] { return merge_convex_hulls_by_concatenation(raw_a, raw_b); });
	measure("rehull + merge_convex_hulls, raw circle clouds 1e6+1e6", [&] {
		return atl::geometry::merge_convex_hulls(hull(raw_a), hull(raw_b)).size();
	});
	measure("by materialized chains with rehull, raw circle clouds 1e6+1e6", [&] {
		return atl::experimental::convex_hull_union_by_materialized_chains(raw_a, raw_b).size();
	});
	measure("by copodal walk with rehull, raw circle clouds 1e6+1e6", [&] {
		return atl::experimental::convex_hull_union_by_copodal_walk(raw_a, raw_b).size();
	});
}

TEST_CASE("convex hull merge benchmarks: bridged", "[geometry]") {
	auto const h = make_merge_hulls();

	struct shape {
		char const *name;
		std::vector<pt> const &a, &b;
	};

	for (shape const s : {shape{"circle", h.circle_a, h.circle_b}, shape{"big circle", h.big_a, h.big_b},
						  shape{"nested circles", h.circle_a, h.nested},
						  shape{"disjoint circles", h.circle_a, h.disjoint},
						  shape{"identical circle", h.circle_a, h.circle_a},
						  shape{"random disk hulls", h.random_a, h.random_b},
						  shape{"left/right halves", h.left_half, h.right_half},
						  shape{"big left/right halves", h.big_left_half, h.big_right_half},
						  shape{"interleaved halves", h.interleaved_a, h.interleaved_b}}) {
		std::string const tag = std::string(s.name) + " (" + std::to_string(s.a.size()) + "+"
							  + std::to_string(s.b.size()) + ")";
		REQUIRE(atl::experimental::merge_convex_hulls_by_streamed_chains(s.a, s.b) == atl::geometry::merge_convex_hulls(s.a, s.b));
		measure("merge_convex_hulls by streamed chains, " + tag, [&] { return atl::experimental::merge_convex_hulls_by_streamed_chains(s.a, s.b).size(); });
		measure("merge_convex_hulls library, " + tag, [&] {
			return atl::geometry::merge_convex_hulls(s.a, s.b).size();
		});
	}
}

TEST_CASE("convex hull merge benchmarks: separated dispatch sizes", "[geometry]") {
	for (std::size_t const total : {std::size_t(256), std::size_t(512), std::size_t(1024), std::size_t(2048), std::size_t(8192), std::size_t(32768), std::size_t(100'000), std::size_t(300'000)})
		separated_merge_rows<atl::int64>(total, 1e15, "int64");
	for (std::size_t const total : {std::size_t(256), std::size_t(1024), std::size_t(8192)})
		separated_merge_rows<atl::int32>(total, 1e8, "int32");
}
