/** @file
 * @brief Partition materialization throughput: library groups() allocates nested
 * vectors; library grouped() uses flat offsets and members. Each loop builds and
 * scans all groups at n=1e6: singletons, groups of 2-4, one group, sqrt(n) groups.
 * Intel Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, -O2 -g -DNDEBUG
 * -std=c++23, also -march=native, 2026-10-23: singletons 198/196 ms nested,
 * 22.5/23.0 ms flat. Full tables: docs/graph.md.
 */
#include <atl/graph/partition.hpp>
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
	template<typename Run>
		requires std::invocable<Run &>
	void measure(std::string name, Run run) {
		BENCHMARK(std::move(name)) { return std::invoke(run); };
	}
}

namespace g = atl::graph;
using atl::uint32;
using atl::uint64;

TEST_CASE("partition groups", "[graph]") {
	std::size_t const n = 1'000'000;
	std::mt19937 rng(1);

	auto race = [&](std::string const &name, g::vertex_partition const &p) {
		auto const nested = p.groups();
		auto const flat = p.grouped();
		if (flat.count() != nested.size()) FAIL("grouped() disagrees with groups()");
		for (std::size_t c = 0; c < nested.size(); ++c)
			if (!std::ranges::equal(flat[c], nested[c])) FAIL("grouped() disagrees with groups()");
		measure(name + " | groups() (nested)", [&] {
			uint64 acc = 0;
			auto const out = p.groups();
			for (std::size_t c = 0; c < out.size(); ++c)
				for (auto const v : out[c]) acc += v ^ c;
			return acc;
		});
		measure(name + " | grouped() (flat)", [&] {
			uint64 acc = 0;
			auto const out = p.grouped();
			for (std::size_t c = 0; c < out.count(); ++c)
				for (auto const v : out[c]) acc += v ^ c;
			return acc;
		});
	};

	auto shuffled = [&](std::vector<uint32> component, std::size_t count) {
		for (std::size_t i = component.size(); i > 1; --i) std::swap(component[i - 1], component[rng() % i]);
		return g::vertex_partition{std::move(component), count};
	};
	{
		std::vector<uint32> component(n);
		for (uint32 v = 0; v < n; ++v) component[v] = v;
		race("all singletons", shuffled(component, n));
	}
	{
		std::vector<uint32> component(n);
		uint32 id = 0;
		for (std::size_t v = 0; v < n;) {
			std::size_t const size = 2 + rng() % 3;
			for (std::size_t i = 0; i < size && v < n; ++i) component[v++] = id;
			++id;
		}
		race("components of 2-4", shuffled(component, id));
	}
	race("one giant component", g::vertex_partition{std::vector<uint32>(n, 0), 1});
	{
		auto const side = std::size_t(std::sqrt(double(n)));
		std::vector<uint32> component(n);
		for (std::size_t v = 0; v < n; ++v) component[v] = uint32(v / side);
		race("sqrt(n) components of sqrt(n)", shuffled(component, (n + side - 1) / side));
	}
}
