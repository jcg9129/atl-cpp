/**
 * @file
 * @brief Races Fenwick build by division and modulo against by odometer and the library constructor.
 * Both standalone passes copy the same flat cells; the constructor consumes the same input.
 * Measures build throughput on 1-D 1e7, 2-D 3000x3000 and 3-D 200-cubed shapes.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-15,
 * -O2 / -O2 -march=native (also -g -DNDEBUG -std=c++23): odometer takes
 * 70.0 / 71.3 ms versus division/modulo 133 / 134 ms on 1-D; complete tables in docs/fenwick-tree.md.
 */

#include <atl/ds/binary_indexed_tree.hpp>
#include <atl/types/alias.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <cstddef>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using atl::int64;

namespace {
	template<std::size_t Dim>
	std::array<std::size_t, Dim> strides_of(std::array<std::size_t, Dim> const &sizes) {
		std::array<std::size_t, Dim> st{};
		std::size_t total = 1;
		for (std::size_t d = Dim; d-- > 0;) st[d] = total, total *= sizes[d];
		return st;
	}

	/** @brief Build using per-cell division and modulo. */
	template<std::size_t Dim>
	int64 build_by_division_and_modulo(std::vector<int64> data, std::array<std::size_t, Dim> const &sizes) {
		auto const strides = strides_of(sizes);
		std::size_t const total = data.size();
		for (std::size_t d = 0; d < Dim; ++d)
			for (std::size_t f = 0; f < total; ++f) {
				std::size_t const index = (f / strides[d]) % sizes[d];
				if (std::size_t const parent = index | (index + 1); parent < sizes[d]) {
					std::size_t const pf = f + (parent - index) * strides[d];
					data[pf] += data[f];
				}
			}
		return data.back();
	}

	/** @brief Build with axis counters instead of per-cell division. */
	template<std::size_t Dim>
	int64 build_by_odometer(std::vector<int64> data, std::array<std::size_t, Dim> const &sizes) {
		auto const strides = strides_of(sizes);
		std::size_t const total = data.size();
		for (std::size_t d = 0; d < Dim; ++d) {
			std::size_t const stride = strides[d], extent = sizes[d], span = extent * stride;
			for (std::size_t block = 0; block < total; block += span)
				for (std::size_t index = 0; index < extent; ++index) {
					std::size_t const parent = index | (index + 1);
					if (parent >= extent) continue;
					std::size_t const from = block + index * stride, to = block + parent * stride;
					for (std::size_t j = 0; j < stride; ++j) data[to + j] += data[from + j];
				}
		}
		return data.back();
	}

	/** @brief Time one build callable; each lambda keeps its contender statically inlinable. */
	template<typename Build>
		requires std::is_invocable_r_v<int64, Build &>
	void measure_build(std::string name, Build build) {
		BENCHMARK(std::move(name)) { return build(); };
	}

	template<std::size_t Dim>
	void race(std::string const &name, std::array<std::size_t, Dim> const &sizes) {
		std::size_t total = 1;
		for (auto s : sizes) total *= s;
		std::mt19937_64 rng(4);
		std::vector<int64> cells(total);
		for (auto &c : cells) c = int64(rng() % 1000);
		REQUIRE(build_by_division_and_modulo(cells, sizes) == build_by_odometer(cells, sizes));
		measure_build("build " + name + ", by division and modulo", [&] { return build_by_division_and_modulo(cells, sizes); });
		measure_build("build " + name + ", by odometer", [&] { return build_by_odometer(cells, sizes); });
		auto const library = [&] {
			atl::binary_indexed_tree<int64, Dim> const t(sizes, cells);
			return [&]<std::size_t... I>(std::index_sequence<I...>)
				requires(sizeof...(I) == Dim)
			{
				return t.prefix_fold(sizes[I]...);
			}(std::make_index_sequence<Dim>{});
		};
		REQUIRE(library() == std::accumulate(cells.begin(), cells.end(), int64{}));
		measure_build("build " + name + ", library", library);
	}
}

TEST_CASE("binary_indexed_tree build indexing strategies", "[binary_indexed_tree][race]") {
	race<1>("1-D n=1e7", {10'000'000});
	race<2>("2-D 3000x3000", {3000, 3000});
	race<3>("3-D 200^3", {200, 200, 200});
}
