/**
 * @file
 * @brief Throughput of subset transforms at 2^16 and 2^20 entries and bitwise
 * products at 2^20 entries. Transform samples copy their input, then run the
 * library blocked radix-2 order, bit-outer radix-2, radix-4, or blocked radix-4;
 * copy-only isolates allocation/copy cost. Subset products compare mask-major
 * restricted ranks with full rank-major arrays. All contenders are checked.
 * Source: this file; Xeon Platinum 8358, GCC 15.2, RelWithDebInfo, assertions
 * off, 2026-10-15, -O2 -g -DNDEBUG -std=c++23, with/without -march=native.
 * Library subset product: 770 / 733 ms; by rank major: 1110 / 1111 ms
 * (plain / native). Full tables: docs/bitwise-convolution.md.
 */
#include <atl/experimental/nt/bitwise_convolution_variants.hpp>
#include <atl/nt/bitwise_convolution.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

using atl::uint32;
using atl::uint64;
namespace ex = atl::experimental;
using Z = atl::mod_int<uint32, 998244353>;

namespace {
	/** @brief Race the four loop orders on one element type and butterfly at
	 *         2^bits elements. Each sample transforms a fresh copy. */
	template<atl::additive_group T, atl::detail::butterfly_of<T> Butterfly>
	void race(std::string const &shape, std::size_t bits, Butterfly butterfly) {
		std::mt19937_64 rng(bits);
		std::vector<T> source(std::size_t(1) << bits);
		for (auto &x : source) x = T(uint64(rng()));
		std::vector<T> f;
		auto const measure = [&]<typename Transform>(std::string const &method, Transform transform)
			requires std::invocable<Transform &, std::vector<T> &>
		{
			BENCHMARK(method + " -- " + shape) {
				f = source;
				transform(f);
				return f[1];
			};
		};
		auto library = source, bit_outer = source, radix4 = source, blocked_radix4 = source;
		atl::detail::butterfly_transform(library.begin(), bits, butterfly);
		ex::transform_by_bit_outer_radix2(bit_outer.begin(), bits, butterfly);
		ex::transform_by_radix4(radix4.begin(), bits, butterfly);
		ex::transform_by_blocked_radix4(blocked_radix4.begin(), bits, butterfly);
		REQUIRE(library == bit_outer);
		REQUIRE(library == radix4);
		REQUIRE(library == blocked_radix4);
		measure("library", [&](std::vector<T> &v) { atl::detail::butterfly_transform(v.begin(), bits, butterfly); });
		measure("by bit outer radix2", [&](std::vector<T> &v) { ex::transform_by_bit_outer_radix2(v.begin(), bits, butterfly); });
		measure("by radix4", [&](std::vector<T> &v) { ex::transform_by_radix4(v.begin(), bits, butterfly); });
		measure("by blocked radix4", [&](std::vector<T> &v) { ex::transform_by_blocked_radix4(v.begin(), bits, butterfly); });
		measure("copy only", [](std::vector<T> &) {});
	}
}

TEST_CASE("bitwise transforms: loop-order race", "[bitwise_convolution]") {
	race<Z>("subset zeta, mod_int 998244353, n = 20", 20, atl::detail::subset_zeta_butterfly{});
	race<uint64>("subset zeta, uint64, n = 20", 20, atl::detail::subset_zeta_butterfly{});
	race<Z>("walsh-hadamard, mod_int 998244353, n = 20", 20, atl::detail::walsh_hadamard_butterfly{});
	race<uint64>("walsh-hadamard, uint64, n = 20", 20, atl::detail::walsh_hadamard_butterfly{});
	race<Z>("subset zeta, mod_int 998244353, n = 16", 16, atl::detail::subset_zeta_butterfly{});
}

TEST_CASE("bitwise convolutions at n = 20", "[bitwise_convolution]") {
	std::mt19937_64 rng(20);
	std::size_t const size = std::size_t(1) << 20;
	std::vector<Z> a(size), b(size);
	for (auto &x : a) x = Z(uint64(rng()));
	for (auto &x : b) x = Z(uint64(rng()));
	BENCHMARK("or_convolution -- mod_int, n = 20") { return atl::or_convolution(a, b)[1]; };
	BENCHMARK("xor_convolution -- mod_int, n = 20") { return atl::xor_convolution(a, b)[1]; };
	REQUIRE(atl::subset_convolution(a, b) == ex::subset_convolution_by_rank_major(a, b));
	BENCHMARK("subset_convolution library -- mod_int, n = 20") {
		return atl::subset_convolution(a, b)[1];
	};
	BENCHMARK("subset_convolution by rank major -- mod_int, n = 20") {
		return ex::subset_convolution_by_rank_major(a, b)[1];
	};
}
