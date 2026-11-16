#include <atl/bits/xor_basis.hpp>
#include <atl/experimental/bits/xor_basis_variants.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

using atl::uint32;
using atl::uint64;
namespace ex = atl::experimental;

namespace {
	/** @brief @p count words from a random subspace of dimension @p dim with
	 *         values below 2^width. */
	template<atl::unsigned_integral Word>
	std::vector<Word> subspace_words(std::mt19937_64 &rng, std::size_t count, std::size_t dim, std::size_t width) {
		Word const mask = width >= sizeof(Word) * 8 ? Word(~Word(0)) : Word((Word(1) << width) - 1);
		std::vector<Word> gens(dim);
		for (auto &g : gens) g = Word(rng()) & mask;
		std::vector<Word> words(count);
		for (auto &w : words) {
			w = 0;
			for (Word const g : gens)
				if (rng() & 1) w ^= g;
		}
		return words;
	}

	template<atl::unsigned_integral Word>
	void race(std::string const &shape, std::vector<Word> const &words, std::vector<Word> const &probes) {
		BENCHMARK("reduced (shipped) -- insert -- " + shape) {
			atl::xor_basis<Word> b;
			for (Word const w : words) b.insert(w);
			return b.rank();
		};
		BENCHMARK("echelon jump -- insert -- " + shape) {
			ex::echelon_xor_basis<Word> b;
			for (Word const w : words) b.insert(w);
			return b.rank();
		};
		BENCHMARK("echelon scan -- insert -- " + shape) {
			ex::scanning_xor_basis<Word> b;
			for (Word const w : words) b.insert(w);
			return b.rank();
		};

		atl::xor_basis<Word> reduced(words);
		ex::echelon_xor_basis<Word> jump;
		ex::scanning_xor_basis<Word> scan;
		for (Word const w : words) jump.insert(w), scan.insert(w);
		BENCHMARK("reduced (shipped) -- max_xor -- " + shape) {
			Word acc = 0;
			for (Word const p : probes) acc ^= reduced.max_xor(p);
			return acc;
		};
		BENCHMARK("echelon jump -- max_xor -- " + shape) {
			Word acc = 0;
			for (Word const p : probes) acc ^= jump.max_xor(p);
			return acc;
		};
		BENCHMARK("echelon scan -- max_xor -- " + shape) {
			Word acc = 0;
			for (Word const p : probes) acc ^= scan.max_xor(p);
			return acc;
		};
		BENCHMARK("reduced (shipped) -- contains -- " + shape) {
			std::size_t hits = 0;
			for (Word const p : probes) hits += reduced.contains(p);
			return hits;
		};
		BENCHMARK("echelon jump -- contains -- " + shape) {
			std::size_t hits = 0;
			for (Word const p : probes) hits += jump.contains(p);
			return hits;
		};
		BENCHMARK("echelon scan -- contains -- " + shape) {
			std::size_t hits = 0;
			for (Word const p : probes) hits += scan.contains(p);
			return hits;
		};
	}
}

TEST_CASE("xor_basis race: reduced vs plain echelon", "[xor_basis]") {
	std::mt19937_64 rng(2026);
	std::size_t const n = 1'000'000;
	{
		std::vector<uint64> words(n), probes(n);
		for (auto &w : words) w = rng();
		for (auto &p : probes) p = rng();
		race<uint64>("1e6 random uint64 (full rank)", words, probes);
	}
	{
		// 30-bit values from a rank-20 subspace: long elimination chains, and
		// half the probes are members by construction
		auto const words = subspace_words<uint32>(rng, n, 20, 30);
		auto probes = subspace_words<uint32>(rng, n, 20, 30);
		for (std::size_t i = 0; i < n; i += 2) probes[i] = words[i];
		race<uint32>("1e6 uint32, rank 20 of 30 bits", words, probes);
	}
	{
		auto const words = subspace_words<uint64>(rng, n, 40, 64);
		std::vector<uint64> probes(n);
		for (auto &p : probes) p = rng();
		race<uint64>("1e6 uint64, rank 40", words, probes);
	}
}

TEST_CASE("xor_basis: order queries", "[xor_basis]") {
	std::mt19937_64 rng(7);
	std::size_t const q = 1'000'000;
	auto const words = subspace_words<uint64>(rng, 1000, 50, 64);
	atl::xor_basis<uint64> const basis(words);
	std::vector<uint64> ks(q), bounds(q);
	for (auto &k : ks) k = rng() >> (64 - basis.rank());
	for (auto &b : bounds) b = rng();
	BENCHMARK("kth_smallest -- 1e6 queries, rank 50") {
		uint64 acc = 0;
		for (uint64 const k : ks) acc ^= basis.kth_smallest(k);
		return acc;
	};
	BENCHMARK("count_below greedy descent -- 1e6 queries, rank 50") {
		uint64 acc = 0;
		for (uint64 const b : bounds) acc += ex::greedy_count_below(basis, b);
		return acc;
	};
	BENCHMARK("count_below pivot pattern (shipped) -- 1e6 queries, rank 50") {
		uint64 acc = 0;
		for (uint64 const b : bounds) acc += basis.count_below(b);
		return acc;
	};
}

TEST_CASE("stamped_xor_basis: range max-xor, offline by right end", "[xor_basis]") {
	// the range-query shape of "max xor of a subsequence of a[l, r)": n = q = 5e5,
	// 20-bit values; queries sorted by r once, outside the timed body
	std::mt19937_64 rng(1100);
	std::size_t const n = 500'000, q = 500'000;
	std::vector<uint32> a(n);
	for (auto &x : a) x = uint32(rng() % (1u << 20));

	struct range_query {
		uint32 l, r;
	};

	std::vector<range_query> queries(q);
	for (auto &[l, r] : queries) {
		l = uint32(rng() % n), r = uint32(rng() % n);
		if (l > r) std::swap(l, r);
		++r;
	}
	std::ranges::sort(queries, {}, &range_query::r);
	BENCHMARK("stamped_xor_basis -- n = q = 5e5, 20-bit") {
		atl::stamped_xor_basis<uint32> basis;
		uint64 acc = 0;
		std::size_t next = 0;
		for (uint32 r = 1; r <= n; ++r) {
			basis.insert(a[r - 1], r - 1);
			for (; next < q && queries[next].r == r; ++next) acc += basis.max_xor(queries[next].l);
		}
		return acc;
	};
}
