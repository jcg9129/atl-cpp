/**
 * @file
 * @brief Races path-copying persistent and in-place mutable treap sequences on the same
 * linear stream of 2e5 insert/fold steps, starting with 1e5 elements. Measures
 * complete-script throughput including construction and allocation, with fresh pools
 * per sample. Not measured yet; docs/persistent-treap-sequence.md.
 */

#include <atl/ds/persistent_treap_sequence.hpp>
#include <atl/ds/treap_sequence.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {
	/** @brief Time a complete workload through a direct, constrained callable. */
	template<typename Run>
		requires std::invocable<Run &>
	void measure(std::string const &label, Run run) {
		BENCHMARK(std::string(label)) { return run(); };
	}
}

using ll = long long;

TEST_CASE("persistent_treap_sequence benchmarks", "[persistent_treap_sequence]") {
	constexpr int n = 100'000, m = 200'000;
	std::mt19937 rng(1);
	std::vector<std::size_t> pos(m);
	std::vector<ll> val(m);
	std::vector<std::size_t> lo(m), hi(m);
	for (int i = 0; i < m; ++i) {
		pos[i] = rng() % std::size_t(n + i + 1);
		val[i] = ll(rng() % 1000);
		std::size_t a = rng() % std::size_t(n), b = rng() % std::size_t(n);
		if (a > b) std::swap(a, b);
		lo[i] = a;
		hi[i] = b + 1;
	}

	auto const run_library = [&] {
		atl::persistent_treap_sequence cur(std::vector<ll>(n, 1), ll(0), std::plus<>{});
		cur.reserve(m / 2);
		ll acc = 0;
		for (int i = 0; i < m; ++i)
			if (i % 2 == 0)
				cur = cur.insert(pos[i] % (cur.size() + 1), val[i]);
			else
				acc += cur.fold(lo[i], hi[i]);
		return acc;
	};
	auto const sequence_by_mutable_treap = [&] {
		atl::treap_sequence t(std::vector<ll>(n, 1), ll(0), std::plus<>{});
		ll acc = 0;
		for (int i = 0; i < m; ++i)
			if (i % 2 == 0)
				t.insert(pos[i] % (t.size() + 1), val[i]);
			else
				acc += t.fold(lo[i], hi[i]);
		return acc;
	};
	REQUIRE(run_library() == sequence_by_mutable_treap());
	measure("library persistent -- 2e5 mixed ops", run_library);
	measure("by mutable treap -- 2e5 mixed ops", sequence_by_mutable_treap);
}
