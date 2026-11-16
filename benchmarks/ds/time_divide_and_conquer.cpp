/**
 * @file
 * Race between `atl::time_divide_and_conquer` (the library, automatic sparse ranks)
 * and engines by recursive vectors, unpruned CSR, configurable pruning/stretching,
 * dense CSR, frame skipping, or ranked frame skipping. Restore modes use
 * rollback_dsu undo or persistent_dsu copies. Loops measure whole-log throughput,
 * including setup, on churn with dense/sparse queries and long/short lifetimes;
 * lifetimes_of-only rows isolate event pairing.
 * Result (Xeon Platinum 8280, GCC 15.2, `-O2` and `-O2 -march=native`,
 * 2026-11-01): at one percent queries dense CSR takes 34.6/32.9 ms by rollback
 * versus 13.8/13.9 ms by forced rank compression. The integrated automatic
 * library dispatch is not measured yet. Numbers: docs/time-divide-and-conquer.md.
 */
#include <atl/ds/persistent_dsu.hpp>
#include <atl/ds/rollback_dsu.hpp>
#include <atl/ds/rollback_weighted_dsu.hpp>
#include <atl/ds/time_divide_and_conquer.hpp>
#include <atl/experimental/ds/time_divide_and_conquer_variants.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

using atl::uint32;
using atl::uint64;
using atl::uint8;

namespace {
	using vertex_pair = std::pair<uint32, uint32>;
	using event = atl::item_event<vertex_pair>;

	/** @brief One connectivity instance: the log, its lifetimes, and each item's endpoints. */
	struct workload {
		std::size_t n;
		std::vector<event> events;
		std::vector<atl::item_lifetime> lifetimes;
		std::vector<vertex_pair> edge; ///< per item
		std::vector<uint8> is_query;   ///< per time
	};

	/** @brief Fills the derived fields from `w.events`. */
	void finish(workload &w) {
		w.lifetimes = atl::lifetimes_of(w.events);
		w.edge.clear();
		for (auto const &l : w.lifetimes) w.edge.push_back(w.events[l.begin].key);
		w.is_query.clear();
		for (auto const &e : w.events) w.is_query.push_back(e.kind == atl::event_kind::query);
	}

	/** @brief A uniform random vertex pair, normalized to `u <= v`. */
	vertex_pair random_edge(std::size_t n, std::mt19937_64 &rng) {
		uint32 u = uint32(rng() % n), v = uint32(rng() % n);
		if (u > v) std::swap(u, v);
		return {u, v};
	}

	/**
	 * @brief `q` events on `n` vertices: a query with probability `query_share`,
	 *        otherwise a link or, half the time, a cut of a random live edge; an
	 *        edge is mortal (may be cut) with probability `mortal_share`.
	 */
	workload make_churn_workload(std::size_t n, std::size_t q, double query_share, double mortal_share, std::mt19937_64 &rng) {
		workload w{n, {}, {}, {}, {}};
		std::vector<vertex_pair> mortal;
		std::uniform_real_distribution<double> coin(0, 1);
		while (w.events.size() < q) {
			if (coin(rng) < query_share)
				w.events.push_back({atl::event_kind::query, {}});
			else if (!mortal.empty() && rng() % 2) {
				std::size_t const j = rng() % mortal.size();
				w.events.push_back({atl::event_kind::remove, mortal[j]});
				mortal[j] = mortal.back(), mortal.pop_back();
			} else {
				auto const e = random_edge(n, rng);
				if (coin(rng) < mortal_share) mortal.push_back(e);
				w.events.push_back({atl::event_kind::add, e});
			}
		}
		finish(w);
		return w;
	}

	/** @brief Every link is cut right after one query: link, query, cut. */
	workload short_lived(std::size_t n, std::size_t q, std::mt19937_64 &rng) {
		workload w{n, {}, {}, {}, {}};
		while (w.events.size() + 3 <= q) {
			auto const e = random_edge(n, rng);
			w.events.push_back({atl::event_kind::add, e});
			w.events.push_back({atl::event_kind::query, {}});
			w.events.push_back({atl::event_kind::remove, e});
		}
		finish(w);
		return w;
	}

	/** @brief The engines raced, one call shape. */
	enum class engine { recursive,
						unpruned,
						unstretched,
						library,
						skipping,
						dense_csr,
						rank_compressed_skipping };

	template<typename... Args>
		requires requires(Args &&...args) { atl::time_divide_and_conquer(std::forward<Args>(args)...); }
	void apply_time_engine(engine e, Args &&...args) {
		switch (e) {
		case engine::recursive:
			atl::experimental::time_divide_and_conquer_by_recursive_vectors(std::forward<Args>(args)...);
			break;
		case engine::unpruned:
			atl::experimental::time_divide_and_conquer_by_unpruned_csr(std::forward<Args>(args)...);
			break;
		case engine::unstretched:
			atl::experimental::time_divide_and_conquer_by_configurable_csr<true, false>(std::forward<Args>(args)...);
			break;
		case engine::library:
			atl::time_divide_and_conquer(std::forward<Args>(args)...);
			break;
		case engine::skipping:
			atl::experimental::time_divide_and_conquer_by_frame_skipping(std::forward<Args>(args)...);
			break;
		case engine::dense_csr:
			atl::experimental::time_divide_and_conquer_by_dense_csr(std::forward<Args>(args)...);
			break;
		case engine::rank_compressed_skipping:
			atl::experimental::time_divide_and_conquer_by_ranked_frame_skipping(std::forward<Args>(args)...);
			break;
		}
	}

	/** @brief Connectivity with rollback_dsu (default by_rollback): sum of the component counts at the queries. */
	uint64 sum_connectivity_counts(engine e, workload const &w) {
		atl::rollback_dsu d(w.n);
		uint64 acc = 0;
		apply_time_engine(
			e, d, w.events.size(), w.lifetimes,
			[&](atl::rollback_dsu &s, uint32 i) { s.unite(w.edge[i].first, w.edge[i].second); },
			[&](atl::rollback_dsu &s, uint32 t) { acc += s.count() * (t | 1); }, [&](uint32 t) { return w.is_query[t] != 0; });
		return acc;
	}

	/** @brief The same answers from persistent_dsu by copy (default through the library engine). */
	uint64 sum_persistent_connectivity_counts(workload const &w, engine e = engine::library) {
		atl::persistent_dsu<> d(w.n);
		uint64 acc = 0;
		apply_time_engine(
			e, d, w.events.size(), w.lifetimes,
			[&](atl::persistent_dsu<> &s, uint32 i) { s = std::move(s).unite(w.edge[i].first, w.edge[i].second); },
			[&](atl::persistent_dsu<> &s, uint32 t) { acc += s.count() * (t | 1); }, [&](uint32 t) { return w.is_query[t] != 0; });
		return acc;
	}

	/** @brief Every engine on one connectivity shape; answers cross-checked before timing. */
	void benchmark_connectivity(std::string const &shape, workload const &w, bool has_persistent) {
		uint64 const expected = sum_connectivity_counts(engine::library, w);
		REQUIRE(sum_connectivity_counts(engine::recursive, w) == expected);
		REQUIRE(sum_connectivity_counts(engine::unpruned, w) == expected);
		REQUIRE(sum_connectivity_counts(engine::unstretched, w) == expected);
		REQUIRE(sum_connectivity_counts(engine::dense_csr, w) == expected);
		if (has_persistent) REQUIRE(sum_persistent_connectivity_counts(w) == expected);
		BENCHMARK(shape + " -- by recursive vectors") { return sum_connectivity_counts(engine::recursive, w); };
		BENCHMARK(shape + " -- by unpruned CSR") { return sum_connectivity_counts(engine::unpruned, w); };
		BENCHMARK(shape + " -- automatic sparse ranks (library)") { return sum_connectivity_counts(engine::library, w); };
		BENCHMARK(shape + " -- by dense CSR") { return sum_connectivity_counts(engine::dense_csr, w); };
		BENCHMARK(shape + " -- by configurable CSR (pruned, unstretched)") { return sum_connectivity_counts(engine::unstretched, w); };
		if (has_persistent)
			BENCHMARK(shape + " -- library, persistent_dsu by_copy") { return sum_persistent_connectivity_counts(w); };
	}

}

/** @brief Compare offline deletion engines and restoration modes. */
TEST_CASE("time_divide_and_conquer race", "[time_divide_and_conquer][race]") {
	std::mt19937_64 rng(2026'10'24);
	constexpr std::size_t n = 300'000, q = 300'000;
	{
		auto const w = make_churn_workload(n, q, 0.5, 1.0, rng);
		benchmark_connectivity("churn n=q=3e5, 50% queries", w, true);
		BENCHMARK("churn n=q=3e5, 50% queries -- lifetimes_of alone") { return atl::lifetimes_of(w.events).size(); };
	}
	{
		auto const w = make_churn_workload(n, q, 0.01, 1.0, rng);
		benchmark_connectivity("churn n=q=3e5, 1% queries", w, true);
		BENCHMARK("churn n=q=3e5, 1% queries -- lifetimes_of alone") { return atl::lifetimes_of(w.events).size(); };
	}
	benchmark_connectivity("long-lived n=q=3e5 (90% never cut), 50% queries", make_churn_workload(n, q, 0.5, 0.1, rng), false);
	benchmark_connectivity("short-lived n=q=3e5 (link, query, cut)", short_lived(n, q, rng), false);
}

/**
 * @brief Sparse-query comparison: the dense CSR baseline and skipping contenders
 *        next to automatic sparse rank compression (library), churn at
 *        50 / 10 / 1% queries by rollback and by copy, with connectivity probes.
 */
TEST_CASE("time_divide_and_conquer sparse ranks and query skipping", "[time_divide_and_conquer][comparison]") {
	std::mt19937_64 rng(2026'10'29);
	constexpr std::size_t n = 300'000, q = 300'000;
	for (auto const &[share, label] : {std::pair{0.5, "50%"}, std::pair{0.1, "10%"}, std::pair{0.01, "1%"}}) {
		auto const w = make_churn_workload(n, q, share, 1.0, rng);
		std::string const shape = std::string("sparse-query churn n=q=3e5, ") + label + " nominal queries (Q=" + std::to_string(std::ranges::count(w.is_query, uint8(1))) + ")";
		uint64 const expected = sum_connectivity_counts(engine::library, w);
		REQUIRE(sum_persistent_connectivity_counts(w) == expected);
		for (engine e : {engine::skipping, engine::dense_csr, engine::rank_compressed_skipping}) {
			REQUIRE(sum_connectivity_counts(e, w) == expected);
			REQUIRE(sum_persistent_connectivity_counts(w, e) == expected);
		}
		BENCHMARK(shape + " rollback_dsu -- time_divide_and_conquer automatic sparse ranks (library)") { return sum_connectivity_counts(engine::library, w); };
		BENCHMARK(shape + " rollback_dsu -- time_divide_and_conquer by frame skipping") { return sum_connectivity_counts(engine::skipping, w); };
		BENCHMARK(shape + " rollback_dsu -- time_divide_and_conquer by dense CSR") {
			return sum_connectivity_counts(engine::dense_csr, w);
		};
		BENCHMARK(shape + " rollback_dsu -- time_divide_and_conquer by ranked frame skipping") {
			return sum_connectivity_counts(engine::rank_compressed_skipping, w);
		};
		BENCHMARK(shape + " persistent_dsu by_copy -- time_divide_and_conquer automatic sparse ranks (library)") {
			return sum_persistent_connectivity_counts(w, engine::library);
		};
		BENCHMARK(shape + " persistent_dsu by_copy -- time_divide_and_conquer by frame skipping") {
			return sum_persistent_connectivity_counts(w, engine::skipping);
		};
		BENCHMARK(shape + " persistent_dsu by_copy -- time_divide_and_conquer by dense CSR") {
			return sum_persistent_connectivity_counts(w, engine::dense_csr);
		};
		BENCHMARK(shape + " persistent_dsu by_copy -- time_divide_and_conquer by ranked frame skipping") {
			return sum_persistent_connectivity_counts(w, engine::rank_compressed_skipping);
		};
	}
}
