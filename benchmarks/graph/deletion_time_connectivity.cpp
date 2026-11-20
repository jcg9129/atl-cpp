/**
 * @file
 * @brief Whole-log connectivity throughput: a maximum link-cut forest keyed by
 * deletion time versus time divide and conquer with rollback DSU. Both normalize
 * and pair the raw log inside the timed call. Shapes cover balanced/sparse churn,
 * 90% unremoved edges, link-query-cut lifetimes, and dense 1000-vertex churn.
 * Intel Xeon Platinum 8168, GCC 15.2, RelWithDebInfo, asserts off, 2026-10-25,
 * `-O2 -g -DNDEBUG -std=c++23` and additionally `-march=native`: balanced churn
 * takes 40.1/40.0 ms by deletion time versus 32.0/32.0 ms by rollback DSU;
 * dense churn takes 27.7/27.8 versus 29.8/30.1 ms (plain/native).
 * Full table: docs/time-divide-and-conquer.md.
 */
#include <atl/ds/time_divide_and_conquer.hpp>
#include <atl/experimental/graph/deletion_time_connectivity.hpp>
#include <atl/experimental/graph/offline_dynamic_connectivity.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

using atl::uint32;
using atl::uint64;
using atl::experimental::graph_event;
using atl::experimental::vertex_pair;

namespace {
	vertex_pair random_edge(std::size_t n, std::mt19937_64 &rng) { return {uint32(rng() % n), uint32(rng() % n)}; }

	/**
	 * @brief `q` events on `n` vertices: a query with probability `query_share`,
	 *        otherwise a link or, half the time, a cut of a random live edge; an
	 *        edge is mortal (may be cut) with probability `mortal_share`.
	 */
	std::vector<graph_event> churn(std::size_t n, std::size_t q, double query_share, double mortal_share, std::mt19937_64 &rng) {
		std::vector<graph_event> events;
		std::vector<vertex_pair> mortal;
		std::uniform_real_distribution<double> coin(0, 1);
		while (events.size() < q) {
			if (coin(rng) < query_share)
				events.push_back({atl::event_kind::query, {}});
			else if (!mortal.empty() && rng() % 2) {
				std::size_t const j = std::size_t(rng() % mortal.size());
				events.push_back({atl::event_kind::remove, mortal[j]});
				mortal[j] = mortal.back(), mortal.pop_back();
			} else {
				auto const e = random_edge(n, rng);
				if (coin(rng) < mortal_share) mortal.push_back(e);
				events.push_back({atl::event_kind::add, e});
			}
		}
		return events;
	}

	/** @brief Every link is cut right after one query: link, query, cut. */
	std::vector<graph_event> short_lived(std::size_t n, std::size_t q, std::mt19937_64 &rng) {
		std::vector<graph_event> events;
		while (events.size() + 3 <= q) {
			auto const e = random_edge(n, rng);
			events.push_back({atl::event_kind::add, e});
			events.push_back({atl::event_kind::query, {}});
			events.push_back({atl::event_kind::remove, e});
		}
		return events;
	}

	uint64 checksum(std::vector<uint32> const &counts) {
		uint64 acc = 0;
		for (std::size_t i = 0; i < counts.size(); ++i) acc += uint64(counts[i]) * (i | 1);
		return acc;
	}

	/** @brief Adapt a log solver's count vector to a scalar throughput checksum. */
	template<typename Solve>
		requires std::is_invocable_r_v<std::vector<uint32>, Solve &, std::size_t, std::span<graph_event const>>
	uint64 measure_counts(Solve solve, std::size_t n, std::vector<graph_event> const &events) {
		return checksum(std::invoke(solve, n, std::span<graph_event const>(events)));
	}

	void race(std::string const &shape, std::size_t n, std::vector<graph_event> const &events) {
		REQUIRE(atl::experimental::component_counts_by_deletion_time(n, events) == atl::experimental::component_counts(n, events));
		BENCHMARK(shape + " -- component_counts by deletion time") {
			return measure_counts([](std::size_t vertices, std::span<graph_event const> log) { return atl::experimental::component_counts_by_deletion_time(vertices, log); }, n, events);
		};
		BENCHMARK(shape + " -- component_counts by rollback DSU") {
			return measure_counts([](std::size_t vertices, std::span<graph_event const> log) { return atl::experimental::component_counts(vertices, log); }, n, events);
		};
	}
}

TEST_CASE("deletion-time connectivity vs time_divide_and_conquer", "[deletion_time_connectivity][race]") {
	std::mt19937_64 rng(2026'10'24);
	constexpr std::size_t n = 300'000, q = 300'000;
	race("churn n=q=3e5, 50% queries", n, churn(n, q, 0.5, 1.0, rng));
	race("churn n=q=3e5, 1% queries", n, churn(n, q, 0.01, 1.0, rng));
	race("long-lived n=q=3e5 (90% never cut), 50% queries", n, churn(n, q, 0.5, 0.1, rng));
	race("short-lived n=q=3e5 (link, query, cut)", n, short_lived(n, q, rng));
	race("dense churn n=1e3, q=3e5, 50% queries", 1'000, churn(1'000, q, 0.5, 1.0, rng));
}
