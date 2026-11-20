/**
 * @file
 * @brief Race between `atl::time_divide_and_conquer` (the library, automatic sparse ranks)
 * and engines by recursive vectors, unpruned CSR, configurable pruning/stretching,
 * dense CSR, frame skipping, or ranked frame skipping. Restore modes use
 * rollback_dsu undo or persistent_dsu copies. Loops measure whole-log throughput,
 * including setup, on bipartite toggles and churn with dense/sparse queries and long/short lifetimes;
 * lifetimes_of-only rows isolate event pairing. Both fixed-seed timed bipartite
 * toggle shapes stay forests. Untimed odd-cycle validation checks every engine
 * against fresh graph coloring at every state and at sparse selected times.
 * Result (Xeon Platinum 8280, GCC 15.2, `-O2` and `-O2 -march=native`,
 * RelWithDebInfo, asserts off, 2026-11-01): at one percent queries dense CSR takes 34.6/32.9 ms by rollback
 * versus 13.8/13.9 ms by forced rank compression. The integrated automatic
 * library dispatch is not measured yet. Numbers: docs/time-divide-and-conquer.md.
 */
#include <atl/ds/flat_hash_set.hpp>
#include <atl/ds/persistent_dsu.hpp>
#include <atl/ds/rollback_dsu.hpp>
#include <atl/ds/rollback_weighted_dsu.hpp>
#include <atl/ds/time_divide_and_conquer.hpp>
#include <atl/experimental/ds/time_divide_and_conquer_variants.hpp>
#include <atl/experimental/graph/offline_bipartiteness.hpp>
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
				std::size_t const j = std::size_t(rng() % mortal.size());
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

	namespace engine {
		/** @brief Store items in one vector per time-tree node and traverse recursively. */
		constexpr auto time_divide_and_conquer_by_recursive_vectors = []<typename... Args>(Args &&...args)
			requires requires { atl::experimental::time_divide_and_conquer_by_recursive_vectors(std::forward<Args>(args)...); }
		{ atl::experimental::time_divide_and_conquer_by_recursive_vectors(std::forward<Args>(args)...); };
		/** @brief Walk flat CSR without pruning subtrees lacking selected times. */
		constexpr auto time_divide_and_conquer_by_unpruned_csr = []<typename... Args>(Args &&...args)
			requires requires { atl::experimental::time_divide_and_conquer_by_unpruned_csr(std::forward<Args>(args)...); }
		{ atl::experimental::time_divide_and_conquer_by_unpruned_csr(std::forward<Args>(args)...); };
		/** @brief Prune unqueried CSR nodes while leaving suffix lifetimes unstretched. */
		constexpr auto time_divide_and_conquer_by_configurable_csr = []<typename... Args>(Args &&...args)
			requires requires { atl::experimental::time_divide_and_conquer_by_configurable_csr<true, false>(std::forward<Args>(args)...); }
		{ atl::experimental::time_divide_and_conquer_by_configurable_csr<true, false>(std::forward<Args>(args)...); };
		/** @brief Delegate to the library density-based time-coordinate dispatch. */
		constexpr auto library = []<typename... Args>(Args &&...args)
			requires requires { atl::time_divide_and_conquer(std::forward<Args>(args)...); }
		{ atl::time_divide_and_conquer(std::forward<Args>(args)...); };
		/** @brief Skip restore frames whose enclosing frame already restores their changes. */
		constexpr auto time_divide_and_conquer_by_frame_skipping = []<typename... Args>(Args &&...args)
			requires requires { atl::experimental::time_divide_and_conquer_by_frame_skipping(std::forward<Args>(args)...); }
		{ atl::experimental::time_divide_and_conquer_by_frame_skipping(std::forward<Args>(args)...); };
		/** @brief Keep original time coordinates with pruning and suffix stretching. */
		constexpr auto time_divide_and_conquer_by_dense_csr = []<typename... Args>(Args &&...args)
			requires requires { atl::experimental::time_divide_and_conquer_by_dense_csr(std::forward<Args>(args)...); }
		{ atl::experimental::time_divide_and_conquer_by_dense_csr(std::forward<Args>(args)...); };
		/** @brief Compress selected times to ranks and skip redundant restore frames. */
		constexpr auto time_divide_and_conquer_by_ranked_frame_skipping = []<typename... Args>(Args &&...args)
			requires requires { atl::experimental::time_divide_and_conquer_by_ranked_frame_skipping(std::forward<Args>(args)...); }
		{ atl::experimental::time_divide_and_conquer_by_ranked_frame_skipping(std::forward<Args>(args)...); };
	}

	template<typename Engine, typename State>
	concept time_engine_for = std::invocable<Engine &, State &, std::size_t, std::vector<atl::item_lifetime> const &,
											 std::function<void(State &, uint32)>, std::function<void(State &, uint32)>, std::function<bool(uint32)>>;

	/** @brief Connectivity with rollback_dsu (default by_rollback): sum of the component counts at the queries. */
	template<time_engine_for<atl::rollback_dsu> Engine>
	uint64 sum_connectivity_counts(Engine e, workload const &w) {
		atl::rollback_dsu d(w.n);
		uint64 acc = 0;
		e(
			d, w.events.size(), w.lifetimes,
			[&](atl::rollback_dsu &s, uint32 i) { s.unite(w.edge[i].first, w.edge[i].second); },
			[&](atl::rollback_dsu &s, uint32 t) { acc += uint64(s.count()) * uint64(t | 1); }, [&](uint32 t) { return w.is_query[t] != 0; });
		return acc;
	}

	/** @brief The same answers from persistent_dsu by copy (default through the library engine). */
	template<time_engine_for<atl::persistent_dsu<>> Engine = decltype(engine::library)>
	uint64 sum_persistent_connectivity_counts(workload const &w, Engine e = engine::library) {
		atl::persistent_dsu<> d(w.n);
		uint64 acc = 0;
		e(
			d, w.events.size(), w.lifetimes,
			[&](atl::persistent_dsu<> &s, uint32 i) { s = std::move(s).unite(w.edge[i].first, w.edge[i].second); },
			[&](atl::persistent_dsu<> &s, uint32 t) { acc += uint64(s.count()) * uint64(t | 1); }, [&](uint32 t) { return w.is_query[t] != 0; });
		return acc;
	}

	/** @brief Every engine on one connectivity shape; answers cross-checked outside timing. */
	void benchmark_connectivity(std::string const &shape, workload const &w, bool has_persistent) {
		uint64 const expected = sum_connectivity_counts(engine::library, w);
		REQUIRE(sum_connectivity_counts(engine::time_divide_and_conquer_by_recursive_vectors, w) == expected);
		REQUIRE(sum_connectivity_counts(engine::time_divide_and_conquer_by_unpruned_csr, w) == expected);
		REQUIRE(sum_connectivity_counts(engine::time_divide_and_conquer_by_configurable_csr, w) == expected);
		REQUIRE(sum_connectivity_counts(engine::time_divide_and_conquer_by_dense_csr, w) == expected);
		if (has_persistent) REQUIRE(sum_persistent_connectivity_counts(w) == expected);
		BENCHMARK(shape + " -- by recursive vectors") { return sum_connectivity_counts(engine::time_divide_and_conquer_by_recursive_vectors, w); };
		BENCHMARK(shape + " -- by unpruned CSR") { return sum_connectivity_counts(engine::time_divide_and_conquer_by_unpruned_csr, w); };
		BENCHMARK(shape + " -- automatic sparse ranks (library)") { return sum_connectivity_counts(engine::library, w); };
		BENCHMARK(shape + " -- by dense CSR") { return sum_connectivity_counts(engine::time_divide_and_conquer_by_dense_csr, w); };
		BENCHMARK(shape + " -- by configurable CSR (pruned, unstretched)") { return sum_connectivity_counts(engine::time_divide_and_conquer_by_configurable_csr, w); };
		if (has_persistent)
			BENCHMARK(shape + " -- library, persistent_dsu by_copy") { return sum_persistent_connectivity_counts(w); };
	}

	/** @brief Weighted rollback DSU with a separately restored contradiction counter. */
	template<time_engine_for<atl::experimental::parity_state> Engine>
	uint64 sum_bipartite_times(Engine apply, workload const &w) {
		using atl::experimental::parity_state;
		parity_state state{decltype(parity_state::sides)(w.n)};
		uint64 acc = 0;
		apply(
			state, w.events.size(), w.lifetimes,
			[&](parity_state &s, uint32 i) {
				if (s.odd_cycles) return;
				if (s.sides.unite(w.edge[i].first, w.edge[i].second, 1) == atl::unite_outcome::contradicts) ++s.odd_cycles;
			},
			[&](parity_state &s, uint32 t) { acc += uint64(s.odd_cycles == 0) * uint64(t | 1); }, atl::every_time{},
			atl::by_rollback{[](parity_state &s) { return std::pair{s.sides.time(), s.odd_cycles}; },
							 [](parity_state &s, std::pair<std::size_t, uint32> token) {
								 s.sides.rollback(token.first);
								 s.odd_cycles = token.second;
							 }});
		return acc;
	}

	/** @brief Half the toggles cut a live edge; the others add a distinct non-loop pair. */
	workload toggles(std::size_t n, std::size_t q, std::mt19937_64 &rng) {
		workload w{n, {}, {}, {}, {}};
		std::vector<vertex_pair> live;
		atl::flat_hash_set<vertex_pair> present;
		while (w.events.size() < q) {
			if (!live.empty() && rng() % 2) {
				std::size_t const j = std::size_t(rng() % live.size());
				present.erase(live[j]);
				w.events.push_back({atl::event_kind::remove, live[j]});
				live[j] = live.back(), live.pop_back();
			} else {
				auto edge = random_edge(n, rng);
				while (edge.first == edge.second || present.contains(edge)) edge = random_edge(n, rng);
				present.insert(edge);
				live.push_back(edge);
				w.events.push_back({atl::event_kind::add, edge});
			}
		}
		finish(w);
		return w;
	}

	/** @brief Recompute each state by graph coloring, independently of time-tree restoration. */
	std::vector<uint8> reference_bipartite_states(workload const &w) {
		std::vector<vertex_pair> live;
		std::vector<uint8> answers;
		for (auto const &change : w.events) {
			if (change.kind == atl::event_kind::add) live.push_back(change.key);
			if (change.kind == atl::event_kind::remove) {
				auto const found = std::ranges::find(live, change.key);
				REQUIRE(found != live.end());
				live.erase(found);
			}
			std::vector<std::vector<uint32>> rows(w.n);
			for (auto const &[u, v] : live) {
				rows[u].push_back(v);
				rows[v].push_back(u);
			}
			std::vector<int> color(w.n, -1);
			bool is_bipartite = true;
			for (uint32 root = 0; root < w.n; ++root) {
				if (color[root] != -1) continue;
				color[root] = 0;
				std::vector<uint32> queue{root};
				for (std::size_t head = 0; head < queue.size(); ++head)
					for (auto adjacent : rows[queue[head]]) {
						if (color[adjacent] == -1) {
							color[adjacent] = color[queue[head]] ^ 1;
							queue.push_back(adjacent);
						} else if (color[adjacent] == color[queue[head]])
							is_bipartite = false;
					}
			}
			answers.push_back(uint8(is_bipartite));
		}
		return answers;
	}

	/** @brief Capture every selected answer, preserving the contender's rollback policy. */
	template<time_engine_for<atl::experimental::parity_state> Engine>
	std::vector<uint8> bipartite_states(Engine apply, workload const &w, uint32 stride) {
		using atl::experimental::parity_state;
		parity_state state{decltype(parity_state::sides)(w.n)};
		std::vector<uint8> answers;
		apply(
			state, w.events.size(), w.lifetimes,
			[&](parity_state &s, uint32 item) {
				if (s.odd_cycles) return;
				if (s.sides.unite(w.edge[item].first, w.edge[item].second, 1) == atl::unite_outcome::contradicts) ++s.odd_cycles;
			},
			[&](parity_state &s, uint32) { answers.push_back(uint8(s.odd_cycles == 0)); },
			[stride](uint32 t) { return t % stride == 0; },
			atl::by_rollback{[](parity_state &s) { return std::pair{s.sides.time(), s.odd_cycles}; },
							 [](parity_state &s, std::pair<std::size_t, uint32> token) {
								 s.sides.rollback(token.first);
								 s.odd_cycles = token.second;
							 }});
		return answers;
	}

	/** @brief Untimed triangles repeatedly close and reopen, with overlapping contradictions. */
	void validate_bipartite_engines() {
		workload w{6, {}, {}, {}, {}};
		auto change = [&](atl::event_kind kind, uint32 u, uint32 v) { w.events.push_back({kind, {u, v}}); };
		for (int repeat = 0; repeat < 32; ++repeat) {
			change(atl::event_kind::add, 0, 1);
			change(atl::event_kind::add, 1, 2);
			change(atl::event_kind::add, 0, 2);
			change(atl::event_kind::add, 3, 4);
			change(atl::event_kind::add, 4, 5);
			change(atl::event_kind::add, 3, 5);
			change(atl::event_kind::remove, 0, 2);
			change(atl::event_kind::remove, 3, 5);
			change(atl::event_kind::add, 0, 2);
			change(atl::event_kind::remove, 0, 1);
			change(atl::event_kind::remove, 1, 2);
			change(atl::event_kind::remove, 0, 2);
			change(atl::event_kind::remove, 3, 4);
			change(atl::event_kind::remove, 4, 5);
		}
		finish(w);
		auto const reference = reference_bipartite_states(w);
		REQUIRE(std::ranges::find(reference, uint8(0)) != reference.end());
		REQUIRE(std::ranges::find(reference, uint8(1)) != reference.end());
		for (uint32 stride : {1U, 13U}) {
			std::vector<uint8> expected;
			for (std::size_t t = 0; t < reference.size(); t += stride) expected.push_back(reference[t]);
			REQUIRE(std::ranges::find(expected, uint8(0)) != expected.end());
			REQUIRE(std::ranges::find(expected, uint8(1)) != expected.end());
			REQUIRE(bipartite_states(engine::library, w, stride) == expected);
			REQUIRE(bipartite_states(engine::time_divide_and_conquer_by_recursive_vectors, w, stride) == expected);
			REQUIRE(bipartite_states(engine::time_divide_and_conquer_by_unpruned_csr, w, stride) == expected);
			REQUIRE(bipartite_states(engine::time_divide_and_conquer_by_configurable_csr, w, stride) == expected);
			REQUIRE(bipartite_states(engine::time_divide_and_conquer_by_frame_skipping, w, stride) == expected);
			REQUIRE(bipartite_states(engine::time_divide_and_conquer_by_dense_csr, w, stride) == expected);
			REQUIRE(bipartite_states(engine::time_divide_and_conquer_by_ranked_frame_skipping, w, stride) == expected);
		}
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
 * @brief Sparse-query comparison: dense CSR and skipping contenders
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
		auto const check = [&]<typename Engine>(Engine e)
			requires time_engine_for<Engine, atl::rollback_dsu> && time_engine_for<Engine, atl::persistent_dsu<>>
		{
			REQUIRE(sum_connectivity_counts(e, w) == expected);
			REQUIRE(sum_persistent_connectivity_counts(w, e) == expected);
		};
		check(engine::time_divide_and_conquer_by_frame_skipping);
		check(engine::time_divide_and_conquer_by_dense_csr);
		check(engine::time_divide_and_conquer_by_ranked_frame_skipping);
		BENCHMARK(shape + " rollback_dsu -- time_divide_and_conquer automatic sparse ranks (library)") { return sum_connectivity_counts(engine::library, w); };
		BENCHMARK(shape + " rollback_dsu -- time_divide_and_conquer by frame skipping") { return sum_connectivity_counts(engine::time_divide_and_conquer_by_frame_skipping, w); };
		BENCHMARK(shape + " rollback_dsu -- time_divide_and_conquer by dense CSR") {
			return sum_connectivity_counts(engine::time_divide_and_conquer_by_dense_csr, w);
		};
		BENCHMARK(shape + " rollback_dsu -- time_divide_and_conquer by ranked frame skipping") {
			return sum_connectivity_counts(engine::time_divide_and_conquer_by_ranked_frame_skipping, w);
		};
		BENCHMARK(shape + " persistent_dsu by_copy -- time_divide_and_conquer automatic sparse ranks (library)") {
			return sum_persistent_connectivity_counts(w, engine::library);
		};
		BENCHMARK(shape + " persistent_dsu by_copy -- time_divide_and_conquer by frame skipping") {
			return sum_persistent_connectivity_counts(w, engine::time_divide_and_conquer_by_frame_skipping);
		};
		BENCHMARK(shape + " persistent_dsu by_copy -- time_divide_and_conquer by dense CSR") {
			return sum_persistent_connectivity_counts(w, engine::time_divide_and_conquer_by_dense_csr);
		};
		BENCHMARK(shape + " persistent_dsu by_copy -- time_divide_and_conquer by ranked frame skipping") {
			return sum_persistent_connectivity_counts(w, engine::time_divide_and_conquer_by_ranked_frame_skipping);
		};
	}
}

TEST_CASE("time_divide_and_conquer bipartite engine race", "[time_divide_and_conquer][race]") {
	validate_bipartite_engines();
	std::mt19937_64 rng(2026'10'24);
	auto const w = toggles(100'000, 100'000, rng);
	uint64 const expected = sum_bipartite_times(engine::library, w);
	REQUIRE(sum_bipartite_times(engine::time_divide_and_conquer_by_recursive_vectors, w) == expected);
	REQUIRE(sum_bipartite_times(engine::time_divide_and_conquer_by_unpruned_csr, w) == expected);
	REQUIRE(sum_bipartite_times(engine::time_divide_and_conquer_by_configurable_csr, w) == expected);
	std::string const shape = "bipartite toggles n=q=1e5, every time";
	BENCHMARK(shape + " -- by recursive vectors") { return sum_bipartite_times(engine::time_divide_and_conquer_by_recursive_vectors, w); };
	BENCHMARK(shape + " -- by unpruned CSR") { return sum_bipartite_times(engine::time_divide_and_conquer_by_unpruned_csr, w); };
	BENCHMARK(shape + " -- library") { return sum_bipartite_times(engine::library, w); };
	BENCHMARK(shape + " -- by configurable CSR (pruned, unstretched)") { return sum_bipartite_times(engine::time_divide_and_conquer_by_configurable_csr, w); };
	BENCHMARK(shape + " -- lifetimes_of alone") { return atl::lifetimes_of(w.events).size(); };
}

TEST_CASE("time_divide_and_conquer bipartite sparse comparison", "[time_divide_and_conquer][comparison]") {
	validate_bipartite_engines();
	std::mt19937_64 rng(2026'10'29);
	auto const w = toggles(100'000, 100'000, rng);
	uint64 const expected = sum_bipartite_times(engine::library, w);
	REQUIRE(sum_bipartite_times(engine::time_divide_and_conquer_by_frame_skipping, w) == expected);
	REQUIRE(sum_bipartite_times(engine::time_divide_and_conquer_by_dense_csr, w) == expected);
	REQUIRE(sum_bipartite_times(engine::time_divide_and_conquer_by_ranked_frame_skipping, w) == expected);
	std::string const shape = "bipartite toggles n=q=1e5, every time";
	BENCHMARK(shape + " -- library") { return sum_bipartite_times(engine::library, w); };
	BENCHMARK(shape + " -- by frame skipping") { return sum_bipartite_times(engine::time_divide_and_conquer_by_frame_skipping, w); };
	BENCHMARK(shape + " -- by dense CSR") { return sum_bipartite_times(engine::time_divide_and_conquer_by_dense_csr, w); };
	BENCHMARK(shape + " -- by ranked frame skipping") { return sum_bipartite_times(engine::time_divide_and_conquer_by_ranked_frame_skipping, w); };
}
