#include <atl/ds/dsu.hpp>
#include <atl/ds/rollback_dsu.hpp>
#include <atl/ds/time_divide_and_conquer.hpp>
#include <atl/experimental/graph/deletion_time_connectivity.hpp>
#include <atl/experimental/graph/offline_dynamic_connectivity.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::uint8;
using atl::experimental::graph_event;
using atl::experimental::vertex_pair;

namespace {
	/**
	 * @brief A random log on `n` vertices: links with repeats and self-loops,
	 *        cuts of a random live copy (either orientation), cut-and-relink,
	 *        queries carrying a random pair.
	 */
	std::vector<graph_event> random_log(std::size_t n, std::size_t length, std::mt19937_64 &rng) {
		std::vector<graph_event> events;
		std::vector<vertex_pair> live;
		auto const random_pair = [&] { return vertex_pair{uint32(rng() % n), uint32(rng() % n)}; };
		for (std::size_t i = 0; i < length; ++i) {
			auto const r = rng() % 8;
			if (r < 3 && !live.empty()) {
				std::size_t const j = std::size_t(rng() % live.size());
				auto [u, v] = live[j];
				live[j] = live.back(), live.pop_back();
				if (rng() & 1) std::swap(u, v);
				events.push_back({atl::event_kind::remove, {u, v}});
				if (r == 0) {
					events.push_back({atl::event_kind::add, {v, u}});
					live.push_back({u, v});
				}
			} else if (r < 6) {
				vertex_pair const e = random_pair();
				live.push_back(e);
				events.push_back({atl::event_kind::add, e});
			} else
				events.push_back({atl::event_kind::query, random_pair()});
		}
		return events;
	}

	/** @brief The brute-force oracle: a fresh dsu over the live multiset at every query. */
	std::pair<std::vector<uint32>, std::vector<uint8>> brute(std::size_t n, std::vector<graph_event> const &events) {
		std::vector<uint32> counts;
		std::vector<uint8> answers;
		std::vector<vertex_pair> live;
		for (auto const &[kind, key] : events) {
			if (kind == atl::event_kind::add)
				live.push_back(key);
			else if (kind == atl::event_kind::remove) {
				for (std::size_t j = 0; j < live.size(); ++j)
					if (live[j] == key || live[j] == vertex_pair{key.second, key.first}) {
						live.erase(live.begin() + std::ptrdiff_t(j));
						break;
					}
			} else {
				atl::dsu d(n);
				for (auto [u, v] : live) d.unite(u, v);
				counts.push_back(uint32(d.count()));
				answers.push_back(d.same(key.first, key.second));
			}
		}
		return {counts, answers};
	}

	/** @brief The offline answer to "connected?" queries: time_divide_and_conquer over a rollback_dsu. */
	std::vector<uint8> offline_connected(std::size_t n, std::vector<graph_event> events) {
		for (auto &[kind, key] : events)
			if (kind != atl::event_kind::query && key.first > key.second) std::swap(key.first, key.second);
		auto const lifetimes = atl::lifetimes_of(events);
		std::vector<uint8> answers;
		atl::rollback_dsu d(n);
		atl::time_divide_and_conquer(
			d, events.size(), lifetimes,
			[&](atl::rollback_dsu &s, uint32 item) {
				auto const [u, v] = events[lifetimes[item].begin].key;
				s.unite(u, v);
			},
			[&](atl::rollback_dsu &s, uint32 t) { answers.push_back(s.same(events[t].key.first, events[t].key.second)); },
			[&](uint32 t) { return events[t].kind == atl::event_kind::query; });
		return answers;
	}
}

TEST_CASE("deletion-time connectivity example: small random multigraph logs vs brute force and offline",
		  "[deletion_time_connectivity][example]") {
	std::mt19937_64 rng(100'551);
	for (int iter = 0; iter < 400; ++iter) {
		std::size_t const n = 1 + std::size_t(rng() % 10);
		auto const events = random_log(n, std::size_t(rng() % 120), rng);
		auto const [counts, answers] = brute(n, events);
		CHECK(atl::experimental::component_counts_by_deletion_time(n, events) == counts);
		CHECK(atl::experimental::component_counts(n, events) == counts);
		CHECK(atl::experimental::connected_by_deletion_time(n, events) == answers);
		CHECK(offline_connected(n, events) == answers);
	}
}

TEST_CASE("deletion-time connectivity example: larger logs vs the offline time_divide_and_conquer example",
		  "[deletion_time_connectivity][example]") {
	std::mt19937_64 rng(813);
	for (int iter = 0; iter < 30; ++iter) {
		std::size_t const n = 1 + std::size_t(rng() % 300);
		auto const events = random_log(n, 1 + std::size_t(rng() % 5000), rng);
		REQUIRE(atl::experimental::component_counts_by_deletion_time(n, events) == atl::experimental::component_counts(n, events));
		REQUIRE(atl::experimental::connected_by_deletion_time(n, events) == offline_connected(n, events));
	}
}

TEST_CASE("deletion-time connectivity example: fixed edge log", "[deletion_time_connectivity][example]") {
	using enum atl::event_kind;
	std::vector<graph_event> const events{{query, {}}, {add, {0, 1}}, {query, {}}, {add, {1, 2}}, {query, {}}, {add, {2, 3}}, {query, {}}, {add, {3, 4}}, {query, {}}, {remove, {2, 1}}, {query, {}}};
	CHECK(atl::experimental::component_counts_by_deletion_time(5, events) == std::vector<uint32>{5, 4, 3, 2, 1, 2});
}

TEST_CASE("deletion-time connectivity example: the structure directly", "[deletion_time_connectivity][example]") {
	atl::experimental::deletion_time_connectivity s(4);
	uint32 const a = s.insert(0, 1, 10);
	uint32 const b = s.insert(1, 2, 3);
	uint32 const c = s.insert(0, 2, 7); // closes 0-1-2, outlives b (3): replaces it
	uint32 const d = s.insert(2, 2, 9); // self-loop
	uint32 const e = s.insert(1, 0, 5); // parallel to a, dies first: stays out
	REQUIRE(s.component_count() == 2);
	s.erase(b); // was replaced: nothing changes
	REQUIRE(s.connected(1, 2));
	s.erase(e);
	s.erase(c);
	REQUIRE(!s.connected(1, 2));
	REQUIRE(s.connected(0, 1));
	s.erase(d);
	REQUIRE(s.component_count() == 3);
	s.erase(a);
	REQUIRE(s.component_count() == 4);
}

TEST_CASE("deletion-time connectivity equal keys and empty logs", "[graph][connectivity]") {
	using atl::experimental::deletion_time_connectivity;
	deletion_time_connectivity empty(0);
	CHECK(empty.component_count() == 0);
	CHECK(atl::experimental::component_counts_by_deletion_time(0, {}).empty());
	CHECK(atl::experimental::connected_by_deletion_time(0, {}).empty());
	deletion_time_connectivity graph(3);
	auto const a = graph.insert(0, 1, 8);
	auto const b = graph.insert(1, 2, 8);
	auto const c = graph.insert(2, 0, 8);
	graph.erase(a);
	CHECK(graph.component_count() == 1);
	CHECK(graph.connected(0, 1));
	graph.erase(b);
	CHECK(graph.component_count() == 2);
	graph.erase(c);
	CHECK(graph.component_count() == 3);
#ifdef ATL_ASSERT
	REQUIRE(atl_test::aborts([] {
		deletion_time_connectivity invalid(1);
		invalid.insert(0, 1, 4);
	}));
	REQUIRE(atl_test::aborts([] {
		deletion_time_connectivity invalid(1);
		invalid.insert(0, 0, atl::graph::none);
	}));
	REQUIRE(atl_test::aborts([] {
		deletion_time_connectivity invalid(0);
		invalid.erase(0);
	}));
#endif
}

TEST_CASE("deletion-time connectivity deep path uses heap storage", "[graph][connectivity]") {
	constexpr uint32 vertices = 100'000;
	atl::experimental::deletion_time_connectivity graph(vertices);
	for (uint32 v = 1; v < vertices; ++v) graph.insert(v - 1, v, v);
	CHECK(graph.connected(0, vertices - 1));
	CHECK(graph.component_count() == 1);
	for (uint32 handle = 0; handle + 1 < vertices; ++handle) graph.erase(handle);
	CHECK(graph.component_count() == vertices);
}
