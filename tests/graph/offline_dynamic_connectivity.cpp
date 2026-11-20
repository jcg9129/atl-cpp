#include <atl/ds/dsu.hpp>
#include <atl/ds/persistent_dsu.hpp>
#include <atl/ds/rollback_dsu.hpp>
#include <atl/ds/time_divide_and_conquer.hpp>
#include <atl/experimental/ds/time_divide_and_conquer_variants.hpp>
#include <atl/experimental/graph/offline_dynamic_connectivity.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::experimental::graph_event;
using atl::experimental::vertex_pair;

namespace {
	/**
	 * @brief A random log on `n` vertices: links with repeats and self-loops, cuts
	 *        of a random live copy (either orientation), cut-and-relink, queries.
	 */
	std::vector<graph_event> random_log(std::size_t n, std::size_t length, std::mt19937_64 &rng) {
		std::vector<graph_event> events;
		std::vector<vertex_pair> live;
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
				vertex_pair const e{uint32(rng() % n), uint32(rng() % n)};
				live.push_back(e);
				events.push_back({atl::event_kind::add, e});
			} else
				events.push_back({atl::event_kind::query, {}});
		}
		return events;
	}

	/** @brief The oracle: a fresh dsu over the live multiset at every query. */
	std::vector<uint32> brute_counts(std::size_t n, std::vector<graph_event> const &events) {
		std::vector<uint32> counts;
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
			}
		}
		return counts;
	}

	template<typename Engine, std::copyable Structure, typename Unite>
		requires std::invocable<Unite &, Structure &, uint32, uint32>
			  && requires(Structure const &s) { { s.count() } -> std::convertible_to<std::size_t>; }
			  && std::invocable<Engine &, Structure &, std::size_t, std::vector<atl::item_lifetime> const &,
								std::function<void(Structure &, uint32)>, std::function<void(Structure &, uint32)>, std::function<bool(uint32)>>
	std::vector<uint32> counts_with(Engine engine, Structure structure, std::vector<graph_event> events, Unite unite) {
		for (auto &[kind, key] : events)
			if (key.first > key.second) std::swap(key.first, key.second);
		auto const lifetimes = atl::lifetimes_of(events);
		std::vector<uint32> counts;
		engine(
			structure, events.size(), lifetimes,
			[&](Structure &s, uint32 i) {
				auto const [u, v] = events[lifetimes[i].begin].key;
				unite(s, u, v);
			},
			[&](Structure &s, uint32) { counts.push_back(uint32(s.count())); },
			[&](uint32 t) { return events[t].kind == atl::event_kind::query; });
		return counts;
	}
}

TEST_CASE("offline dynamic connectivity example: random multigraph logs", "[time_divide_and_conquer][example]") {
	std::mt19937_64 rng(100551);
	auto const unite_in_place = [](atl::rollback_dsu &d, uint32 u, uint32 v) { d.unite(u, v); };
	auto const unite_persistent = [](atl::persistent_dsu<> &d, uint32 u, uint32 v) { d = std::move(d).unite(u, v); };
	auto const library = []<typename... Args>(Args &&...args)
		requires requires { atl::time_divide_and_conquer(std::forward<Args>(args)...); }
	{ atl::time_divide_and_conquer(std::forward<Args>(args)...); };
	auto const recursive = []<typename... Args>(Args &&...args)
		requires requires { atl::time_divide_and_conquer(std::forward<Args>(args)...); }
	{
		atl::experimental::time_divide_and_conquer_by_recursive_vectors(std::forward<Args>(args)...);
	};
	auto const unpruned = []<typename... Args>(Args &&...args)
		requires requires { atl::time_divide_and_conquer(std::forward<Args>(args)...); }
	{
		atl::experimental::time_divide_and_conquer_by_unpruned_csr(std::forward<Args>(args)...);
	};
	for (int iter = 0; iter < 300; ++iter) {
		std::size_t const n = 1 + std::size_t(rng() % 10);
		auto const events = random_log(n, std::size_t(rng() % 80), rng);
		auto const expected = brute_counts(n, events);
		CHECK(atl::experimental::component_counts(n, events) == expected);
		CHECK(counts_with(recursive, atl::rollback_dsu(n), events, unite_in_place) == expected);
		CHECK(counts_with(unpruned, atl::rollback_dsu(n), events, unite_in_place) == expected);
		CHECK(counts_with(library, atl::persistent_dsu<>(n), events, unite_persistent) == expected);
	}
}

TEST_CASE("offline dynamic connectivity example: fixed edge log", "[time_divide_and_conquer][example]") {
	using enum atl::event_kind;
	std::vector<graph_event> const events{{query, {}}, {add, {0, 1}}, {query, {}}, {add, {1, 2}}, {query, {}}, {add, {2, 3}}, {query, {}}, {add, {3, 4}}, {query, {}}, {remove, {2, 1}}, {query, {}}};
	CHECK(atl::experimental::component_counts(5, events) == std::vector<uint32>{5, 4, 3, 2, 1, 2});
}

TEST_CASE("offline connectivity empty and unqueried logs", "[graph][connectivity]") {
	using enum atl::event_kind;
	CHECK(atl::experimental::component_counts(0, {}).empty());
	std::vector<graph_event> const queries{{query, {atl::graph::none, atl::graph::none}}};
	CHECK(atl::experimental::component_counts(0, queries) == std::vector<uint32>{0});
	std::vector<graph_event> const edges{{add, {0, 1}}, {remove, {1, 0}}};
	CHECK(atl::experimental::component_counts(2, edges).empty());
#ifdef ATL_ASSERT
	REQUIRE(atl_test::aborts([] {
		std::vector<graph_event> const invalid{{atl::event_kind::add, {0, 2}}};
		static_cast<void>(atl::experimental::component_counts(2, invalid));
	}));
	REQUIRE(atl_test::aborts([] {
		std::vector<graph_event> const invalid{{atl::event_kind::remove, {0, 1}}};
		static_cast<void>(atl::experimental::component_counts(2, invalid));
	}));
#endif
}
