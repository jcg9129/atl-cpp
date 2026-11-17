/**
 * @file
 * @brief Subtree add/sum, vertex add/subtree sum, static subtree sums, bulk build, repeated links and connectivity churn. The contender stores each tour in a splay tree instead of a parent-linked treap; narrower forest classes provide references.
 *
 * Each timed loop measures whole-workload latency, including construction;
 * operations depend on earlier mutations. Untimed runs compare answer checksums.
 * Xeon Platinum 8168, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-25,
 * -O2 / -O2 -march=native. Connectivity churn: library 180 / 173 ms, by splay 208 / 208 ms.
 * Full tables: docs/euler-tour-tree.md.
 */
#include <atl/ds/euler_tour_tree.hpp>
#include <atl/ds/link_cut_tree.hpp>
#include <atl/ds/subtree_link_cut_tree.hpp>
#include <atl/experimental/ds/euler_tour_tree_variants.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

using atl::uint32;
using atl::uint64;
using ll = long long;

namespace {

	/** @brief Compare untimed answer checksums for the same workload across contenders. */
	template<typename Run>
		requires std::invocable<Run &> && std::convertible_to<std::invoke_result_t<Run &>, long long>
	void check_answers(std::map<std::string, long long> &answers, std::string const &row, Run &&run) {
		auto const actual = std::invoke(run);
		if constexpr (std::same_as<decltype(actual), bool const>) {
			REQUIRE(actual);
		} else {
			auto const key = row.substr(row.find(" -- ") + 4);
			if (key.starts_with("N - 1 links") || key.starts_with("assign_parents (random tree)")) {
				REQUIRE(static_cast<long long>(actual) == (row.ends_with("2e5") ? 200'000LL : 500'000LL));
			}
			auto const [it, is_new] = answers.emplace(key, static_cast<long long>(actual));
			static_cast<void>(is_new);
			REQUIRE(static_cast<long long>(actual) == it->second);
		}
	}

	constexpr std::size_t n = 200'000;
	constexpr ll mod = 998'244'353;
	constexpr uint32 none = atl::graph::none;

	std::string label(char const *kind, char const *shape) { return std::string(kind) + " -- " + shape + ", N=Q=2e5"; }

	struct sum_count {
		ll sum = 0, count = 0;
	};

	struct sum_op {
		constexpr sum_count operator()(sum_count const &x, sum_count const &y) const {
			ll const s = x.sum + y.sum;
			return {s >= mod ? s - mod : s, x.count + y.count};
		}
	};

	struct add_mapping {
		constexpr sum_count operator()(ll c, sum_count const &x) const { return {(x.sum + c * x.count) % mod, x.count}; }
	};

	struct mod_add {
		constexpr ll operator()(ll a, ll b) const {
			ll const s = a + b;
			return s >= mod ? s - mod : s;
		}
	};

	/** @brief The inverse of @ref mod_add (the subtree link-cut tree needs a group). */
	struct mod_negate {
		constexpr ll operator()(ll a) const { return a == 0 ? 0 : mod - a; }
	};

	/** @brief The library treap-store euler_tour_tree, built by deduction. */
	struct library_deduction {
		static constexpr char const *name = "library";

		template<typename... Args>
			requires requires(Args &&...forwarded) { atl::euler_tour_tree(std::forward<Args>(forwarded)...); }
		static auto make(Args &&...args) {
			return atl::euler_tour_tree(std::forward<Args>(args)...);
		}
	};

	/** @brief The splay-engine loser, built by deduction. */
	struct euler_tour_tree_by_splay {
		static constexpr char const *name = "by splay";

		template<typename... Args>
			requires requires(Args &&...forwarded) { atl::experimental::euler_tour_tree_by_splay(std::forward<Args>(forwarded)...); }
		static auto make(Args &&...args) {
			return atl::experimental::euler_tour_tree_by_splay(std::forward<Args>(args)...);
		}
	};

	/** @brief A raced euler tour tree class: a row name and a deducing factory. */
	template<typename Kind>
	concept ett_kind = std::convertible_to<decltype(Kind::name), char const *> && requires { Kind::make(std::size_t{1}); };

	/** @brief Run @p fn once per raced class, the library one first. */
	template<typename Fn>
		requires std::invocable<Fn &, library_deduction> && std::invocable<Fn &, euler_tour_tree_by_splay>
	void for_each_ett(Fn &&fn) {
		fn(library_deduction{});
		fn(euler_tour_tree_by_splay{});
	}

	/** @brief The row label of a structure-only (`<>`) instantiation. */
	std::string structure_label(char const *name, char const *shape) { return label((std::string(name) + "<>").c_str(), shape); }

	std::vector<uint32> random_parents(uint64 seed) {
		std::mt19937_64 rng(seed);
		std::vector<uint32> parents(n, none);
		for (std::size_t v = 1; v < n; ++v) parents[v] = uint32(rng() % v);
		return parents;
	}

	/** @brief One query: kind 0 = cut (a, b) then link (c, d); 1 = update at (a, b); 2 = query at (a, b). */
	struct query {
		uint32 kind, a, b, c, d;
		ll x;
	};

	/**
	 * @brief The dynamic subtree shapes: a random initial tree, then thirds of
	 * edge replacement (cut a random edge, reconnect a random vertex to the
	 * other side), an update and a query on a random oriented edge (on
	 * @p is_vertex_update, the update is a point add at a random vertex).
	 * Generated against a structure-only euler_tour_tree, outside the timing.
	 */
	std::vector<query> dynamic_queries(std::vector<uint32> const &parents, bool is_vertex_update, uint64 seed) {
		std::mt19937_64 rng(seed);
		atl::euler_tour_tree sim(n);
		std::vector<std::pair<uint32, uint32>> edges;
		for (uint32 v = 1; v < n; ++v) {
			sim.link(v, parents[v]);
			edges.emplace_back(v, parents[v]);
		}
		std::vector<query> qs(n);
		for (query &q : qs) {
			q.kind = uint32(rng() % 3);
			q.x = ll(rng() % 1'000'000);
			std::size_t const i = rng() % edges.size();
			auto [u, v] = edges[i];
			if (rng() % 2) std::swap(u, v);
			if (q.kind == 0) {
				sim.cut(u, v);
				uint32 const r = uint32(rng() % n);
				uint32 const other = sim.connected(r, u) ? v : u;
				sim.link(r, other);
				edges[i] = {r, other};
				q = {0, u, v, r, other, 0};
			} else if (q.kind == 1 && is_vertex_update) {
				q.a = uint32(rng() % n);
			} else {
				q.a = u;
				q.b = v;
			}
		}
		return qs;
	}
}

TEST_CASE("euler_tour_tree benchmarks: dynamic subtree add + subtree sum (batch shape)", "[euler_tour_tree]") {
	std::map<std::string, long long> answers;
	auto const parents = random_parents(41);
	auto const qs = dynamic_queries(parents, false, 42);
	std::mt19937_64 rng(43);
	std::vector<sum_count> values(n);
	for (sum_count &v : values) v = {ll(rng() % 1'000'000), 1};
	for_each_ett([&]<ett_kind Kind>(Kind) {
		{
			auto const row = label(Kind::name, "subtree add + subtree sum");
			auto run_workload = [&] {
				auto t = Kind::make(values, sum_count{}, 0LL, sum_op{}, add_mapping{}, std::plus<>{});
				t.assign_parents(parents);
				ll acc = 0;
				for (query const &q : qs) {
					if (q.kind == 0) {
						t.cut(q.a, q.b);
						t.link(q.c, q.d);
					} else if (q.kind == 1) {
						t.subtree_apply(q.a, q.b, q.x);
					} else {
						acc += t.subtree_fold(q.a, q.b).sum;
					}
				}
				return acc;
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
	});
	// The splay-tour contender checks the same lazy subtree workload.
}

TEST_CASE("euler_tour_tree benchmarks: dynamic vertex add + subtree sum (batch shape)", "[euler_tour_tree]") {
	std::map<std::string, long long> answers;
	auto const parents = random_parents(51);
	auto const qs = dynamic_queries(parents, true, 52);
	std::mt19937_64 rng(53);
	std::vector<ll> values(n);
	for (ll &v : values) v = ll(rng() % 1'000'000);
	for_each_ett([&]<ett_kind Kind>(Kind) {
		{
			auto const row = label(Kind::name, "vertex add + subtree sum");
			auto run_workload = [&] {
				auto t = Kind::make(values, 0LL, mod_add{});
				t.assign_parents(parents);
				ll acc = 0;
				for (query const &q : qs) {
					if (q.kind == 0) {
						t.cut(q.a, q.b);
						t.link(q.c, q.d);
					} else if (q.kind == 1) {
						t.update(q.a, [x = q.x](ll a) { return mod_add{}(a, x); });
					} else {
						acc += t.subtree_fold(q.a, q.b);
					}
				}
				return acc;
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
	});
	{
		auto const row = label("subtree_link_cut_tree", "vertex add + subtree sum");
		auto run_workload = [&] {
			atl::subtree_link_cut_tree t(values, 0LL, mod_add{}, mod_negate{});
			t.assign_parents(parents);
			ll acc = 0;
			for (query const &q : qs) {
				if (q.kind == 0) {
					t.cut(q.a, q.b);
					t.link(q.c, q.d);
				} else if (q.kind == 1) {
					t.update(q.a, [x = q.x](ll a) { return mod_add{}(a, x); });
				} else {
					acc += t.subtree_fold(q.a, q.b);
				}
			}
			return acc;
		};
		check_answers(answers, row, run_workload);
		BENCHMARK(std::string(row)) { return run_workload(); };
	}
}

TEST_CASE("euler_tour_tree benchmarks: static vertex add + subtree sum", "[euler_tour_tree]") {
	std::map<std::string, long long> answers;
	auto const parents = random_parents(61);
	std::mt19937_64 rng(62);
	std::vector<ll> values(n);
	for (ll &v : values) v = ll(rng() % 1'000'000);
	std::vector<std::pair<uint32, ll>> ops(n);
	for (auto &[v, x] : ops) {
		v = uint32(rng() % n);
		x = rng() % 2 ? ll(rng() % 1'000'000) : -1; // -1: a subtree query
	}
	for_each_ett([&]<ett_kind Kind>(Kind) {
		{
			auto const row = label(Kind::name, "static vertex add + subtree sum");
			auto run_workload = [&] {
				auto t = Kind::make(values, 0LL, mod_add{});
				t.assign_parents(parents);
				ll acc = 0;
				for (auto const &[v, x] : ops) {
					if (x >= 0)
						t.update(v, [x](ll a) { return mod_add{}(a, x); });
					else
						acc += v == 0 ? t.component_fold(0) : t.subtree_fold(v, parents[v]);
				}
				return acc;
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
	});
}

TEST_CASE("euler_tour_tree benchmarks: connectivity churn vs link_cut_tree<>", "[euler_tour_tree]") {
	std::map<std::string, long long> answers;
	auto const parents = random_parents(71);
	// thirds of cut / link / connected, made valid against a simulation outside the timing
	std::mt19937_64 rng(72);
	atl::euler_tour_tree sim(n);
	std::vector<std::pair<uint32, uint32>> edges;
	for (uint32 v = 1; v < n; ++v) {
		sim.link(v, parents[v]);
		edges.emplace_back(v, parents[v]);
	}
	std::vector<query> qs(n);
	for (query &q : qs) {
		uint32 const u = uint32(rng() % n), v = uint32(rng() % n);
		uint32 const kind = uint32(rng() % 3);
		if (kind == 0 && !edges.empty()) {
			std::size_t const i = rng() % edges.size();
			q = {0, edges[i].first, edges[i].second, 0, 0, 0};
			sim.cut(edges[i].first, edges[i].second);
			edges[i] = edges.back();
			edges.pop_back();
		} else if (kind == 1 && !sim.connected(u, v)) {
			q = {1, u, v, 0, 0, 0};
			sim.link(u, v);
			edges.emplace_back(u, v);
		} else {
			q = {2, u, v, 0, 0, 0};
		}
	}
	auto run = [&]<typename Tree>(Tree &t)
		requires requires(Tree &forest) { forest.assign_parents(parents); forest.connected(uint32{}, uint32{}); }
	{
		ll acc = 0;
		for (query const &q : qs) {
			if (q.kind == 0)
				t.cut(q.a, q.b);
			else if (q.kind == 1)
				t.link(q.a, q.b);
			else
				acc += t.connected(q.a, q.b);
		}
		return acc;
	};
	for_each_ett([&]<ett_kind Kind>(Kind) {
		{
			auto const row = structure_label(Kind::name, "assign_parents (random tree)");
			auto run_workload = [&] {
				auto t = Kind::make(n);
				t.assign_parents(parents);
				return t.component_size(0);
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
	});
	// link_cut_tree's link of a fresh singleton below an existing vertex is O(1) (evert of a
	// lone node, one parent pointer): the link-build rows are not a like-for-like race
	for_each_ett([&]<ett_kind Kind>(Kind) {
		{
			auto const row = structure_label(Kind::name, "N - 1 links (random tree)");
			auto run_workload = [&] {
				auto t = Kind::make(n);
				for (uint32 v = 1; v < n; ++v) t.link(v, parents[v]);
				return t.component_size(0);
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
	});
	{
		auto const row = label("link_cut_tree<>", "N - 1 links (random tree)");
		auto run_workload = [&] {
			atl::link_cut_tree t(n);
			for (uint32 v = 1; v < n; ++v) t.link(v, parents[v]);
			return t.connected(0, uint32(n - 1));
		};
		check_answers(answers, row, run_workload);
		BENCHMARK(std::string(row)) { return run_workload(); };
	}
	for_each_ett([&]<ett_kind Kind>(Kind) {
		{
			auto const row = structure_label(Kind::name, "assign_parents + churn (cut / link / connected thirds)");
			auto run_workload = [&] {
				auto t = Kind::make(n);
				t.assign_parents(parents);
				return run(t);
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
	});
	{
		auto const row = label("link_cut_tree<>", "assign_parents + churn (cut / link / connected thirds)");
		auto run_workload = [&] {
			atl::link_cut_tree t(n);
			t.assign_parents(parents);
			return run(t);
		};
		check_answers(answers, row, run_workload);
		BENCHMARK(std::string(row)) { return run_workload(); };
	}
}
