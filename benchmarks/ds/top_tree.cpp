/**
 * @file
 * @brief Path and subtree lazy folds, affine composition, mixed path/subtree operations, weighted diameter, marked distances and connectivity on random, path, star, caterpillar and binary trees. Contenders change layout, per-splay push, rake insertion, edge leaves, rerooting, cached vertex clusters or cluster reversal; narrower forest classes provide references.
 *
 * Each timed loop measures whole-workload latency, including construction;
 * operations depend on earlier mutations. Untimed runs compare answer checksums.
 * Xeon Platinum 8168, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-25,
 * -O2 / -O2 -march=native. Path-sum path N=Q=5e5: library 864 / 824 ms, by per-splay push 975 / 975 ms.
 * Full tables: docs/top-tree.md.
 */
#include <atl/ds/euler_tour_tree.hpp>
#include <atl/ds/link_cut_tree.hpp>
#include <atl/ds/subtree_link_cut_tree.hpp>
#include <atl/ds/top_clusters.hpp>
#include <atl/ds/top_tree.hpp>
#include <atl/experimental/ds/top_tree_variants.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <algorithm>
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

	namespace ex = atl::experimental;
	namespace tc = atl::top_clusters;

	constexpr uint32 none = atl::graph::none;
	constexpr ll mod = 998'244'353;
	constexpr std::size_t sizes[] = {200'000, 500'000};

	std::string label(std::string const &kind, char const *shape, std::size_t n) {
		return kind + " -- " + shape + ", N=Q=" + (n == 200'000 ? "2e5" : "5e5");
	}

	/** @brief A top tree class template over a ready-made set: `type<Set, ValuePlacement>`. */
	template<template<typename...> typename Class>
	struct set_kind {
		template<typename Set, typename ValuePlacement = void>
		using type =
			Class<typename Set::value_type, typename Set::path_type, typename Set::point_type, typename Set::compress,
				  typename Set::rake, typename Set::add_vertex, typename Set::add_edge, typename Set::reverse,
				  typename Set::tag_type, typename Set::mapping, typename Set::map_path, typename Set::map_point,
				  typename Set::compose, ValuePlacement>;
	};

	template<typename... Args>
	using library_class = atl::top_tree<Args...>;

	/** @brief Adapt the library class template to the common contender loop. */
	struct library : set_kind<library_class> {
		static constexpr char const *name = "library";
	};

	/** @brief Keep each node in one interleaved record. */
	struct top_tree_by_interleaved_layout : set_kind<ex::top_tree_by_interleaved_layout> {
		static constexpr char const *name = "by interleaved layout";
	};

	/** @brief Repeat ancestor propagation inside each splay tree. */
	struct top_tree_by_per_splay_push : set_kind<ex::top_tree_by_per_splay_push> {
		static constexpr char const *name = "by per-splay push";
	};

	/** @brief Join light children at the end of the rake tree. */
	struct top_tree_by_rake_append : set_kind<ex::top_tree_by_rake_append> {
		static constexpr char const *name = "by rake append";
	};

	/** @brief Represent edges as compress-tree leaves. */
	struct top_tree_by_edge_leaves : set_kind<ex::top_tree_by_edge_leaves> {
		static constexpr char const *name = "by edge leaves";
	};

	/** @brief Form paths by rerooting one endpoint and restoring the root. */
	struct top_tree_by_rerooting : set_kind<ex::top_tree_by_rerooting> {
		static constexpr char const *name = "by rerooting";
	};

	/** @brief Cache the one-vertex cluster in each compress node. */
	struct top_tree_by_cached_vertex_cluster : set_kind<ex::top_tree_by_cached_vertex_cluster> {
		static constexpr char const *name = "by cached vertex cluster";
	};

	template<typename... Kinds, typename Fn>
		requires(std::invocable<Fn &, Kinds> && ...)
	void for_each_kind(Fn &&fn) {
		(fn(Kinds{}), ...);
	}

	/** @brief Raw draws per operation. */
	struct workload {
		std::vector<uint32> op, x, y;
		std::vector<ll> value;

		workload(std::size_t m, uint64 seed) : op(m), x(m), y(m), value(m) {
			std::mt19937_64 rng(seed);
			for (std::size_t i = 0; i < m; ++i) {
				op[i] = uint32(rng());
				x[i] = uint32(rng());
				y[i] = uint32(rng());
				value[i] = ll(rng() % 1'000'000);
			}
		}
	};

	enum class shape : atl::uint8 { random_tree,
									path,
									star,
									caterpillar,
									binary };

	constexpr shape all_shapes[] = {shape::random_tree, shape::path, shape::star, shape::caterpillar, shape::binary};

	char const *shape_name(shape s) {
		switch (s) {
		case shape::random_tree:
			return "random tree";
		case shape::path:
			return "path";
		case shape::star:
			return "star";
		case shape::caterpillar:
			return "caterpillar";
		default:
			return "binary";
		}
	}

	std::vector<uint32> make_parents(shape s, std::size_t n, uint64 seed) {
		std::mt19937_64 rng(seed);
		std::vector<uint32> parents(n, none);
		std::size_t const spine = n / 2;
		for (std::size_t v = 1; v < n; ++v) {
			switch (s) {
			case shape::random_tree:
				parents[v] = uint32(rng() % v);
				break;
			case shape::path:
				parents[v] = uint32(v - 1);
				break;
			case shape::star:
				parents[v] = 0;
				break;
			case shape::caterpillar:
				parents[v] = v < spine ? uint32(v - 1) : uint32(rng() % spine);
				break;
			default:
				parents[v] = uint32((v - 1) / 2);
			}
		}
		return parents;
	}

	/** @brief Query endpoints: on the path shape, the two far ends (first and last 1%). */
	std::pair<uint32, uint32> endpoints(shape s, uint32 a, uint32 b, std::size_t n) {
		if (s == shape::path) {
			std::size_t const band = std::max<std::size_t>(1, n / 100);
			return {uint32(a % band), uint32(n - 1 - b % band)};
		}
		return {uint32(a % n), uint32(b % n)};
	}

	struct sum_count {
		ll sum = 0, count = 0;
	};

	struct sum_op {
		constexpr sum_count operator()(sum_count const &x, sum_count const &y) const { return {x.sum + y.sum, x.count + y.count}; }
	};

	struct add_mapping {
		constexpr sum_count operator()(ll c, sum_count const &x) const { return {x.sum + c * x.count, x.count}; }
	};

	struct affine {
		ll a = 1, b = 0;
	};

	constexpr affine then(affine const &f, affine const &g) { return {g.a * f.a % mod, (g.a * f.b + g.b) % mod}; }

	struct affine_then {
		constexpr affine operator()(affine const &f, affine const &g) const { return then(f, g); }
	};

	/** @brief The empty hanging set of a pure path algebra. */
	struct nothing {};

	struct no_rake {
		constexpr nothing operator()(nothing, nothing) const { return {}; }
	};

	struct composite_vertex {
		constexpr affine operator()(nothing, affine const &f) const { return f; }
	};

	struct composite_edge {
		constexpr nothing operator()(affine const &) const { return {}; }
	};

	/** @brief 's user-reversed composite: both orders in the cluster, a swap reverses. */
	struct two_way {
		affine forward, backward;
	};

	struct two_way_compress {
		constexpr two_way operator()(two_way const &x, two_way const &y) const {
			return {then(x.forward, y.forward), then(y.backward, x.backward)};
		}
	};

	struct two_way_vertex {
		constexpr two_way operator()(nothing, affine const &f) const { return {f, f}; }
	};

	struct two_way_edge {
		constexpr nothing operator()(two_way const &) const { return {}; }
	};

	struct two_way_reverse {
		constexpr two_way operator()(two_way const &x) const { return {x.backward, x.forward}; }
	};

	/** @brief Structure only: every cluster empty. */
	struct no_compress {
		constexpr nothing operator()(nothing, nothing) const { return {}; }
	};

	struct no_vertex {
		constexpr nothing operator()(nothing, nothing) const { return {}; }
	};

	struct no_edge {
		constexpr nothing operator()(nothing) const { return {}; }
	};

	using structure_top_tree = atl::top_tree<nothing, nothing, nothing, no_compress, no_rake, no_vertex, no_edge, atl::commutative_fold_t>;
}

TEST_CASE("top_tree benchmarks: static shapes - path add + path sum", "[top_tree]") {
	std::map<std::string, long long> answers;
	for (shape const s : all_shapes) {
		for (std::size_t const n : sizes) {
			auto const parents = make_parents(s, n, 11);
			std::vector<ll> const values(n, 1);
			workload const w(n, 12);
			for_each_kind<library, top_tree_by_cached_vertex_cluster, top_tree_by_rerooting, top_tree_by_interleaved_layout, top_tree_by_per_splay_push, top_tree_by_rake_append>([&]<typename Kind>(Kind)
																																													  requires requires { typename Kind::template type<tc::diameter<ll>, atl::edge_values_t>; }
																																												  {
																																													  {
																																														  auto const row = label(std::string(Kind::name) + " (sum)", shape_name(s), n);
																																														  auto run_workload = [&] {
																																															  typename Kind::template type<tc::sum<ll>> t(values);
																																															  t.assign_parents(parents);
																																															  ll acc = 0;
																																															  for (std::size_t i = 0; i < n; ++i) {
																																																  auto const [u, v] = endpoints(s, w.x[i], w.y[i], n);
																																																  if (w.op[i] % 2 == 0)
																																																	  t.path_add(u, v, w.value[i]);
																																																  else
																																																	  acc += t.path_sum(u, v);
																																															  }
																																															  return acc;
																																														  };
																																														  check_answers(answers, row, run_workload);
																																														  BENCHMARK(std::string(row)) { return run_workload(); };
																																													  }
																																												  });
			{
				auto const row = label("link_cut_tree (commutative, lazy add)", shape_name(s), n);
				auto run_workload = [&] {
					atl::link_cut_tree t(atl::commutative_fold, std::vector<sum_count>(n, {1, 1}), sum_count{}, 0LL, sum_op{},
										 add_mapping{}, std::plus<>{});
					t.assign_parents(parents);
					ll acc = 0;
					for (std::size_t i = 0; i < n; ++i) {
						auto const [u, v] = endpoints(s, w.x[i], w.y[i], n);
						if (w.op[i] % 2 == 0)
							t.apply(u, v, w.value[i]);
						else
							acc += t.fold(u, v).sum;
					}
					return acc;
				};
				check_answers(answers, row, run_workload);
				BENCHMARK(std::string(row)) { return run_workload(); };
			}
		}
	}
}

TEST_CASE("top_tree benchmarks: point set + affine path composite (reversal modes)", "[top_tree]") {
	std::map<std::string, long long> answers;
	for (std::size_t const n : sizes) {
		auto const parents = make_parents(shape::random_tree, n, 21);
		std::vector<affine> values(n);
		std::mt19937_64 rng(22);
		for (auto &f : values) f = {ll(rng() % mod), ll(rng() % mod)};
		workload const w(n, 23);
		auto const run = [&]<typename Tree, typename Read>(Tree &t, Read const &read)
			requires requires(Tree &forest) { forest.assign_parents(parents); } && std::invocable<Read const &, decltype(std::declval<Tree &>().fold(uint32{}, uint32{}))>
		{
			t.assign_parents(parents);
			ll acc = 0;
			for (std::size_t i = 0; i < n; ++i) {
				uint32 const u = uint32(w.x[i] % n), v = uint32(w.y[i] % n);
				if (w.op[i] % 2 == 0) {
					t.set(u, affine{w.value[i] + 1, w.value[i]});
				} else {
					affine const f = read(t.fold(u, v));
					acc += (f.a * w.value[i] + f.b) % mod;
				}
			}
			return acc;
		};
		{
			auto const row = label("top_tree library reversed cluster (Reverse = void)", "set + affine composite", n);
			auto run_workload = [&] {
				atl::top_tree t(values, affine{}, nothing{}, affine_then{}, no_rake{}, composite_vertex{}, composite_edge{});
				return run(t, [](affine const &f) { return f; });
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("by rerooting (Reverse = void)", "set + affine composite", n);
			auto run_workload = [&] {
				ex::top_tree_by_rerooting<affine, affine, nothing, affine_then, no_rake, composite_vertex, composite_edge> t(
					values, affine{}, nothing{}, affine_then{}, no_rake{}, composite_vertex{}, composite_edge{});
				return run(t, [](affine const &f) { return f; });
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("top_tree user Reverse (two-way cluster)", "set + affine composite", n);
			auto run_workload = [&] {
				atl::top_tree t(values, two_way{}, nothing{}, two_way_compress{}, no_rake{}, two_way_vertex{}, two_way_edge{},
								two_way_reverse{});
				return run(t, [](two_way const &f) { return f.forward; });
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("link_cut_tree", "set + affine composite", n);
			auto run_workload = [&] {
				atl::link_cut_tree t(values, affine{}, affine_then{});
				return run(t, [](affine const &f) { return f; });
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
	}
}

namespace {
	/** @brief One query: kind 0 = cut (a, b) then link (c, d); 1 = update at (a, b); 2 = query at (a, b). */
	struct query {
		uint32 kind, a, b, c, d;
		ll x;
	};

	/**
	 * @brief Thirds of edge replacement (cut a random edge, reconnect a random
	 *        vertex to the other side), an update and a query on a random
	 *        oriented edge (on @p is_vertex_update, the update is a point add at
	 *        a random vertex); made valid against a simulation outside the timing.
	 */
	std::vector<query> dynamic_queries(std::vector<uint32> const &parents, bool is_vertex_update, uint64 seed) {
		std::size_t const n = parents.size();
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

TEST_CASE("top_tree benchmarks: churn + vertex add + subtree sum", "[top_tree]") {
	std::map<std::string, long long> answers;
	for (std::size_t const n : sizes) {
		auto const parents = make_parents(shape::random_tree, n, 31);
		auto const qs = dynamic_queries(parents, true, 32);
		std::mt19937_64 rng(33);
		std::vector<ll> values(n);
		for (ll &v : values) v = ll(rng() % 1'000'000);
		auto const run = [&]<typename Tree, typename Add, typename Read>(Tree &t, Add const &add, Read const &read)
			requires std::invocable<Add const &, Tree &, uint32, ll> && std::invocable<Read const &, Tree &, uint32, uint32>
		{
			t.assign_parents(parents);
			ll acc = 0;
			for (query const &q : qs) {
				if (q.kind == 0) {
					t.cut(q.a, q.b);
					t.link(q.c, q.d);
				} else if (q.kind == 1) {
					add(t, q.a, q.x);
				} else {
					acc += read(t, q.a, q.b);
				}
			}
			return acc;
		};
		auto const add = []<typename Tree>(Tree &t, uint32 v, ll x)
			requires requires(Tree &forest) { forest.update(v, [](ll value) { return value; }); }
		{ t.update(v, [x](ll a) { return a + x; }); };
		{
			auto const row = label("sum_top_tree", "churn + vertex add + subtree sum", n);
			auto run_workload = [&] {
				atl::sum_top_tree<ll> t(values);
				return run(t, add, []<typename Tree>(Tree &tree, uint32 a, uint32 b)
							   requires requires { tree.subtree_sum(a, b); }
						   { return tree.subtree_sum(a, b); });
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("by rerooting (sum)", "churn + vertex add + subtree sum", n);
			auto run_workload = [&] {
				top_tree_by_rerooting::type<tc::sum<ll>> t(values);
				return run(t, add, []<typename Tree>(Tree &tree, uint32 a, uint32 b)
							   requires requires { tree.subtree_sum(a, b); }
						   { return tree.subtree_sum(a, b); });
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("subtree_link_cut_tree", "churn + vertex add + subtree sum", n);
			auto run_workload = [&] {
				atl::subtree_link_cut_tree t(values, 0LL, std::plus<>{}, std::negate<>{});
				return run(t, add, []<typename Tree>(Tree &tree, uint32 a, uint32 b)
							   requires requires { tree.subtree_fold(a, b); }
						   { return tree.subtree_fold(a, b); });
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("euler_tour_tree", "churn + vertex add + subtree sum", n);
			auto run_workload = [&] {
				atl::euler_tour_tree t(values, 0LL, std::plus<>{});
				return run(t, add, []<typename Tree>(Tree &tree, uint32 a, uint32 b)
							   requires requires { tree.subtree_fold(a, b); }
						   { return tree.subtree_fold(a, b); });
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
	}
}

TEST_CASE("top_tree benchmarks: churn + subtree add + subtree sum (batch shape)", "[top_tree]") {
	std::map<std::string, long long> answers;
	for (std::size_t const n : sizes) {
		auto const parents = make_parents(shape::random_tree, n, 41);
		auto const qs = dynamic_queries(parents, false, 42);
		std::mt19937_64 rng(43);
		std::vector<ll> values(n);
		for (ll &v : values) v = ll(rng() % 1'000'000);
		auto const churn = [&]<typename Tree>(Tree &t)
			requires(requires(Tree &forest) { forest.assign_parents(parents); } || requires(Tree &forest) { forest.assign_parents(parents, std::vector<ll>{}); })
		{
			t.assign_parents(parents);
			ll acc = 0;
			for (query const &q : qs) {
				if (q.kind == 0) {
					t.cut(q.a, q.b);
					t.link(q.c, q.d);
				} else if (q.kind == 1) {
					t.subtree_add(q.a, q.b, q.x);
				} else {
					acc += t.subtree_sum(q.a, q.b);
				}
			}
			return acc; };
		{
			auto const row = label("by cached vertex cluster (sum)", "churn + subtree add + subtree sum", n);
			auto run_workload = [&] {
				top_tree_by_cached_vertex_cluster::type<tc::sum<ll>> t(values);
				return churn(t);
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("by rerooting (sum)", "churn + subtree add + subtree sum", n);
			auto run_workload = [&] {
				top_tree_by_rerooting::type<tc::sum<ll>> t(values);
				return churn(t);
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("sum_top_tree", "churn + subtree add + subtree sum", n);
			auto run_workload = [&] {
				atl::sum_top_tree<ll> t(values);
				t.assign_parents(parents);
				ll acc = 0;
				for (query const &q : qs) {
					if (q.kind == 0) {
						t.cut(q.a, q.b);
						t.link(q.c, q.d);
					} else if (q.kind == 1) {
						t.subtree_add(q.a, q.b, q.x);
					} else {
						acc += t.subtree_sum(q.a, q.b);
					}
				}
				return acc;
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("euler_tour_tree", "churn + subtree add + subtree sum", n);
			auto run_workload = [&] {
				std::vector<sum_count> counted(n);
				for (std::size_t v = 0; v < n; ++v) counted[v] = {values[v], 1};
				atl::euler_tour_tree t(counted, sum_count{}, 0LL, sum_op{}, add_mapping{}, std::plus<>{});
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
	}
}

TEST_CASE("top_tree benchmarks: mixed path/subtree mix (re-root, path / subtree assign and add, min / max / sum)", "[top_tree]") {
	{
		atl::extremes_top_tree<ll> tree(std::vector<ll>{1, 2, 4, 8});
		tree.assign_parents(std::vector<uint32>{none, 0, 0, 1});
		tree.path_add(3, 2, 10);
		tree.subtree_assign(1, 7);
		REQUIRE(tree.path_sum(2, 3) == 39);
		REQUIRE(tree.component_min(0) == 7);
		REQUIRE(tree.component_max(0) == 14);
		tree.make_root(3);
		REQUIRE(tree.subtree_sum(0) == 25);
	}

	std::map<std::string, long long> answers;
	for (std::size_t const n : sizes) {
		auto const parents = make_parents(shape::random_tree, n, 51);
		std::mt19937_64 rng(52);
		std::vector<ll> values(n);
		for (ll &v : values) v = ll(rng() % 1'000'000);
		workload const w(n, 53);
		{
			auto const row = label("extremes_top_tree", "mixed path/subtree mix", n);
			auto run_workload = [&] {
				atl::extremes_top_tree<ll> t(values);
				t.assign_parents(parents);
				ll acc = 0;
				for (std::size_t i = 0; i < n; ++i) {
					uint32 const u = uint32(w.x[i] % n), v = uint32(w.y[i] % n);
					ll const x = w.value[i];
					switch (w.op[i] % 10) {
					case 0:
						t.make_root(u);
						break;
					case 1:
						t.path_assign(u, v, x);
						break;
					case 2:
						t.path_add(u, v, x);
						break;
					case 3:
						t.subtree_assign(u, x);
						break;
					case 4:
						t.subtree_add(u, x);
						break;
					case 5:
						acc += t.path_min(u, v) + t.path_max(u, v);
						break;
					case 6:
						acc += t.path_sum(u, v) % mod;
						break;
					case 7:
						acc += t.subtree_min(u) + t.subtree_max(u);
						break;
					case 8:
						acc += t.subtree_sum(u) % mod;
						break;
					default:
						// change u's parent to v unless v is in u's subtree (mixed path/subtree's operation 9)
						if (t.parent(u) != none && t.lca(u, v) != u) {
							t.cut_from_parent(u);
							t.link(u, v);
						}
					}
				}
				return acc;
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
	}
}

TEST_CASE("top_tree benchmarks: dynamic diameter", "[top_tree]") {
	std::map<std::string, long long> answers;
	for (shape const s : {shape::random_tree, shape::path, shape::star, shape::caterpillar}) {
		for (std::size_t const n : sizes) {
			auto const parents = make_parents(s, n, 61);
			std::vector<ll> weights(n);
			std::mt19937_64 rng(62);
			for (ll &x : weights) x = ll(rng() % 1'000'000);
			workload const w(n, 63);
			for_each_kind<library, top_tree_by_cached_vertex_cluster, top_tree_by_edge_leaves, top_tree_by_interleaved_layout, top_tree_by_per_splay_push, top_tree_by_rake_append>([&]<typename Kind>(Kind)
																																														requires requires { typename Kind::template type<tc::diameter<ll>, atl::edge_values_t>; }
																																													{
																																														{
																																															auto const row = label(std::string(Kind::name) + " (diameter)", (std::string(shape_name(s)) + ": weight set + diameter").c_str(),
																																																				   n);
																																															auto run_workload = [&] {
																																																typename Kind::template type<tc::diameter<ll>, atl::edge_values_t> t(n);
																																																t.assign_parents(parents, weights);
																																																ll acc = 0;
																																																for (std::size_t i = 0; i < n; ++i) {
																																																	uint32 const v = 1 + w.x[i] % uint32(n - 1);
																																																	t.set(v - 1, w.value[i]); // edge ids follow v in assign_parents
																																																	acc += t.diameter(0);
																																																}
																																																return acc;
																																															};
																																															check_answers(answers, row, run_workload);
																																															BENCHMARK(std::string(row)) { return run_workload(); };
																																														}
																																													});
		}
	}
	for (std::size_t const n : sizes) {
		auto const parents = make_parents(shape::random_tree, n, 64);
		std::vector<ll> weights(n);
		std::mt19937_64 rng(65);
		for (ll &x : weights) x = ll(rng() % 1'000'000);
		auto const qs = dynamic_queries(parents, false, 66);
		auto const churn = [&]<typename Tree>(Tree &t)
			requires(requires(Tree &forest) { forest.assign_parents(parents); } || requires(Tree &forest) { forest.assign_parents(parents, std::vector<ll>{}); })
		{
			t.assign_parents(parents, weights);
			ll acc = 0;
			for (query const &q : qs) {
				if (q.kind == 0) {
					t.cut(q.a, q.b);
					t.link(q.c, q.d, q.x);
				} else if (q.kind == 1) {
					t.set(t.edge(q.a, q.b), q.x);
				} else {
					acc += t.diameter(q.a) + t.eccentricity(q.b);
				}
			}
			return acc; };
		{
			auto const row = label("by edge leaves (diameter)", "churn: edge replacement + diameter", n);
			auto run_workload = [&] {
				top_tree_by_edge_leaves::type<tc::diameter<ll>, atl::edge_values_t> t(n);
				return churn(t);
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("diameter_top_tree", "churn: edge replacement + diameter", n);
			auto run_workload = [&] {
				atl::diameter_top_tree<ll> t(n);
				t.assign_parents(parents, weights);
				ll acc = 0;
				for (query const &q : qs) {
					if (q.kind == 0) {
						t.cut(q.a, q.b);
						t.link(q.c, q.d, q.x);
					} else if (q.kind == 1) {
						t.set(t.edge(q.a, q.b), q.x);
					} else {
						acc += t.diameter(q.a) + t.eccentricity(q.b);
					}
				}
				return acc;
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
	}
}

TEST_CASE("top_tree benchmarks: marked toggles + distance sums", "[top_tree]") {
	std::map<std::string, long long> answers;
	for (std::size_t const n : sizes) {
		auto const parents = make_parents(shape::random_tree, n, 71);
		std::vector<ll> weights(n);
		std::mt19937_64 rng(72);
		for (ll &x : weights) x = ll(rng() % 1000);
		workload const w(n, 73);
		auto const run = [&]<typename Tree>(Tree &t)
			requires(requires(Tree &forest) { forest.assign_parents(parents); } || requires(Tree &forest) { forest.assign_parents(parents, std::vector<ll>{}); })
		{
			t.assign_parents(parents, weights);
			ll acc = 0;
			for (std::size_t i = 0; i < n; ++i) {
				uint32 const v = uint32(w.x[i] % n);
				if (w.op[i] % 2 == 0)
					t.set_marked(v, w.op[i] % 4 == 0);
				else
					acc += t.marked_distance_sum(v) % mod;
			}
			return acc; };
		{
			auto const row = label("marked_distance_top_tree", "marked toggles + distance sums", n);
			auto run_workload = [&] {
				atl::marked_distance_top_tree<ll> t(n);
				return run(t);
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("by edge leaves (marked distance)", "marked toggles + distance sums", n);
			auto run_workload = [&] {
				top_tree_by_edge_leaves::type<tc::marked_distance<ll>, atl::edge_values_t> t(n);
				return run(t);
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
	}
}

TEST_CASE("top_tree benchmarks: structure only - links, churn", "[top_tree]") {
	std::map<std::string, long long> answers;
	for (std::size_t const n : sizes) {
		auto const parents = make_parents(shape::random_tree, n, 81);
		// thirds of cut / link / connected, made valid against a simulation outside the timing
		std::mt19937_64 rng(82);
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
		auto const run = [&]<typename Tree>(Tree &t)
			requires(requires(Tree &forest) { forest.assign_parents(parents); } || requires(Tree &forest) { forest.assign_parents(parents, std::vector<ll>{}); })
		{
			t.assign_parents(parents);
			ll acc = 0;
			for (query const &q : qs) {
				if (q.kind == 0)
					t.cut(q.a, q.b);
				else if (q.kind == 1)
					t.link(q.a, q.b);
				else
					acc += t.connected(q.a, q.b);
			}
			return acc; };
		{
			auto const row = label("top_tree (empty clusters)", "N - 1 links (random tree)", n);
			auto run_workload = [&] {
				structure_top_tree t(n, nothing{}, nothing{}, nothing{});
				for (uint32 v = 1; v < n; ++v) t.link(v, parents[v]);
				return t.connected(0, uint32(n - 1));
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("link_cut_tree<>", "N - 1 links (random tree)", n);
			auto run_workload = [&] {
				atl::link_cut_tree t(n);
				for (uint32 v = 1; v < n; ++v) t.link(v, parents[v]);
				return t.connected(0, uint32(n - 1));
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("euler_tour_tree<>", "N - 1 links (random tree)", n);
			auto run_workload = [&] {
				atl::euler_tour_tree t(n);
				for (uint32 v = 1; v < n; ++v) t.link(v, parents[v]);
				return t.component_size(0);
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("top_tree (empty clusters)", "assign_parents + churn (cut / link / connected thirds)", n);
			auto run_workload = [&] {
				structure_top_tree t(n, nothing{}, nothing{}, nothing{});
				return run(t);
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("by rake append (empty clusters)", "assign_parents + churn (cut / link / connected thirds)", n);
			auto run_workload = [&] {
				ex::top_tree_by_rake_append<nothing, nothing, nothing, no_compress, no_rake, no_vertex, no_edge, atl::commutative_fold_t> t(
					n, nothing{}, nothing{}, nothing{});
				return run(t);
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("link_cut_tree<>", "assign_parents + churn (cut / link / connected thirds)", n);
			auto run_workload = [&] {
				atl::link_cut_tree t(n);
				return run(t);
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
		{
			auto const row = label("euler_tour_tree<>", "assign_parents + churn (cut / link / connected thirds)", n);
			auto run_workload = [&] {
				atl::euler_tour_tree t(n);
				return run(t);
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
	}
}
