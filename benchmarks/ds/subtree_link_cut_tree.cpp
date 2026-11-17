/**
 * @file
 * @brief Subtree sums, component sizes and path sums under cut/link churn. Contenders separate links and payloads or each field; a path-only link-cut reference measures subtree bookkeeping.
 *
 * Each timed loop measures whole-workload latency, including construction;
 * operations depend on earlier mutations. Untimed runs compare answer checksums.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-25,
 * -O2 / -O2 -march=native. Edge swap + vertex add + subtree sum: library 166 / 167 ms, split 169 / 173 ms, columnar 228 / 237 ms.
 * Full tables: docs/subtree-link-cut-tree.md.
 */
#include <atl/ds/detail/splay_layout.hpp>
#include <atl/ds/link_cut_tree.hpp>
#include <atl/ds/subtree_link_cut_tree.hpp>
#include <atl/types/algebraic.hpp>
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

	constexpr uint32 none = atl::graph::none;
	constexpr std::size_t n = 200'000;

	template<atl::detail::splay_layout Layout>
	using layout_tree = atl::detail::subtree_link_cut_tree_impl<Layout, ll, std::plus<>, std::negate<>>;

	/** @brief Adapt the library class template to the common contender loop. */
	struct library {
		static constexpr char const *name = "library";
		using type = atl::subtree_link_cut_tree<ll, std::plus<>, std::negate<>>;
	};

	/** @brief Keep node links apart from payloads. */
	struct subtree_link_cut_tree_by_split_layout {
		static constexpr char const *name = "by split layout";
		using type = layout_tree<atl::detail::splay_layout_split>;
	};

	/** @brief Keep each node field in a separate array. */
	struct subtree_link_cut_tree_by_columnar_layout {
		static constexpr char const *name = "by columnar layout";
		using type = layout_tree<atl::detail::splay_layout_columnar>;
	};

	template<typename... Kinds, typename Fn>
		requires(std::invocable<Fn &, Kinds> && ...)
	void for_each_kind(Fn &&fn) {
		(fn(Kinds{}), ...);
	}

	std::string label(char const *kind, char const *shape) { return std::string(kind) + " -- " + shape + ", N=Q=2e5"; }

	std::vector<uint32> random_tree(std::size_t size, std::mt19937_64 &rng) {
		std::vector<uint32> parents(size, none);
		for (std::size_t v = 1; v < size; ++v) parents[v] = uint32(rng() % v);
		return parents;
	}

	/** @brief One operation of the dynamic forest workload: 0 = cut (a, b) + link (c, d), 1 = add c to a, 2 = subtree of a under b. */
	struct lc_op {
		uint32 kind, a, b, c, d;
	};

	/**
	 * @brief A dynamic_tree_vertex_add_subtree_sum log on a random tree: thirds
	 *        of edge swaps (the new edge between random vertices of the two
	 *        sides, or the old edge back after 8 misses), vertex adds and
	 *        subtree sums of a random edge side; generated with a structural
	 *        link_cut_tree, outside the timed region.
	 */
	std::vector<lc_op> lc_log(std::vector<uint32> const &parents, std::size_t q, std::mt19937_64 &rng) {
		atl::link_cut_tree<> g(parents.size());
		g.assign_parents(parents);
		std::vector<std::pair<uint32, uint32>> edges;
		for (uint32 v = 1; v < parents.size(); ++v) edges.push_back({v, parents[v]});
		std::vector<lc_op> ops(q);
		for (auto &op : ops) {
			uint32 const kind = uint32(rng() % 3);
			std::size_t const i = rng() % edges.size();
			auto const [u, v] = edges[i];
			if (kind == 0) {
				g.cut(u, v);
				std::pair<uint32, uint32> e{u, v};
				for (int attempt = 0; attempt < 8; ++attempt) {
					uint32 const w = uint32(rng() % parents.size()), x = uint32(rng() % parents.size());
					if (!g.connected(w, x)) {
						e = {w, x};
						break;
					}
				}
				g.link(e.first, e.second);
				edges[i] = e;
				op = {0, u, v, e.first, e.second};
			} else if (kind == 1) {
				op = {1, uint32(rng() % parents.size()), 0, uint32(rng() % 1'000'000), 0};
			} else {
				op = rng() % 2 ? lc_op{2, u, v, 0, 0} : lc_op{2, v, u, 0, 0};
			}
		}
		return ops;
	}

	/** @brief Raw draws per operation, for the churn shapes. */
	struct workload {
		std::vector<uint32> op, x, y;

		workload(std::size_t m, uint64 seed) : op(m), x(m), y(m) {
			std::mt19937_64 rng(seed);
			for (std::size_t i = 0; i < m; ++i) op[i] = uint32(rng()), x[i] = uint32(rng()), y[i] = uint32(rng());
		}
	};
}

TEST_CASE("subtree_link_cut_tree benchmarks: batch dynamic_tree_vertex_add_subtree_sum shape", "[subtree_link_cut_tree]") {
	std::map<std::string, long long> answers;
	std::mt19937_64 rng(2026'10'24);
	auto const parents = random_tree(n, rng);
	auto const ops = lc_log(parents, n, rng);
	std::vector<ll> values(n);
	for (auto &v : values) v = ll(rng() % 1'000'000);
	for_each_kind<library, subtree_link_cut_tree_by_split_layout, subtree_link_cut_tree_by_columnar_layout>([&]<typename Kind>(Kind)
																												requires requires { typename Kind::type; }
																											{
																												{
																													auto const row = label(Kind::name, "link/cut swap + vertex add + subtree sum");
																													auto run_workload = [&] {
																														typename Kind::type t(values, 0LL, std::plus<>{}, std::negate<>{});
																														t.assign_parents(parents);
																														ll acc = 0;
																														for (auto const &[kind, a, b, c, d] : ops) {
																															if (kind == 0) {
																																t.cut(a, b);
																																t.link(c, d);
																															} else if (kind == 1) {
																																t.update(a, [c](ll x) { return x + c; });
																															} else {
																																acc += t.subtree_fold(a, b);
																															}
																														}
																														return acc;
																													};
																													check_answers(answers, row, run_workload);
																													BENCHMARK(std::string(row)) { return run_workload(); };
																												}
																											});
}

TEST_CASE("subtree_link_cut_tree benchmarks: component_size churn", "[subtree_link_cut_tree]") {
	std::map<std::string, long long> answers;
	std::mt19937_64 rng(31);
	auto const parents = random_tree(n, rng);
	workload const w(n, 32);
	std::vector<ll> const values(n, 1);
	for_each_kind<library, subtree_link_cut_tree_by_split_layout, subtree_link_cut_tree_by_columnar_layout>([&]<typename Kind>(Kind)
																												requires requires { typename Kind::type; }
																											{
																												{
																													auto const row = label(Kind::name, "cut / link / component_size thirds");
																													auto run_workload = [&] {
																														typename Kind::type t(values, 0LL, std::plus<>{}, std::negate<>{});
																														t.assign_parents(parents);
																														uint64 acc = 0;
																														for (std::size_t i = 0; i < n; ++i) {
																															uint32 const u = w.x[i] % n, v = w.y[i] % n;
																															switch (w.op[i] % 3) {
																															case 0:
																																if (t.parent(u) != none) t.cut_from_parent(u);
																																break;
																															case 1:
																																if (!t.connected(u, v)) t.link(u, v);
																																break;
																															default:
																																acc += t.component_size(u);
																															}
																														}
																														return acc;
																													};
																													check_answers(answers, row, run_workload);
																													BENCHMARK(std::string(row)) { return run_workload(); };
																												}
																											});
}

TEST_CASE("subtree_link_cut_tree benchmarks: path-sum churn against link_cut_tree", "[subtree_link_cut_tree]") {
	std::map<std::string, long long> answers;
	std::mt19937_64 rng(41);
	auto const parents = random_tree(n, rng);
	workload const w(n, 42);
	std::vector<ll> const values(n, 1);
	auto const churn = [&]<typename Tree>(Tree &t)
		requires requires(Tree &forest) { forest.assign_parents(parents); forest.connected(uint32{}, uint32{}); }
	{
		t.assign_parents(parents);
		ll acc = 0;
		for (std::size_t i = 0; i < n; ++i) {
			uint32 const u = w.x[i] % n, v = w.y[i] % n;
			switch (w.op[i] % 3) {
			case 0:
				if (t.parent(u) != none) t.cut_from_parent(u);
				break;
			case 1:
				if (!t.connected(u, v)) t.link(u, v);
				break;
			default:
				if (t.connected(u, v)) acc += t.fold(u, v);
			}
		}
		return acc;
	};
	{
		auto const row = label(library::name, "cut / link / path sum thirds");
		auto run_workload = [&] {
			library::type t(values, 0LL, std::plus<>{}, std::negate<>{});
			return churn(t);
		};
		check_answers(answers, row, run_workload);
		BENCHMARK(std::string(row)) { return run_workload(); };
	}
	{
		auto const row = label("reference: link_cut_tree, commutative_fold", "cut / link / path sum thirds");
		auto run_workload = [&] {
			atl::link_cut_tree t(atl::commutative_fold, values, 0LL, std::plus<>{});
			return churn(t);
		};
		check_answers(answers, row, run_workload);
		BENCHMARK(std::string(row)) { return run_workload(); };
	}
}
