/**
 * @file
 * @brief Path folds, lazy updates, connectivity churn and incremental MST on random, path, star and caterpillar forests. Contenders change rerooting, size storage, ancestor push, or node layout; commutative folding is a public mode.
 *
 * Each timed loop measures whole-workload latency, including construction;
 * operations depend on earlier mutations. Untimed runs compare answer checksums.
 * Xeon Platinum 8168, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-25,
 * -O2 / -O2 -march=native. Random churn N=Q=2e5: library 205 / 211 ms, by rerooting 223 / 232 ms.
 * Full tables: docs/link-cut-tree.md.
 */
#include <atl/ds/detail/splay_layout.hpp>
#include <atl/ds/link_cut_tree.hpp>
#include <atl/experimental/ds/link_cut_tree_variants.hpp>
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
#include <tuple>
#include <type_traits>
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
	using atl::detail::splay_layout_columnar;
	using atl::detail::splay_layout_interleaved;
	using atl::detail::splay_layout_split;

	constexpr uint32 none = atl::graph::none;
	constexpr ll mod = 998'244'353;

	template<typename Engine, atl::detail::splay_layout Layout, bool IsSized, bool IsEvertForm, bool IsCommutative = false>
	struct variant {
		template<typename Value = void, typename Op = void, typename Tag = void, typename Mapping = void, typename Compose = void,
				 typename FoldOrder = void, typename ValuePlacement = void>
		using type = ex::link_cut_tree_by_configuration<Engine, Layout, IsSized, IsEvertForm, Value, Op, Tag, Mapping, Compose,
														std::conditional_t<IsCommutative, atl::commutative_fold_t, FoldOrder>,
														ValuePlacement>;
	};

	/** @brief Adapt the library class template to the common contender loop. */
	struct library {
		static constexpr char const *name = "library";
		template<typename Value = void, typename Op = void, typename Tag = void, typename Mapping = void, typename Compose = void,
				 typename FoldOrder = void, typename ValuePlacement = void>
		using type = atl::link_cut_tree<Value, Op, Tag, Mapping, Compose, FoldOrder, ValuePlacement>;
	};

	/** @brief Use the public commutative-fold mode to store one aggregate per node. */
	struct commutative : variant<atl::splay_engine, splay_layout_interleaved, true, false, true> {
		static constexpr char const *name = "commutative_fold";
	};

	/** @brief Form paths by rerooting one endpoint and restoring the root. */
	struct link_cut_tree_by_rerooting : variant<atl::splay_engine, splay_layout_interleaved, true, true> {
		static constexpr char const *name = "by rerooting";
	};

	/** @brief Omit node sizes and the indexed path queries they support. */
	struct link_cut_tree_by_omitting_size : variant<atl::splay_engine, splay_layout_interleaved, false, false> {
		static constexpr char const *name = "by omitting size";
	};

	/** @brief Push reversal flags with two upward parity walks. */
	struct link_cut_tree_by_parity_walk : variant<ex::link_cut_parity_engine, splay_layout_interleaved, true, false> {
		static constexpr char const *name = "by parity walk";
	};

	/** @brief Keep node links apart from payloads. */
	struct link_cut_tree_by_split_layout : variant<atl::splay_engine, splay_layout_split, true, false> {
		static constexpr char const *name = "by split layout";
	};

	/** @brief Keep each node field in a separate array. */
	struct link_cut_tree_by_columnar_layout : variant<atl::splay_engine, splay_layout_columnar, true, false> {
		static constexpr char const *name = "by columnar layout";
	};

	template<typename... Kinds, typename Fn>
		requires(std::invocable<Fn &, Kinds> && ...)
	void for_each_kind(Fn &&fn) {
		(fn(Kinds{}), ...);
	}

	constexpr std::size_t sizes[] = {200'000, 500'000};

	std::string label(char const *kind, char const *shape, std::size_t n) {
		return std::string(kind) + " -- " + shape + ", N=Q=" + (n == 200'000 ? "2e5" : "5e5");
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
									caterpillar };

	char const *shape_name(shape s) {
		switch (s) {
		case shape::random_tree:
			return "random tree";
		case shape::path:
			return "path";
		case shape::star:
			return "star";
		default:
			return "caterpillar";
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
			default:
				parents[v] = v < spine ? uint32(v - 1) : uint32(rng() % spine);
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

	struct affine {
		ll a = 1, b = 0;
	};

	constexpr affine then(affine const &f, affine const &g) { return {g.a * f.a % mod, (g.a * f.b + g.b) % mod}; }

	struct affine_then {
		constexpr affine operator()(affine const &f, affine const &g) const { return then(f, g); }
	};

	struct sum_count {
		ll sum = 0, count = 0;
	};

	struct sum_op {
		constexpr sum_count operator()(sum_count const &x, sum_count const &y) const {
			ll const s = x.sum + y.sum;
			return {s >= mod ? s - mod : s, x.count + y.count};
		}
	};

	struct affine_mapping {
		constexpr sum_count operator()(affine const &f, sum_count const &x) const {
			return {(f.a * x.sum + f.b * x.count) % mod, x.count};
		}
	};

	/** @brief Tag composition, "g first, then f". */
	struct affine_compose {
		constexpr affine operator()(affine const &f, affine const &g) const { return then(g, f); }
	};

}

TEST_CASE("link_cut_tree benchmarks: static shapes - point add + path sum", "[link_cut_tree]") {
	std::map<std::string, long long> answers;
	for (shape const s : {shape::random_tree, shape::path, shape::star, shape::caterpillar}) {
		for (std::size_t const n : sizes) {
			auto const parents = make_parents(s, n, 11);
			std::vector<ll> const values(n, 1);
			workload const w(n, 12);
			for_each_kind<library, commutative, link_cut_tree_by_rerooting, link_cut_tree_by_omitting_size, link_cut_tree_by_parity_walk, link_cut_tree_by_split_layout, link_cut_tree_by_columnar_layout>(
				[&]<typename Kind>(Kind)
					requires requires { typename Kind::template type<ll, std::plus<>>; }
				{
					{
						auto const row = label(Kind::name, shape_name(s), n);
						auto run_workload = [&] {
							typename Kind::template type<ll, std::plus<>> t(values, 0LL, std::plus<>{});
							t.assign_parents(parents);
							ll acc = 0;
							for (std::size_t i = 0; i < n; ++i) {
								auto const [u, v] = endpoints(s, w.x[i], w.y[i], n);
								if (w.op[i] % 2 == 0)
									t.update(u, [d = w.value[i]](ll a) { return a + d; });
								else
									acc += t.fold(u, v);
							}
							return acc;
						};
						check_answers(answers, row, run_workload);
						BENCHMARK(std::string(row)) { return run_workload(); };
					}
				});
		}
	}
}

TEST_CASE("link_cut_tree benchmarks: random churn (cut / link / path sum thirds)", "[link_cut_tree]") {
	std::map<std::string, long long> answers;
	for (std::size_t const n : sizes) {
		auto const parents = make_parents(shape::random_tree, n, 21);
		std::vector<ll> const values(n, 1);
		workload const w(n, 22);
		for_each_kind<library, commutative, link_cut_tree_by_rerooting, link_cut_tree_by_omitting_size, link_cut_tree_by_parity_walk, link_cut_tree_by_split_layout, link_cut_tree_by_columnar_layout>([&]<typename Kind>(Kind)
																																																		   requires requires { typename Kind::template type<ll, std::plus<>>; }
																																																	   {
																																																		   {
																																																			   auto const row = label(Kind::name, "random churn", n);
																																																			   auto run_workload = [&] {
																																																				   typename Kind::template type<ll, std::plus<>> t(values, 0LL, std::plus<>{});
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
																																																			   check_answers(answers, row, run_workload);
																																																			   BENCHMARK(std::string(row)) { return run_workload(); };
																																																		   }
																																																	   });
	}
}

TEST_CASE("link_cut_tree benchmarks: make_root every query", "[link_cut_tree]") {
	std::map<std::string, long long> answers;
	for (std::size_t const n : sizes) {
		auto const parents = make_parents(shape::random_tree, n, 31);
		std::vector<ll> const values(n, 1);
		workload const w(n, 32);
		for_each_kind<library, commutative, link_cut_tree_by_rerooting, link_cut_tree_by_omitting_size, link_cut_tree_by_parity_walk, link_cut_tree_by_split_layout, link_cut_tree_by_columnar_layout>([&]<typename Kind>(Kind)
																																																		   requires requires { typename Kind::template type<ll, std::plus<>>; }
																																																	   {
																																																		   {
																																																			   auto const row = label(Kind::name, "make_root + path sum", n);
																																																			   auto run_workload = [&] {
																																																				   typename Kind::template type<ll, std::plus<>> t(values, 0LL, std::plus<>{});
																																																				   t.assign_parents(parents);
																																																				   ll acc = 0;
																																																				   for (std::size_t i = 0; i < n; ++i) {
																																																					   uint32 const u = w.x[i] % n, v = w.y[i] % n;
																																																					   t.make_root(u);
																																																					   acc += t.fold(u, v);
																																																				   }
																																																				   return acc;
																																																			   };
																																																			   check_answers(answers, row, run_workload);
																																																			   BENCHMARK(std::string(row)) { return run_workload(); };
																																																		   }
																																																	   });
	}
}

TEST_CASE("link_cut_tree benchmarks: point set + path composite", "[link_cut_tree]") {
	std::map<std::string, long long> answers;
	for (std::size_t const n : sizes) {
		auto const parents = make_parents(shape::random_tree, n, 41);
		std::vector<affine> values(n);
		std::mt19937_64 rng(42);
		for (auto &f : values) f = {ll(rng() % mod), ll(rng() % mod)};
		workload const w(n, 43);
		for_each_kind<library, link_cut_tree_by_rerooting, link_cut_tree_by_omitting_size, link_cut_tree_by_parity_walk, link_cut_tree_by_split_layout, link_cut_tree_by_columnar_layout>([&]<typename Kind>(Kind)
																																															  requires requires { typename Kind::template type<ll, std::plus<>>; }
																																														  {
																																															  {
																																																  auto const row = label(Kind::name, "set + affine composite", n);
																																																  auto run_workload = [&] {
																																																	  typename Kind::template type<affine, affine_then> t(values, affine{}, affine_then{});
																																																	  t.assign_parents(parents);
																																																	  ll acc = 0;
																																																	  for (std::size_t i = 0; i < n; ++i) {
																																																		  uint32 const u = w.x[i] % n, v = w.y[i] % n;
																																																		  if (w.op[i] % 2 == 0) {
																																																			  t.set(u, affine{w.value[i] + 1, w.value[i]});
																																																		  } else {
																																																			  affine const f = t.fold(u, v);
																																																			  acc += (f.a * w.value[i] + f.b) % mod;
																																																		  }
																																																	  }
																																																	  return acc;
																																																  };
																																																  check_answers(answers, row, run_workload);
																																																  BENCHMARK(std::string(row)) { return run_workload(); };
																																															  }
																																														  });
	}
}

TEST_CASE("link_cut_tree benchmarks: lazy affine path update + path sum (Tree II)", "[link_cut_tree]") {
	std::map<std::string, long long> answers;
	for (std::size_t const n : sizes) {
		auto const parents = make_parents(shape::random_tree, n, 51);
		std::vector<sum_count> const values(n, sum_count{1, 1});
		workload const w(n, 52);
		for_each_kind<library, commutative, link_cut_tree_by_rerooting, link_cut_tree_by_omitting_size, link_cut_tree_by_split_layout, link_cut_tree_by_columnar_layout>([&]<typename Kind>(Kind)
																																											 requires requires { typename Kind::template type<ll, std::plus<>>; }
																																										 {
																																											 {
																																												 auto const row = label(Kind::name, "lazy affine path + sum", n);
																																												 auto run_workload = [&] {
																																													 typename Kind::template type<sum_count, sum_op, affine, affine_mapping, affine_compose> t(values, sum_count{},
																																																																			   affine{});
																																													 t.assign_parents(parents);
																																													 ll acc = 0;
																																													 for (std::size_t i = 0; i < n; ++i) {
																																														 uint32 const u = w.x[i] % n, v = w.y[i] % n;
																																														 if (w.op[i] % 2 == 0)
																																															 t.apply(u, v, w.op[i] % 4 == 0 ? affine{w.value[i], 0} : affine{1, w.value[i]});
																																														 else
																																															 acc += t.fold(u, v).sum;
																																													 }
																																													 return acc;
																																												 };
																																												 check_answers(answers, row, run_workload);
																																												 BENCHMARK(std::string(row)) { return run_workload(); };
																																											 }
																																										 });
	}
}

TEST_CASE("link_cut_tree benchmarks: structure only - link / cut / connected", "[link_cut_tree]") {
	std::map<std::string, long long> answers;
	for (std::size_t const n : sizes) {
		auto const parents = make_parents(shape::random_tree, n, 61);
		workload const w(n, 62);
		for_each_kind<library, link_cut_tree_by_omitting_size, link_cut_tree_by_parity_walk, link_cut_tree_by_split_layout, link_cut_tree_by_columnar_layout>([&]<typename Kind>(Kind)
																																								  requires requires { typename Kind::template type<ll, std::plus<>>; }
																																							  {
																																								  {
																																									  auto const row = label(Kind::name, "structure-only churn", n);
																																									  auto run_workload = [&] {
																																										  typename Kind::template type<> t(n);
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
																																												  acc += t.connected(u, v);
																																											  }
																																										  }
																																										  return acc;
																																									  };
																																									  check_answers(answers, row, run_workload);
																																									  BENCHMARK(std::string(row)) { return run_workload(); };
																																								  }
																																							  });
	}
}

namespace {
	/** @brief (weight, label) under max: the heaviest edge on a path and which it is. */
	struct heaviest {
		constexpr std::pair<ll, uint32> operator()(std::pair<ll, uint32> const &a, std::pair<ll, uint32> const &b) const {
			return a.first < b.first ? b : a;
		}
	};
}

TEST_CASE("link_cut_tree benchmarks: edge tier - incremental MST by path max", "[link_cut_tree]") {
	std::map<std::string, long long> answers;
	using weighted = std::pair<ll, uint32>;
	for (std::size_t const n : sizes) {
		workload const w(n, 71);
		auto const run = [&]<typename Tree>(Tree &t)
			requires requires(Tree &forest) { forest.link(uint32{}, uint32{}, weighted{}); forest.fold(uint32{}, uint32{}); }
		{
			std::vector<std::pair<uint32, uint32>> ends(n);
			ll total = 0;
			for (std::size_t i = 0; i < n; ++i) {
				uint32 const u = w.x[i] % n, v = w.y[i] % n;
				if (u == v) continue;
				ll const weight = w.value[i];
				ends[i] = {u, v};
				if (!t.connected(u, v)) {
					t.link(u, v, weighted{weight, uint32(i)});
					total += weight;
				} else if (weighted const top = t.fold(u, v); top.first > weight) {
					t.cut(ends[top.second].first, ends[top.second].second);
					t.link(u, v, weighted{weight, uint32(i)});
					total += weight - top.first;
				}
			}
			return total;
		};
		for_each_kind<library, link_cut_tree_by_rerooting, link_cut_tree_by_parity_walk, link_cut_tree_by_split_layout, link_cut_tree_by_columnar_layout>([&]<typename Kind>(Kind)
																																							  requires requires { typename Kind::template type<ll, std::plus<>>; }
																																						  {
																																							  {
																																								  auto const row = label(Kind::name, "edge tier MST (commutative max)", n);
																																								  auto run_workload = [&] {
																																									  typename Kind::template type<weighted, heaviest, void, void, void, atl::commutative_fold_t, atl::edge_values_t> t(
																																										  n, weighted{-1, none}, heaviest{});
																																									  return run(t);
																																								  };
																																								  check_answers(answers, row, run_workload);
																																								  BENCHMARK(std::string(row)) { return run_workload(); };
																																							  }
																																						  });
		{
			auto const row = label("link_cut_tree, default fold order", "edge tier MST (commutative max)", n);
			auto run_workload = [&] {
				atl::link_cut_tree<weighted, heaviest, void, void, void, void, atl::edge_values_t> t(n, weighted{-1, none}, heaviest{});
				return run(t);
			};
			check_answers(answers, row, run_workload);
			BENCHMARK(std::string(row)) { return run_workload(); };
		}
	}
}
