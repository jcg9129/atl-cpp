/**
 * @file
 * @brief Keyed-map throughput for mixed mutations, Zipf finds, successor chains,
 * sliding working sets and sorted inserts, at 200,000 and 1,000,000 keys.
 * The library splays during descent with split storage. Contenders use a treap,
 * std::map, bottom-up splaying, or interleaved storage. Setup is timed too.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-24:
 * -O2 / -O2 -march=native, n=1e6: library mixed 1098/1065 ms,
 * successor chains 80.1/82.3 ms; full tables in docs/splay-map.md.
 */
#include <atl/ds/splay_engine.hpp>
#include <atl/ds/splay_map.hpp>
#include <atl/ds/treap_map.hpp>
#include <atl/experimental/ds/splay_map_variants.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using atl::uint32;
using atl::uint64;
using ll = long long;

namespace {
	namespace ex = atl::experimental;
	using atl::detail::splay_layout_interleaved;
	using atl::detail::splay_layout_split;
	using entries = std::vector<std::pair<int, ll>>;

	/** @brief Adapt a standard map, which uses its own balanced-tree storage. */
	struct map_by_std_map {
		static constexpr char const *name = "by std map";
		using map = std::map<int, ll>;

		static map build(entries const &e) {
			map m;
			for (auto const &[k, v] : e) m.emplace_hint(m.end(), k, v);
			return m;
		}

		static map make_empty() { return {}; }

		static ll find(map &m, int k) {
			auto const it = m.find(k);
			return it == m.end() ? 0 : it->second;
		}

		static void assign(map &m, int k, ll v) { m.insert_or_assign(k, v); }

		static bool insert(map &m, int k, ll v) { return m.emplace(k, v).second; }

		static bool erase(map &m, int k) { return m.erase(k) != 0; }

		static int next(map &m, int k) {
			auto const it = m.lower_bound(k);
			return it == m.end() ? -1 : it->first;
		}
	};

	/** @brief A fold-tier keyed map with the treap_map interface this benchmark drives. */
	template<typename Map>
	concept benchmarked_map = std::movable<Map> && requires(Map &m, int k, ll v) {
		{ m.find(k) } -> std::convertible_to<ll const *>;
		m.insert_or_assign(k, v);
		{ m.insert(k, v) } -> std::convertible_to<bool>;
		{ m.erase(k) } -> std::convertible_to<bool>;
		{ m.next(k)->key } -> std::convertible_to<int>;
	};

	/** @brief treap_map, splay_map and the splay contenders share one interface. */
	template<benchmarked_map Map>
	struct tree_kind_base {
		using map = Map;

		static map build(entries const &e) { return map(atl::sorted_unique, e, 0LL, std::plus<>{}); }

		static map make_empty() { return map(0LL, std::plus<>{}); }

		static ll find(map &m, int k) {
			ll const *v = m.find(k);
			return v != nullptr ? *v : 0;
		}

		static void assign(map &m, int k, ll v) { m.insert_or_assign(k, v); }

		static bool insert(map &m, int k, ll v) { return m.insert(k, v); }

		static bool erase(map &m, int k) { return m.erase(k); }

		static int next(map &m, int k) {
			auto const e = m.next(k);
			return e ? e->key : -1;
		}
	};

	/** @brief Query by random-priority balancing rather than splaying. */
	struct map_by_treap_map : tree_kind_base<atl::treap_map<int, ll, std::ranges::less, std::plus<>>> {
		static constexpr char const *name = "by treap map";
	};

	/** @brief Use the library map directly, with its default layout. */
	struct map_api_adapter : tree_kind_base<atl::splay_map<int, ll, std::ranges::less, std::plus<>>> {
		static constexpr char const *name = "library";
	};

	/** @brief Change descent shape or link/payload placement while sharing the map operations. */
	template<typename Engine, atl::detail::splay_map_layout Layout>
	struct map_by_strategy : tree_kind_base<ex::splay_map_by_strategy<Engine, Layout, int, ll, std::ranges::less, std::plus<>>> {
		static constexpr char const *name = [] {
			if constexpr (std::same_as<Engine, ex::splay_by_bottom_up>)
				return std::same_as<Layout, splay_layout_split> ? "by bottom up split" : "by bottom up interleaved";
			else
				return std::same_as<Layout, splay_layout_split> ? "by top down split" : "by top down interleaved";
		}();
	};

	using kinds = std::tuple<map_by_std_map, map_by_treap_map, map_api_adapter, map_by_strategy<ex::splay_by_top_down, splay_layout_interleaved>,
							 map_by_strategy<ex::splay_by_bottom_up, splay_layout_split>,
							 map_by_strategy<ex::splay_by_bottom_up, splay_layout_interleaved>>;

	template<typename Kind>
	concept map_workload_kind = requires(entries const &initial, typename Kind::map &map) {
		{ Kind::name } -> std::convertible_to<char const *>;
		{ Kind::build(initial) } -> std::same_as<typename Kind::map>;
		{ Kind::make_empty() } -> std::same_as<typename Kind::map>;
		{ Kind::find(map, 0) } -> std::convertible_to<ll>;
		{ Kind::next(map, 0) } -> std::convertible_to<int>;
		Kind::assign(map, 0, 0LL);
		{ Kind::insert(map, 0, 0LL) } -> std::convertible_to<bool>;
		{ Kind::erase(map, 0) } -> std::convertible_to<bool>;
	};

	template<typename Fn>
		requires([]<typename... Kind>(std::tuple<Kind...> *) { return (std::invocable<Fn &, Kind> && ...); }(static_cast<kinds *>(nullptr)))
	void for_each_kind(Fn &&fn) {
		std::apply([&]<typename... Kind>(Kind... kind)
					   requires(std::invocable<Fn &, Kind> && ...)
				   { (std::invoke(fn, kind), ...); },
				   kinds{});
	}

	std::string label(char const *kind, char const *shape, std::size_t n) {
		return std::string(kind) + " -- " + shape + ", n=" + (n == 200'000 ? "2e5" : "1e6");
	}

	/** @brief The n even keys 0, 2, ..., 2n - 2 with small values. */
	entries even_keys(std::size_t n) {
		entries e(n);
		for (std::size_t i = 0; i < n; ++i) e[i] = {int(2 * i), ll(i % 1000)};
		return e;
	}

	/**
	 * @brief @p m stored keys drawn by Zipf(@p s) over n ranks, each rank
	 *        mapped to a key through one random permutation (so hot keys are
	 *        spread over the key space, not clustered at the left end).
	 */
	std::vector<int> zipf_keys(std::size_t n, std::size_t m, double s, uint64 seed) {
		std::vector<double> cdf(n);
		double total = 0;
		for (std::size_t r = 0; r < n; ++r) cdf[r] = total += 1.0 / std::pow(double(r + 1), s);
		std::vector<int> key_of_rank(n);
		for (std::size_t r = 0; r < n; ++r) key_of_rank[r] = int(r);
		std::mt19937_64 rng(seed);
		std::ranges::shuffle(key_of_rank, rng);
		std::uniform_real_distribution<double> uniform(0, total);
		std::vector<int> keys(m);
		for (int &k : keys) {
			auto const rank = std::size_t(std::ranges::upper_bound(cdf, uniform(rng)) - cdf.begin());
			k = 2 * key_of_rank[(std::min)(rank, n - 1)];
		}
		return keys;
	}

	constexpr std::size_t sizes[] = {200'000, 1'000'000};
}

TEST_CASE("splay_map benchmarks: uniform random insert/find/erase", "[splay_map]") {
	for (std::size_t const n : sizes) {
		entries const initial = even_keys(n);
		std::mt19937_64 rng(1);
		std::vector<uint32> op(n);
		std::vector<int> key(n);
		for (std::size_t i = 0; i < n; ++i) {
			op[i] = uint32(rng() % 3);
			key[i] = int(rng() % (2 * n));
		}
		auto const compute_checksum = [&]<map_workload_kind Kind>(Kind) {
			auto m = Kind::build(initial);
			ll acc = 0;
			for (std::size_t i = 0; i < n; ++i) {
				if (op[i] == 0)
					Kind::assign(m, key[i], ll(i));
				else if (op[i] == 1)
					acc += Kind::find(m, key[i]);
				else
					acc += ll(Kind::erase(m, key[i]));
			}
			return acc;
		};
		auto const expected = compute_checksum(map_by_std_map{});
		for_each_kind([&]<map_workload_kind Kind>(Kind) {
			REQUIRE(compute_checksum(Kind{}) == expected);
			BENCHMARK(label(Kind::name, "uniform insert/find/erase", n)) { return compute_checksum(Kind{}); };
		});
	}
}

TEST_CASE("splay_map benchmarks: Zipf finds", "[splay_map]") {
	for (std::size_t const n : sizes) {
		entries const initial = even_keys(n);
		for (double const s : {1.0, 1.5}) {
			std::vector<int> const keys = zipf_keys(n, n, s, 2);
			char const *shape = s == 1.0 ? "Zipf s=1.0 finds" : "Zipf s=1.5 finds";
			auto const compute_checksum = [&]<map_workload_kind Kind>(Kind) {
				auto m = Kind::build(initial);
				ll acc = 0;
				for (int const k : keys) acc += Kind::find(m, k);
				return acc;
			};
			auto const expected = compute_checksum(map_by_std_map{});
			for_each_kind([&]<map_workload_kind Kind>(Kind) {
				REQUIRE(compute_checksum(Kind{}) == expected);
				BENCHMARK(label(Kind::name, shape, n)) { return compute_checksum(Kind{}); };
			});
		}
	}
}

TEST_CASE("splay_map benchmarks: sequential next chains", "[splay_map]") {
	for (std::size_t const n : sizes) {
		entries const initial = even_keys(n);
		constexpr std::size_t chain = 1000;
		std::mt19937_64 rng(3);
		std::vector<int> starts(n / chain);
		for (int &k : starts) k = int(rng() % (2 * n));
		auto const compute_checksum = [&]<map_workload_kind Kind>(Kind) {
			auto m = Kind::build(initial);
			ll acc = 0;
			for (int const start : starts) {
				int k = start;
				for (std::size_t step = 0; step < chain; ++step) {
					int const got = Kind::next(m, k);
					acc += got;
					k = got < 0 ? 0 : got + 1;
				}
			}
			return acc;
		};
		auto const expected = compute_checksum(map_by_std_map{});
		for_each_kind([&]<map_workload_kind Kind>(Kind) {
			REQUIRE(compute_checksum(Kind{}) == expected);
			BENCHMARK(label(Kind::name, "sequential next chains (1e3 long)", n)) { return compute_checksum(Kind{}); };
		});
	}
}

TEST_CASE("splay_map benchmarks: sliding working set", "[splay_map]") {
	for (std::size_t const n : sizes) {
		entries const initial = even_keys(n);
		constexpr std::size_t window = 1000;
		std::mt19937_64 rng(4);
		std::vector<int> keys(n);
		// the window of stored keys [i, i + 1e3) slides by one rank per access
		for (std::size_t i = 0; i < n; ++i) keys[i] = int(2 * ((i + rng() % window) % n));
		auto const compute_checksum = [&]<map_workload_kind Kind>(Kind) {
			auto m = Kind::build(initial);
			ll acc = 0;
			for (int const k : keys) acc += Kind::find(m, k);
			return acc;
		};
		auto const expected = compute_checksum(map_by_std_map{});
		for_each_kind([&]<map_workload_kind Kind>(Kind) {
			REQUIRE(compute_checksum(Kind{}) == expected);
			BENCHMARK(label(Kind::name, "working set (window 1e3) finds", n)) { return compute_checksum(Kind{}); };
		});
	}
}

TEST_CASE("splay_map benchmarks: sorted-order inserts", "[splay_map]") {
	for (std::size_t const n : sizes) {
		auto const compute_checksum = [&]<map_workload_kind Kind>(Kind) {
			auto m = Kind::make_empty();
			ll acc = 0;
			for (std::size_t i = 0; i < n; ++i) acc += ll(Kind::insert(m, int(i), ll(i)));
			return acc;
		};
		auto const expected = compute_checksum(map_by_std_map{});
		for_each_kind([&]<map_workload_kind Kind>(Kind) {
			REQUIRE(compute_checksum(Kind{}) == expected);
			BENCHMARK(label(Kind::name, "sorted-order inserts", n)) { return compute_checksum(Kind{}); };
		});
	}
}
