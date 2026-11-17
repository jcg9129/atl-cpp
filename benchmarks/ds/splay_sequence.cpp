/**
 * @file
 * @brief Sequence throughput at 200,000 and 1,000,000 elements: random surgery,
 * random/sequential reads, cursor edits, hot ranges, reversal and affine tags.
 * The library splays top-down in interleaved storage; contenders use a treap,
 * bottom-up/repush/semi splaying or split/columnar storage. Handle splays compare
 * ancestor stacks, store buffers, pointer reversal and reversal-parity walks.
 * Construction is timed. Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions
 * off, 2026-10-24, -O2 / -O2 -march=native, n=1e6: library random surgery
 * 1545/1543 ms, sequential reads 73.4/75.7 ms. Full tables: docs/splay-sequence.md.
 */
#include <atl/ds/splay_engine.hpp>
#include <atl/ds/splay_sequence.hpp>
#include <atl/ds/treap_sequence.hpp>
#include <atl/experimental/ds/splay_engine_variants.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <numeric>
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
	namespace ex = atl::experimental;
	using atl::detail::splay_layout_columnar;
	using atl::detail::splay_layout_interleaved;
	using atl::detail::splay_layout_split;

	/** @brief Query and mutate by random-priority balancing. */
	struct sequence_by_treap_sequence {
		static constexpr char const *name = "by treap sequence";
		template<std::semiregular T, typename Op = void, typename F = void, typename Mapping = void, typename Compose = void>
		using sequence = atl::treap_sequence<T, Op, F, Mapping, Compose>;
		template<std::semiregular T, typename Op = void, typename F = void, typename Mapping = void, typename Compose = void>
		using reversible = atl::reversible_treap_sequence<T, Op, F, Mapping, Compose>;
	};

	/** @brief Use the library sequence directly with its default layout. */
	struct sequence_api_adapter {
		static constexpr char const *name = "library";
		template<std::semiregular T, typename Op = void, typename F = void, typename Mapping = void, typename Compose = void>
		using sequence = atl::splay_sequence<T, Op, F, Mapping, Compose>;
		template<std::semiregular T, typename Op = void, typename F = void, typename Mapping = void, typename Compose = void>
		using reversible = atl::reversible_splay_sequence<T, Op, F, Mapping, Compose>;
	};

	/** @brief Change descent shape or field layout while sharing the sequence operations. */
	template<ex::splay_shape Shape, atl::detail::splay_layout Layout>
	struct sequence_by_strategy {
		static constexpr char const *name = [] {
			if constexpr (std::same_as<Layout, splay_layout_interleaved>)
				return Shape == ex::splay_shape::bottom_up_repush ? "by bottom up repush interleaved"
					 : Shape == ex::splay_shape::top_down		  ? "by top down interleaved"
					 : Shape == ex::splay_shape::semi			  ? "by semi splay interleaved"
																  : "by bottom up interleaved";
			else if constexpr (std::same_as<Layout, splay_layout_split>)
				return Shape == ex::splay_shape::top_down ? "by top down split" : "by bottom up split";
			else
				return Shape == ex::splay_shape::top_down ? "by columnar layout" : "by columnar layout";
		}();
		using engine = ex::splay_engine_by_strategy<Shape, ex::splay_path_push::ancestor_stack>;
		template<std::semiregular T, typename Op = void, typename F = void, typename Mapping = void, typename Compose = void>
		using sequence = ex::splay_sequence_by_strategy<engine, Layout, T, Op, F, Mapping, Compose>;
		template<std::semiregular T, typename Op = void, typename F = void, typename Mapping = void, typename Compose = void>
		using reversible = ex::splay_sequence_by_strategy<engine, Layout, T, Op, F, Mapping, Compose, true>;
	};

	using kinds = std::tuple<sequence_by_treap_sequence, sequence_api_adapter, sequence_by_strategy<ex::splay_shape::bottom_up_repush, splay_layout_interleaved>,
							 sequence_by_strategy<ex::splay_shape::bottom_up, splay_layout_interleaved>,
							 sequence_by_strategy<ex::splay_shape::semi, splay_layout_interleaved>,
							 sequence_by_strategy<ex::splay_shape::top_down, splay_layout_columnar>,
							 sequence_by_strategy<ex::splay_shape::top_down, splay_layout_split>,
							 sequence_by_strategy<ex::splay_shape::bottom_up, splay_layout_split>>;

	template<typename Kind>
	concept sequence_workload_kind = requires {
		{ Kind::name } -> std::convertible_to<char const *>;
		typename Kind::template sequence<int>;
		typename Kind::template sequence<ll, std::plus<>>;
		typename Kind::template reversible<ll, std::plus<>>;
	};

	template<typename Fn>
		requires([]<typename... Kind>(std::tuple<Kind...> *) { return (std::invocable<Fn &, Kind> && ...); }(static_cast<kinds *>(nullptr)))
	void for_each_kind(Fn &&fn) {
		std::apply([&]<typename... Kind>(Kind... kind)
					   requires(std::invocable<Fn &, Kind> && ...)
				   { (std::invoke(fn, kind), ...); },
				   kinds{});
	}

	/// Raw draws per operation; reduced against the live size at run time.
	struct workload {
		std::vector<uint32> op, x, y;
		std::vector<ll> value;

		workload(std::size_t m, uint64 seed) : op(m), x(m), y(m), value(m) {
			std::mt19937_64 rng(seed);
			for (std::size_t i = 0; i < m; ++i) {
				op[i] = uint32(rng());
				x[i] = uint32(rng());
				y[i] = uint32(rng());
				value[i] = ll(rng() % 1000);
			}
		}
	};

	/// `[l, r)` from two raw draws, within `[0, size]`.
	std::pair<std::size_t, std::size_t> range_of(uint32 a, uint32 b, std::size_t size) {
		std::size_t l = a % (size + 1), r = b % (size + 1);
		if (l > r) std::swap(l, r);
		return {l, r};
	}

	std::string label(char const *kind, char const *shape, std::size_t n) {
		return std::string(kind) + " -- " + shape + ", n=" + (n == 200'000 ? "2e5" : "1e6");
	}

	constexpr std::size_t sizes[] = {200'000, 1'000'000};
}

TEST_CASE("splay_sequence benchmarks: uniform random insert/erase/fold", "[splay_sequence]") {
	for (std::size_t const n : sizes) {
		workload const w(n, 1);
		auto const compute_checksum = [&]<sequence_workload_kind Kind>(Kind) {
			typename Kind::template sequence<ll, std::plus<>> t(std::vector<ll>(n, 1), ll(0), std::plus<>{});
			ll acc = 0;
			for (std::size_t i = 0; i < n; ++i) {
				std::size_t const size = t.size();
				switch (w.op[i] % 3) {
				case 0:
					t.insert(w.x[i] % (size + 1), w.value[i]);
					break;
				case 1:
					t.erase(w.x[i] % size);
					break;
				default: {
					auto const [l, r] = range_of(w.x[i], w.y[i], size);
					acc += t.fold(l, r);
				}
				}
			}
			return acc;
		};
		auto const expected = compute_checksum(sequence_by_treap_sequence{});
		for_each_kind([&]<sequence_workload_kind Kind>(Kind) {
			REQUIRE(compute_checksum(Kind{}) == expected);
			BENCHMARK(label(Kind::name, "uniform insert/erase/fold", n)) { return compute_checksum(Kind{}); };
		});
	}
}

TEST_CASE("splay_sequence benchmarks: random at", "[splay_sequence]") {
	for (std::size_t const n : sizes) {
		workload const w(n, 2);
		auto const compute_checksum = [&]<sequence_workload_kind Kind>(Kind) {
			typename Kind::template sequence<int> t(std::vector<int>(n, 1));
			ll acc = 0;
			for (std::size_t i = 0; i < n; ++i) acc += t.at(w.x[i] % n);
			return acc;
		};
		auto const expected = compute_checksum(sequence_by_treap_sequence{});
		for_each_kind([&]<sequence_workload_kind Kind>(Kind) {
			REQUIRE(compute_checksum(Kind{}) == expected);
			BENCHMARK(label(Kind::name, "random at", n)) { return compute_checksum(Kind{}); };
		});
	}
}

TEST_CASE("splay_sequence benchmarks: sequential scan by index", "[splay_sequence]") {
	for (std::size_t const n : sizes) {
		auto const compute_checksum = [&]<sequence_workload_kind Kind>(Kind) {
			typename Kind::template sequence<int> t(std::vector<int>(n, 1));
			ll acc = 0;
			for (std::size_t i = 0; i < n; ++i) acc += t.at(i);
			return acc;
		};
		auto const expected = compute_checksum(sequence_by_treap_sequence{});
		for_each_kind([&]<sequence_workload_kind Kind>(Kind) {
			REQUIRE(compute_checksum(Kind{}) == expected);
			BENCHMARK(label(Kind::name, "sequential at(0..n-1)", n)) { return compute_checksum(Kind{}); };
		});
	}
}

TEST_CASE("splay_sequence benchmarks: cursor editing within +-16", "[splay_sequence]") {
	for (std::size_t const n : sizes) {
		workload const w(n, 3);
		auto const compute_checksum = [&]<sequence_workload_kind Kind>(Kind) {
			typename Kind::template sequence<ll, std::plus<>> t(std::vector<ll>(n, 1), ll(0), std::plus<>{});
			ll acc = 0;
			std::size_t cursor = n / 2;
			for (std::size_t i = 0; i < n; ++i) {
				std::size_t const size = t.size();
				cursor = (std::min)(size - 1, (cursor + size + (w.x[i] % 33) - 16) % size);
				switch (w.op[i] % 4) {
				case 0:
					t.insert(cursor, w.value[i]);
					break;
				case 1:
					t.erase(cursor);
					break;
				case 2:
					acc += t.at((std::min)(cursor, size - 1));
					break;
				default:
					acc += t.fold(cursor, (std::min)(size, cursor + 16));
				}
			}
			return acc;
		};
		auto const expected = compute_checksum(sequence_by_treap_sequence{});
		for_each_kind([&]<sequence_workload_kind Kind>(Kind) {
			REQUIRE(compute_checksum(Kind{}) == expected);
			BENCHMARK(label(Kind::name, "cursor +-16 insert/erase/at/fold", n)) { return compute_checksum(Kind{}); };
		});
	}
}

TEST_CASE("splay_sequence benchmarks: hot range (90% of ops on 1% of positions)", "[splay_sequence]") {
	for (std::size_t const n : sizes) {
		workload const w(n, 4);
		std::size_t const hot = n / 100, base = n / 3;
		auto const compute_checksum = [&]<sequence_workload_kind Kind>(Kind) {
			typename Kind::template sequence<ll, std::plus<>> t(std::vector<ll>(n, 1), ll(0), std::plus<>{});
			ll acc = 0;
			for (std::size_t i = 0; i < n; ++i) {
				bool const is_hot = w.op[i] % 10 != 0;
				std::size_t const span = is_hot ? hot : n, start = is_hot ? base : 0;
				std::size_t const p = start + w.x[i] % span, q = start + w.y[i] % span;
				switch ((w.op[i] >> 8) % 3) {
				case 0:
					acc += t.at(p);
					break;
				case 1:
					t.set(p, w.value[i]);
					break;
				default:
					acc += t.fold((std::min)(p, q), (std::max)(p, q) + 1);
				}
			}
			return acc;
		};
		auto const expected = compute_checksum(sequence_by_treap_sequence{});
		for_each_kind([&]<sequence_workload_kind Kind>(Kind) {
			REQUIRE(compute_checksum(Kind{}) == expected);
			BENCHMARK(label(Kind::name, "hot range 90%/1% at/set/fold", n)) { return compute_checksum(Kind{}); };
		});
	}
}

TEST_CASE("splay_sequence benchmarks: reverse-heavy", "[splay_sequence]") {
	for_each_kind([]<sequence_workload_kind Kind>(Kind) {
		std::vector<ll> expected(127);
		std::iota(expected.begin(), expected.end(), 1LL);
		typename Kind::template reversible<ll, std::plus<>> tree(expected, 0LL, std::plus<>{});
		workload const draws(512, 5);
		for (std::size_t i = 0; i < draws.op.size(); ++i) {
			auto const [l, r] = range_of(draws.x[i], draws.y[i], expected.size());
			if (draws.op[i] % 2) {
				tree.reverse(l, r);
				std::ranges::reverse(expected.begin() + std::ptrdiff_t(l), expected.begin() + std::ptrdiff_t(r));
			}
			REQUIRE(tree.fold(l, r) == std::accumulate(expected.begin() + std::ptrdiff_t(l), expected.begin() + std::ptrdiff_t(r), 0LL));
		}
		std::vector<ll> actual;
		tree.for_each([&](ll value) { actual.push_back(value); });
		REQUIRE(actual == expected);
	});
	for (std::size_t const n : sizes) {
		workload const w(n, 5);
		auto const compute_checksum = [&]<sequence_workload_kind Kind>(Kind) {
			typename Kind::template reversible<ll, std::plus<>> t(std::vector<ll>(n, 1), ll(0), std::plus<>{});
			ll acc = 0;
			for (std::size_t i = 0; i < n; ++i) {
				auto const [l, r] = range_of(w.x[i], w.y[i], n);
				if (w.op[i] % 2)
					t.reverse(l, r);
				else
					acc += t.fold(l, r);
			}
			return acc;
		};
		auto const expected = compute_checksum(sequence_by_treap_sequence{});
		for_each_kind([&]<sequence_workload_kind Kind>(Kind) {
			REQUIRE(compute_checksum(Kind{}) == expected);
			BENCHMARK(label(Kind::name, "reverse/fold 50/50", n)) { return compute_checksum(Kind{}); };
		});
	}
}

TEST_CASE("splay_sequence benchmarks: range affine + range sum (batch dynamic sequence)", "[splay_sequence]") {
	constexpr ll mod = 998'244'353;

	struct node {
		ll s, w;
	};

	struct affine {
		ll b, c;
	};

	auto const op = [](node const &a, node const &b) { return node{(a.s + b.s) % mod, a.w + b.w}; };
	auto const mapping = [](affine const &f, node const &a) { return node{(f.b * a.s + f.c * a.w) % mod, a.w}; };
	auto const compose = [](affine const &f, affine const &g) { return affine{f.b * g.b % mod, (f.b * g.c + f.c) % mod}; };
	using op_t = std::remove_const_t<decltype(op)>;
	using mapping_t = std::remove_const_t<decltype(mapping)>;
	using compose_t = std::remove_const_t<decltype(compose)>;
	for_each_kind([&]<sequence_workload_kind Kind>(Kind) {
		std::vector<node> expected;
		for (ll value = 1; value <= 127; ++value) expected.push_back({value, 1});
		typename Kind::template reversible<node, op_t, affine, mapping_t, compose_t> tree(expected, node{0, 0}, affine{1, 0}, op, mapping, compose);
		workload const draws(512, 6);
		for (std::size_t i = 0; i < draws.op.size(); ++i) {
			auto const [l, r] = range_of(draws.x[i], draws.y[i], expected.size());
			switch (draws.op[i] % 5) {
			case 0: {
				std::size_t const pos = draws.x[i] % (expected.size() + 1);
				node const value{draws.value[i], 1};
				tree.insert(pos, value);
				expected.insert(expected.begin() + std::ptrdiff_t(pos), value);
				break;
			}
			case 1: {
				std::size_t const pos = draws.x[i] % expected.size();
				tree.erase(pos);
				expected.erase(expected.begin() + std::ptrdiff_t(pos));
				break;
			}
			case 2:
				tree.reverse(l, r);
				std::ranges::reverse(expected.begin() + std::ptrdiff_t(l), expected.begin() + std::ptrdiff_t(r));
				break;
			case 3: {
				affine const tag{1 + draws.value[i], draws.value[i]};
				tree.apply(l, r, tag);
				for (std::size_t j = l; j < r; ++j) expected[j] = mapping(tag, expected[j]);
				break;
			}
			default:
				break;
			}
			auto const [ql, qr] = range_of(draws.y[i], draws.x[i], expected.size());
			node sum{0, 0};
			for (std::size_t j = ql; j < qr; ++j) sum = op(sum, expected[j]);
			REQUIRE(tree.fold(ql, qr).s == sum.s);
		}
		std::size_t pos = 0;
		tree.for_each([&](node const &value) {
			REQUIRE(pos < expected.size());
			REQUIRE(value.s == expected[pos].s);
			REQUIRE(value.w == expected[pos].w);
			++pos;
		});
		REQUIRE(pos == expected.size());
	});
	for (std::size_t const n : sizes) {
		workload const w(n, 6);
		auto const compute_checksum = [&]<sequence_workload_kind Kind>(Kind) {
			typename Kind::template reversible<node, op_t, affine, mapping_t, compose_t> t(
				std::vector<node>(n, node{1, 1}), node{0, 0}, affine{1, 0}, op, mapping, compose);
			ll acc = 0;
			for (std::size_t i = 0; i < n; ++i) {
				std::size_t const size = t.size();
				auto const [l, r] = range_of(w.x[i], w.y[i], size);
				switch (w.op[i] % 5) {
				case 0:
					t.insert(w.x[i] % (size + 1), node{w.value[i], 1});
					break;
				case 1:
					t.erase(w.x[i] % size);
					break;
				case 2:
					t.reverse(l, r);
					break;
				case 3:
					t.apply(l, r, affine{1 + w.value[i], w.value[i]});
					break;
				default:
					acc += t.fold(l, r).s;
				}
			}
			return acc;
		};
		auto const expected = compute_checksum(sequence_by_treap_sequence{});
		for_each_kind([&]<sequence_workload_kind Kind>(Kind) {
			REQUIRE(compute_checksum(Kind{}) == expected);
			BENCHMARK(label(Kind::name, "insert/erase/reverse/apply/fold", n)) { return compute_checksum(Kind{}); };
		});
	}
}

// Handle splays (the link-cut access pattern): splay a random node, then mark
// everything before it reversed. Only here do the push strategies differ.
TEST_CASE("splay_sequence benchmarks: engine handle splays + reversal", "[splay_sequence]") {
	using rope = ex::reversible_fold_rope<ll, std::plus<>>;
	auto const compute_handle_checksum = []<bool ShouldValidate = false, typename Join, typename Splay>(Join join, Splay splay, std::size_t n, workload const &w)
		requires std::is_invocable_r_v<uint32, Join &, rope &, uint32, uint32> && std::invocable<Splay &, rope &, uint32>
	{
		rope s(0, std::plus<>{});
		s.reserve(n);
		std::vector<uint32> ids(n);
		for (std::size_t i = 0; i < n; ++i) ids[i] = s.make(ll(i));
		std::vector<uint32> order = ids; // a balanced start: bottom-up halving joins
		while (order.size() > 1) {
			std::vector<uint32> next;
			for (std::size_t i = 0; i + 1 < order.size(); i += 2) next.push_back(std::invoke(join, s, order[i], order[i + 1]));
			if (order.size() % 2) next.push_back(order.back());
			order = std::move(next);
		}
		std::vector<ll> expected_order;
		if constexpr (ShouldValidate) {
			expected_order.resize(n);
			std::iota(expected_order.begin(), expected_order.end(), 0LL);
		}
		ll acc = 0;
		for (std::size_t i = 0; i < n; ++i) {
			uint32 const h = ids[w.x[i] % n];
			std::invoke(splay, s, h);
			if constexpr (ShouldValidate) {
				auto const pos = std::ranges::find(expected_order, ll(w.x[i] % n));
				REQUIRE(pos != expected_order.end());
				REQUIRE(s.aggregate(s.child(h, 0)) == std::accumulate(expected_order.begin(), pos, 0LL));
				std::ranges::reverse(expected_order.begin(), pos);
			}
			s.flip(s.child(h, 0));
			acc += s.aggregate(h);
		}
		if constexpr (ShouldValidate) {
			std::vector<ll> actual;
			uint32 const root = ids[w.x[n - 1] % n];
			atl::splay_engine::for_each(s, root, [&](uint32 x) { actual.push_back(s.value(x)); });
			REQUIRE(actual == expected_order);
		}
		return acc;
	};
	using ex::splay_path_push;
	using ex::splay_shape;
	for (std::size_t const n : sizes) {
		workload const w(n, 7);
		auto const run_library = [&]<bool ShouldValidate = false>() {
			return compute_handle_checksum.template operator()<ShouldValidate>([](rope &store, uint32 a, uint32 b) { return atl::splay_engine::join(store, a, b); }, [](rope &store, uint32 x) { atl::splay_engine::splay(store, x); }, ShouldValidate ? 127 : n, w);
		};
		auto const run_by_store_buffer = [&]<bool ShouldValidate = false>() {
			using engine = ex::splay_engine_by_strategy<splay_shape::bottom_up, splay_path_push::store_buffer>;
			return compute_handle_checksum.template operator()<ShouldValidate>([](rope &store, uint32 a, uint32 b) { return engine::join(store, a, b); }, [](rope &store, uint32 x) { engine::splay(store, x); }, ShouldValidate ? 127 : n, w);
		};
		auto const run_by_pointer_reversal = [&]<bool ShouldValidate = false>() {
			using engine = ex::splay_engine_by_strategy<splay_shape::bottom_up, splay_path_push::pointer_reversal>;
			return compute_handle_checksum.template operator()<ShouldValidate>([](rope &store, uint32 a, uint32 b) { return engine::join(store, a, b); }, [](rope &store, uint32 x) { engine::splay(store, x); }, ShouldValidate ? 127 : n, w);
		};
		auto const run_by_parity_walk = [&]<bool ShouldValidate = false>() {
			using engine = ex::splay_engine_by_strategy<splay_shape::bottom_up, splay_path_push::parity_walk>;
			return compute_handle_checksum.template operator()<ShouldValidate>([](rope &store, uint32 a, uint32 b) { return engine::join(store, a, b); }, [](rope &store, uint32 x) { engine::splay(store, x); }, ShouldValidate ? 127 : n, w);
		};
		run_library.template operator()<true>();
		run_by_store_buffer.template operator()<true>();
		run_by_pointer_reversal.template operator()<true>();
		run_by_parity_walk.template operator()<true>();
		auto const expected = run_library();
		REQUIRE(run_by_store_buffer() == expected);
		REQUIRE(run_by_pointer_reversal() == expected);
		REQUIRE(run_by_parity_walk() == expected);
		BENCHMARK(label("library", "handle splay + reverse prefix", n)) { return run_library(); };
		BENCHMARK(label("by store buffer", "handle splay + reverse prefix", n)) { return run_by_store_buffer(); };
		BENCHMARK(label("by pointer reversal", "handle splay + reverse prefix", n)) { return run_by_pointer_reversal(); };
		BENCHMARK(label("by parity walk", "handle splay + reverse prefix", n)) { return run_by_parity_walk(); };
	}
}
