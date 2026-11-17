/**
 * @file
 * @brief Races persistent versus mutable range actions, singleton reads by direct descent
 * versus singleton fold, permanent versus settled commuting tags, and partial versus
 * whole-node copies. Measures complete-script throughput on affine/add/wide-add
 * streams, linear and branching versions, n=1e5/2e5, and 1e6 point reads.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-11-01,
 * -O2 / -O2 -march=native (also -g -DNDEBUG -std=c++23): branching add n=1e5
 * whole-node 472 / 492 ms versus partial-copy 471 / 474.
 * See docs/persistent-lazy-segment-tree.md for tables.
 */

#include <atl/ds/lazy_segment_tree.hpp>
#include <atl/ds/persistent_lazy_segment_tree.hpp>
#include <atl/experimental/ds/persistent_lazy_segment_tree_variants.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <format>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
	/** @brief Time a complete workload through a direct, constrained callable. */
	template<typename Run>
		requires std::invocable<Run &>
	void measure(std::string const &label, Run run) {
		BENCHMARK(std::string(label)) { return run(); };
	}
}

using ll = long long;

TEST_CASE("persistent_lazy_segment_tree benchmarks", "[persistent_lazy_segment_tree]") {
	constexpr int n = 100'000, m = 200'000;
	constexpr ll mod = 998'244'353;
	std::mt19937 rng(1);
	std::vector<int> lo(m), hi(m);
	std::vector<ll> fb(m), fc(m);
	std::vector<char> is_apply(m);
	for (int i = 0; i < m; ++i) {
		int a = int(rng() % n), b = int(rng() % n);
		if (a > b) std::swap(a, b);
		lo[i] = a;
		hi[i] = b + 1;
		fb[i] = ll(1 + rng() % (mod - 1));
		fc[i] = ll(rng() % mod);
		is_apply[i] = char(rng() % 2);
	}

	struct node {
		ll sum, width;
	};

	struct affine {
		ll b, c;
	};

	auto const op = [](node const &x, node const &y) { return node{(x.sum + y.sum) % mod, x.width + y.width}; };
	auto const map = [](affine const &f, node const &s) {
		return node{(f.b * s.sum + f.c * s.width) % mod, s.width};
	};
	auto const compose = [](affine const &f, affine const &g) {
		return affine{f.b * g.b % mod, (f.b * g.c + f.c) % mod};
	};

	auto const run_library = [&] {
		atl::persistent_lazy_segment_tree cur(std::vector<node>(n, node{1, 1}), node{0, 0}, affine{1, 0}, op, map,
											  compose);
		cur.reserve(m / 2);
		ll acc = 0;
		for (int i = 0; i < m; ++i)
			if (is_apply[i])
				cur = cur.apply(std::size_t(lo[i]), std::size_t(hi[i]), affine{fb[i], fc[i]});
			else
				acc += cur.fold(std::size_t(lo[i]), std::size_t(hi[i])).sum;
		return acc;
	};
	auto const apply_by_mutable_lazy_tree = [&] {
		atl::lazy_segment_tree st(std::vector<node>(n, node{1, 1}), node{0, 0}, affine{1, 0}, op, map, compose);
		ll acc = 0;
		for (int i = 0; i < m; ++i)
			if (is_apply[i])
				st.apply(std::size_t(lo[i]), std::size_t(hi[i]), affine{fb[i], fc[i]});
			else
				acc += st.fold(std::size_t(lo[i]), std::size_t(hi[i])).sum;
		return acc;
	};
	REQUIRE(run_library() == apply_by_mutable_lazy_tree());
	measure("library persistent -- 2e5 mixed ops", run_library);
	measure("by mutable lazy tree -- 2e5 mixed ops", apply_by_mutable_lazy_tree);

	atl::persistent_lazy_segment_tree ver(std::vector<node>(n, node{1, 1}), node{0, 0}, affine{1, 0}, op, map, compose);
	for (int i = 0; i < m; ++i)
		if (is_apply[i]) ver = ver.apply(std::size_t(lo[i]), std::size_t(hi[i]), affine{fb[i], fc[i]});
	std::vector<std::size_t> reads(1'000'000);
	for (auto &r : reads) r = rng() % n;
	for (std::size_t k = 0; k < 10000; ++k)
		if (ver.at(reads[k]).sum != ver.fold(reads[k], reads[k] + 1).sum) FAIL("at disagrees with single-element fold");
	measure("1e6 at() after 1e5 applies, affine library direct descent", [&] {
		ll acc = 0;
		for (auto r : reads) acc += ver.at(r).sum;
		return acc;
	});
	measure("1e6 at() after 1e5 applies, affine (by singleton fold)", [&] {
		ll acc = 0;
		for (auto r : reads) acc += ver.fold(r, r + 1).sum;
		return acc;
	});
}

TEST_CASE("persistent_lazy_segment_tree commutative mode", "[persistent_lazy_segment_tree][commutative]") {
	constexpr int m = 200'000;

	struct node {
		ll sum, width;
	};

	auto op = [](node const &x, node const &y) { return node{x.sum + y.sum, x.width + y.width}; };
	auto map = [](ll f, node const &s) { return node{s.sum + f * s.width, s.width}; };
	auto add = [](ll f, ll g) { return f + g; };
	using ordered = atl::persistent_lazy_segment_tree<node, ll, decltype(op), decltype(map), decltype(add)>;
	using commuting =
		atl::persistent_lazy_segment_tree<node, ll, decltype(op), decltype(map), decltype(add), atl::commutative_tags_t>;
	for (int n : {100'000, 200'000})
		for (bool const is_branching : {false, true}) {
			std::mt19937 rng(3);
			std::vector<int> lo(m), hi(m), base(m);
			std::vector<ll> f(m);
			std::vector<char> is_apply(m);
			for (int i = 0; i < m; ++i) {
				int a = int(rng() % n), b = int(rng() % n);
				if (a > b) std::swap(a, b);
				lo[i] = a;
				hi[i] = b + 1;
				f[i] = ll(rng() % 1000);
				is_apply[i] = char(rng() % 2);
				base[i] = int(rng() % (i + 1)); // the version an op of a branching history acts on
			}
			// (checksum, arena nodes) of one run
			auto const run = [&]<typename Tree>(std::type_identity<Tree>)
				requires requires(Tree const &tree) {
					tree.fold(0, 1);
					tree.apply(0, 1, ll{});
					tree.arena_size();
				}
			{
				Tree cur(std::vector<node>(std::size_t(n), node{1, 1}), node{0, 0}, ll(0));
				ll acc = 0;
				if (!is_branching) {
					for (int i = 0; i < m; ++i)
						if (is_apply[i])
							cur = std::move(cur).apply(std::size_t(lo[i]), std::size_t(hi[i]), f[i]);
						else
							acc += cur.fold(std::size_t(lo[i]), std::size_t(hi[i])).sum;
					return std::pair{acc, cur.arena_size()};
				}
				std::vector<Tree> versions{cur};
				versions.reserve(m + 1);
				for (int i = 0; i < m; ++i) {
					Tree const &v = versions[std::size_t(base[i]) % versions.size()];
					if (is_apply[i])
						versions.push_back(v.apply(std::size_t(lo[i]), std::size_t(hi[i]), f[i]));
					else
						acc += v.fold(std::size_t(lo[i]), std::size_t(hi[i])).sum;
				}
				return std::pair{acc, cur.arena_size()};
			};
			auto const [want, ordered_nodes] = run(std::type_identity<ordered>{});
			auto const [got, commuting_nodes] = run(std::type_identity<commuting>{});
			if (want != got) FAIL("default and commutative modes disagree");
			auto const shape = std::format("n={}k {} history", n / 1000, is_branching ? "branching" : "linear");
			auto const mb = [](std::size_t nodes, std::size_t bytes) { return double(nodes * bytes) / 1e6; };
			measure(std::format("{}, default mode (arena {} nodes, {:.1f} MB)", shape, ordered_nodes,
								mb(ordered_nodes, sizeof(node) + sizeof(ll) + 8)),
					[&] {
						return run(std::type_identity<ordered>{}).first;
					});
			measure(std::format("{}, commutative mode (arena {} nodes, {:.1f} MB)", shape, commuting_nodes,
								mb(commuting_nodes, sizeof(node) + sizeof(ll) + 8)),
					[&] {
						return run(std::type_identity<commuting>{}).first;
					});
		}
}

TEST_CASE("persistent_lazy_segment_tree lean partial copy", "[persistent_lazy_segment_tree][lean]") {
	constexpr int m = 200'000;
	constexpr ll mod = 998'244'353;
	namespace ex = atl::experimental;

	auto const race = [&]<typename Value, typename Tag, typename Op, typename Map, typename Compose, typename TagOrder>(
						  std::string const &shape, int n, bool is_branching, Value element, Value zero, Tag tag_identity,
						  Op op, Map map, Compose compose, auto make_tag, auto sum_of, std::type_identity<TagOrder>)
		requires std::copyable<Value> && std::copyable<Tag> && atl::binary_operation<Op, Value> && atl::action<Map, Tag, Value>
			  && atl::binary_operation<Compose, Tag> && atl::tag_order<TagOrder>
			  && std::is_invocable_r_v<Tag, decltype(make_tag) &, std::mt19937 &>
			  && std::is_invocable_r_v<ll, decltype(sum_of) &, Value const &>
	{
		std::mt19937 rng(3);
		std::vector<int> lo(m), hi(m), base(m);
		std::vector<Tag> f(m);
		std::vector<char> is_apply(m);
		for (int i = 0; i < m; ++i) {
			int a = int(rng() % n), b = int(rng() % n);
			if (a > b) std::swap(a, b);
			lo[i] = a;
			hi[i] = b + 1;
			f[i] = make_tag(rng);
			is_apply[i] = char(rng() % 2);
			base[i] = int(rng() % (i + 1));
		}
		auto const run = [&]<typename Tree>(std::type_identity<Tree>)
			requires std::constructible_from<Tree, std::vector<Value>, Value, Tag, Op, Map, Compose> && requires(Tree const &tree, Tag const &tag) {
				tree.fold(0, 1);
				tree.apply(0, 1, tag);
			}
		{
			Tree cur(std::vector<Value>(std::size_t(n), element), zero, tag_identity, op, map, compose);
			ll acc = 0;
			if (!is_branching) {
				for (int i = 0; i < m; ++i)
					if (is_apply[i])
						cur = std::move(cur).apply(std::size_t(lo[i]), std::size_t(hi[i]), f[i]);
					else
						acc += sum_of(cur.fold(std::size_t(lo[i]), std::size_t(hi[i])));
				return acc;
			}
			std::vector<Tree> versions{cur};
			versions.reserve(m + 1);
			for (int i = 0; i < m; ++i) {
				Tree const &v = versions[std::size_t(base[i]) % versions.size()];
				if (is_apply[i])
					versions.push_back(v.apply(std::size_t(lo[i]), std::size_t(hi[i]), f[i]));
				else
					acc += sum_of(v.fold(std::size_t(lo[i]), std::size_t(hi[i])));
			}
			return acc;
		};
		using library = atl::persistent_lazy_segment_tree<Value, Tag, Op, Map, Compose, TagOrder>;
		using lean = ex::persistent_lazy_segment_tree_by_partial_copy<Value, Tag, Op, Map, Compose, TagOrder, true>;
		if (run(std::type_identity<library>{}) != run(std::type_identity<lean>{})) FAIL("whole-node and lean partial-copy variants disagree");
		auto const label = std::format("lean partial copy: {}, n={}k {} history", shape, n / 1000, is_branching ? "branching" : "linear");
		measure(label + ", persistent_lazy_segment_tree (library)", [&] { return run(std::type_identity<library>{}); });
		measure(label + ", persistent_lazy_segment_tree lean partial copy", [&] { return run(std::type_identity<lean>{}); });
	};

	struct node {
		ll sum, width;
	};

	struct affine {
		ll b, c;
	};

	race(std::string("affine"), 100'000, false, node{1, 1}, node{0, 0}, affine{1, 0}, [](node const &x, node const &y) { return node{(x.sum + y.sum) % mod, x.width + y.width}; }, [](affine const &f, node const &s) { return node{(f.b * s.sum + f.c * s.width) % mod, s.width}; }, [](affine const &f, affine const &g) { return affine{f.b * g.b % mod, (f.b * g.c + f.c) % mod}; }, [](std::mt19937 &g) { return affine{ll(1 + g() % (mod - 1)), ll(g() % mod)}; }, [](node const &x) { return x.sum; }, std::type_identity<void>{});

	for (int n : {100'000, 200'000})
		for (bool const is_branching : {false, true})
			race(std::string("add"), n, is_branching, node{1, 1}, node{0, 0}, ll(0), [](node const &x, node const &y) { return node{x.sum + y.sum, x.width + y.width}; }, [](ll f, node const &s) { return node{s.sum + f * s.width, s.width}; }, [](ll f, ll g) { return f + g; }, [](std::mt19937 &g) { return ll(g() % 1000); }, [](node const &x) { return x.sum; }, std::type_identity<atl::commutative_tags_t>{});

	struct wide {
		ll sum, width, lo, hi;
	};

	auto const wide_op = [](wide const &x, wide const &y) {
		return wide{x.sum + y.sum, x.width + y.width, (std::min)(x.lo, y.lo), (std::max)(x.hi, y.hi)};
	};
	auto const wide_map = [](ll f, wide const &s) { return wide{s.sum + f * s.width, s.width, s.lo + f, s.hi + f}; };
	auto const wide_add = [](ll f, ll g) { return f + g; };
	auto const wide_tag = [](std::mt19937 &g) { return ll(g() % 1000); };
	auto const wide_sum = [](wide const &x) { return x.sum + x.hi - x.lo; };
	wide const w1{1, 1, 1, 1}, w0{0, 0, (std::numeric_limits<ll>::max)() / 4, (std::numeric_limits<ll>::min)() / 4};
	race(std::string("wide add, default mode"), 100'000, false, w1, w0, ll(0), wide_op, wide_map, wide_add, wide_tag,
		 wide_sum, std::type_identity<void>{});
	race(std::string("wide add, commutative mode"), 100'000, false, w1, w0, ll(0), wide_op, wide_map, wide_add, wide_tag,
		 wide_sum, std::type_identity<atl::commutative_tags_t>{});
}
