/**
 * @file
 * @brief Range-action and point-read stream throughput.
 *
 * Lazy aggregation maintains width-carrying sums; commutative mode skips boundary pushes; difference Fenwick uses inverse tags at endpoints. Fresh trees replay one million affine, add or xor operations at n=1e6 with 50/50 and 90/10 apply/read mixes.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, -O2 and -O2 -march=native, 2026-11-01: add 50/50 library 93.0/98.0 ms versus difference Fenwick 40.1/39.9 ms. Affine and ordered-mode rows: 2026-10-24, same CPU/compiler/flags.
 * Full measurements: docs/dual-segment-tree.md.
 */
#include <atl/ds/difference_fenwick.hpp>
#include <atl/ds/dual_segment_tree.hpp>
#include <atl/ds/lazy_segment_tree.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <format>
#include <functional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

using ll = long long;

// Dual segment tree vs lazy_segment_tree on the dual tree's home workload:
// range affine apply / point get. The lazy tree must carry (sum, width)
// aggregates and repair them after every apply; the dual tree stores tags only.
// Each sample builds fresh trees and replays the same mixed 50/50 op stream.
TEST_CASE("dual_segment_tree benchmarks", "[dual_segment_tree]") {
	constexpr int n = 1'000'000, m = 1'000'000;
	constexpr ll mod = 998'244'353;
	std::mt19937 rng(1);
	std::vector<int> lo(m), hi(m), pos(m);
	std::vector<ll> fb(m), fc(m);
	std::vector<char> is_apply(m);
	for (int i = 0; i < m; ++i) {
		int a = int(rng() % n), b = int(rng() % n);
		if (a > b) std::swap(a, b);
		lo[i] = a;
		hi[i] = b + 1;
		pos[i] = int(rng() % n);
		fb[i] = ll(1 + rng() % (mod - 1));
		fc[i] = ll(rng() % mod);
		is_apply[i] = char(rng() % 2);
	}

	struct affine {
		ll b, c;
	};

	auto const map1 = [](affine const &f, ll s) { return (f.b * s + f.c) % mod; };
	auto const compose = [](affine const &f, affine const &g) {
		return affine{f.b * g.b % mod, (f.b * g.c + f.c) % mod};
	};

	BENCHMARK("dual -- 1e6 mixed apply/get") {
		atl::dual_segment_tree st(std::vector<ll>(n, 1), affine{1, 0}, map1, compose);
		ll acc = 0;
		for (int i = 0; i < m; ++i)
			if (is_apply[i])
				st.apply(std::size_t(lo[i]), std::size_t(hi[i]), affine{fb[i], fc[i]});
			else
				acc += st.at(std::size_t(pos[i]));
		return acc;
	};

	using node = std::pair<ll, ll>; // (sum, width) — the lawful lazy algebra
	BENCHMARK("lazy -- 1e6 mixed apply/get") {
		atl::lazy_segment_tree st(
			std::vector<node>(n, node{1, 1}), node{0, 0}, affine{1, 0},
			[](node const &x, node const &y) { return node{(x.first + y.first) % mod, x.second + y.second}; },
			[](affine const &f, node const &s) { return node{(f.b * s.first + f.c * s.second) % mod, s.second}; },
			compose);
		ll acc = 0;
		for (int i = 0; i < m; ++i)
			if (is_apply[i])
				st.apply(std::size_t(lo[i]), std::size_t(hi[i]), affine{fb[i], fc[i]});
			else
				acc += st.at(std::size_t(pos[i])).first;
		return acc;
	};
}

// Compare the commutative mode (no boundary pushes) vs the default mode on
// range add / point get, n = 1e6, 1e6 ops, at 50/50 and 90/10 apply-heavy. Same algebra
// in both; only the TagOrder argument differs.
TEST_CASE("dual_segment_tree commutative mode", "[dual_segment_tree][commutative]") {
	constexpr int n = 1'000'000, m = 1'000'000;
	auto add = [](ll f, ll s) { return f + s; };
	using ordered = atl::dual_segment_tree<ll, ll, decltype(add), decltype(add)>;
	using commuting = atl::dual_segment_tree<ll, ll, decltype(add), decltype(add), atl::commutative_tags_t>;
	for (int apply_percent : {50, 90}) {
		std::mt19937 rng(2);
		std::vector<int> lo(m), hi(m), pos(m);
		std::vector<ll> f(m);
		std::vector<char> is_apply(m);
		for (int i = 0; i < m; ++i) {
			int a = int(rng() % n), b = int(rng() % n);
			if (a > b) std::swap(a, b);
			lo[i] = a;
			hi[i] = b + 1;
			pos[i] = int(rng() % n);
			f[i] = ll(rng() % 2001) - 1000;
			is_apply[i] = char(int(rng() % 100) < apply_percent);
		}
		auto const replay_points = [&]<typename Factory>(Factory factory)
			requires std::invocable<Factory &> && requires(std::invoke_result_t<Factory &> &tree) { tree.apply(0, 1, ll(0)); { tree.at(0) } -> std::convertible_to<ll>; }
		{
			auto st = std::invoke(factory);
			ll acc = 0;
			for (int i = 0; i < m; ++i)
				if (is_apply[i])
					st.apply(std::size_t(lo[i]), std::size_t(hi[i]), f[i]);
				else
					acc += st.at(std::size_t(pos[i]));
			return acc;
		};
		if (replay_points([] { return ordered(std::size_t(n), ll(0), ll(0)); }) != replay_points([] { return commuting(std::size_t(n), ll(0), ll(0)); }))
			FAIL("default and commutative modes disagree");
		BENCHMARK(std::format("range add / point get {}/{}, default mode", apply_percent, 100 - apply_percent)) {
			return replay_points([] { return ordered(std::size_t(n), ll(0), ll(0)); });
		};
		BENCHMARK(std::format("range add / point get {}/{}, commutative mode", apply_percent,
							  100 - apply_percent)) {
			return replay_points([] { return commuting(std::size_t(n), ll(0), ll(0)); });
		};
	}
}

// Compare the public Fenwick difference array vs the retained
// commutative dual tree on the invertible tags, range add and range xor / point get,
// n = 1e6, 1e6 ops, at 50/50 and 90/10 apply-heavy. The two structures use identical operation streams.
TEST_CASE("dual_segment_tree vs difference_fenwick", "[dual_segment_tree][difference_fenwick]") {
	constexpr int n = 1'000'000, m = 1'000'000;
	auto const race = [&]<atl::binary_operation<ll> Op, atl::inverse_operation<ll> Inverse>(char const *tag, bool is_xor, Op op, Inverse inverse) {
		using op_type = decltype(op);
		using inverse_type = decltype(inverse);
		for (int apply_percent : {50, 90}) {
			std::mt19937 rng(2);
			std::vector<int> lo(m), hi(m), pos(m);
			std::vector<ll> f(m);
			std::vector<char> is_apply(m);
			for (int i = 0; i < m; ++i) {
				int a = int(rng() % n), b = int(rng() % n);
				if (a > b) std::swap(a, b);
				lo[i] = a;
				hi[i] = b + 1;
				pos[i] = int(rng() % n);
				f[i] = is_xor ? ll(rng() % 1024) : ll(rng() % 2001) - 1000;
				is_apply[i] = char(int(rng() % 100) < apply_percent);
			}
			auto const replay_points = [&]<typename Factory>(Factory factory)
				requires std::invocable<Factory &> && requires(std::invoke_result_t<Factory &> &tree) { tree.apply(0, 1, ll(0)); { tree.at(0) } -> std::convertible_to<ll>; }
			{
				auto st = std::invoke(factory);
				ll acc = 0;
				for (int i = 0; i < m; ++i)
					if (is_apply[i])
						st.apply(std::size_t(lo[i]), std::size_t(hi[i]), f[i]);
					else
						acc += st.at(std::size_t(pos[i]));
				return acc;
			};
			auto const dual = [&] { return atl::dual_segment_tree(atl::commutative_tags, std::size_t(n), ll(0), ll(0), op, op); };
			auto const fenwick = [&] {
				return atl::difference_fenwick<ll, op_type, inverse_type>(std::size_t(n), ll(0), ll(0), op, inverse);
			};
			if (replay_points(dual) != replay_points(fenwick)) FAIL("dual tree and difference Fenwick results disagree");
			BENCHMARK(std::format("range {} / point get {}/{}, dual_segment_tree (library)", tag, apply_percent,
								  100 - apply_percent)) {
				return replay_points(dual);
			};
			BENCHMARK(std::format("range {} / point get {}/{}, difference_fenwick (library)", tag, apply_percent,
								  100 - apply_percent)) {
				return replay_points(fenwick);
			};
		}
	};
	race("add", false, std::plus<ll>{}, std::negate<ll>{});
	race("xor", true, std::bit_xor<ll>{}, std::identity{});
}
