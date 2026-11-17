/**
 * @file
 * @brief Sequence throughput: shared-engine versus direct-field treap recursion
 * on 400,000 inserts/folds from 200,000 elements; treap versus lazy segment tree
 * on fixed-size affine actions and sums. Construction is timed. The direct-field
 * contender folds by split/join; the library reads without splitting.
 * Not measured yet: no CPU/compiler/flags/date record is available.
 */
#include <atl/ds/lazy_segment_tree.hpp>
#include <atl/ds/treap_sequence.hpp>
#include <atl/utils/hash.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <utility>
#include <vector>

using atl::uint32;
using atl::uint64;
using ll = long long;

namespace {
	/** @brief Implicit treap with direct node-field recursion and split/join folds. */
	struct treap_sequence_by_direct_fields {
		struct node {
			ll value, agg;
			uint32 left = 0, right = 0, size = 0, priority = 0;
		};

		std::vector<node> nodes{1};
		uint64 rng = atl::detail::hash_seed() ^ 0x9e3779b97f4a7c15ULL;
		uint32 root = 0;

		uint32 make(ll v) {
			atl_assert(nodes.size() < (std::numeric_limits<uint32>::max)());
			nodes.push_back({v, v, 0, 0, 1, uint32(atl::detail::splitmix64(rng += 0x9e3779b97f4a7c15ULL) >> 32)});
			return uint32(nodes.size() - 1);
		}

		void pull(uint32 t) {
			node &nd = nodes[t];
			nd.size = 1 + nodes[nd.left].size + nodes[nd.right].size;
			nd.agg = nodes[nd.left].agg + nd.value + nodes[nd.right].agg;
		}

		uint32 merge(uint32 a, uint32 b) {
			if (a == 0 || b == 0) return a | b;
			if (nodes[a].priority < nodes[b].priority) {
				nodes[b].left = merge(a, nodes[b].left);
				pull(b);
				return b;
			}
			nodes[a].right = merge(nodes[a].right, b);
			pull(a);
			return a;
		}

		std::pair<uint32, uint32> split_at(uint32 t, uint32 k) {
			if (t == 0) return {0, 0};
			uint32 const ls = nodes[nodes[t].left].size;
			if (ls < k) {
				auto [m, r] = split_at(nodes[t].right, k - ls - 1);
				nodes[t].right = m;
				pull(t);
				return {t, r};
			}
			auto [l, m] = split_at(nodes[t].left, k);
			nodes[t].left = m;
			pull(t);
			return {l, t};
		}

		std::size_t size() const { return nodes[root].size; }

		void insert(std::size_t i, ll v) {
			atl_assert(i <= size());
			auto [l, r] = split_at(root, uint32(i));
			root = merge(merge(l, make(v)), r);
		}

		ll fold(std::size_t l, std::size_t r) {
			atl_assert(l <= r && r <= size());
			auto [lm, c] = split_at(root, uint32(r));
			auto [a, b] = split_at(lm, uint32(l));
			ll const res = nodes[b].agg;
			root = merge(merge(a, b), c);
			return res;
		}
	};
}

TEST_CASE("treap_sequence benchmarks: engine vs hand-rolled", "[treap_sequence]") {
	constexpr int n = 200'000, m = 400'000;
	std::mt19937 rng(1);
	std::vector<uint32> pos(m), lo(m), hi(m);
	std::vector<ll> val(m);
	for (int i = 0; i < m; ++i) {
		pos[i] = rng() % uint32(n + i + 1);
		val[i] = ll(rng() % 1000);
		uint32 a = rng() % uint32(n), b = rng() % uint32(n);
		if (a > b) std::swap(a, b);
		lo[i] = a;
		hi[i] = b + 1;
	}

	auto const compute_checksum = [&]<typename Make>(Make make)
		requires std::invocable<Make &> && requires(std::invoke_result_t<Make &> &tree) {
			{ tree.size() } -> std::convertible_to<std::size_t>;
			tree.insert(std::size_t{}, ll{});
			{ tree.fold(std::size_t{}, std::size_t{}) } -> std::convertible_to<ll>;
		}
	{
		auto tree = std::invoke(make);
		ll acc = 0;
		for (int i = 0; i < m; ++i)
			if (i % 2 == 0)
				tree.insert(pos[i] % (tree.size() + 1), val[i]);
			else
				acc += tree.fold(lo[i], hi[i]);
		return acc;
	};
	auto const make_library = [&] { return atl::treap_sequence(std::vector<ll>(n, 1), ll(0), std::plus<>{}); };
	auto const make_direct_fields = [&] {
		treap_sequence_by_direct_fields tree;
		for (int i = 0; i < n; ++i) tree.root = tree.merge(tree.root, tree.make(1));
		return tree;
	};
	REQUIRE(compute_checksum(make_library) == compute_checksum(make_direct_fields));
	BENCHMARK("library -- insert/fold mix") { return compute_checksum(make_library); };
	BENCHMARK("by direct fields -- insert/fold mix") { return compute_checksum(make_direct_fields); };
}

TEST_CASE("treap_sequence benchmarks: vs lazy_segment_tree", "[treap_sequence]") {
	constexpr int n = 200'000, m = 400'000;
	constexpr ll mod = 998'244'353;
	std::mt19937 rng(1);
	std::vector<uint32> lo(m), hi(m);
	std::vector<ll> fb(m), fc(m);
	std::vector<char> is_apply(m);
	for (int i = 0; i < m; ++i) {
		uint32 a = rng() % uint32(n), b = rng() % uint32(n);
		if (a > b) std::swap(a, b);
		lo[i] = a;
		hi[i] = b + 1;
		fb[i] = ll(1 + rng() % (mod - 1));
		fc[i] = ll(rng() % mod);
		is_apply[i] = char(rng() % 2);
	}

	struct node {
		ll s, w;
	};

	struct aff {
		ll b, c;
	};

	auto const op = [](node const &x, node const &y) { return node{(x.s + y.s) % mod, x.w + y.w}; };
	auto const mp = [](aff const &f, node const &s) { return node{(f.b * s.s + f.c * s.w) % mod, s.w}; };
	auto const cp = [](aff const &f, aff const &g) { return aff{f.b * g.b % mod, (f.b * g.c + f.c) % mod}; };

	auto const compute_checksum = [&]<typename Make>(Make make)
		requires std::invocable<Make &> && requires(std::invoke_result_t<Make &> &tree) {
			tree.apply(std::size_t{}, std::size_t{}, aff{});
			{ tree.fold(std::size_t{}, std::size_t{}).s } -> std::convertible_to<ll>;
		}
	{
		auto tree = std::invoke(make);
		ll acc = 0;
		for (int i = 0; i < m; ++i)
			if (is_apply[i])
				tree.apply(lo[i], hi[i], aff{fb[i], fc[i]});
			else
				acc += tree.fold(lo[i], hi[i]).s;
		return acc;
	};
	auto const make_library = [&] { return atl::treap_sequence(std::vector<node>(n, node{1, 1}), node{0, 0}, aff{1, 0}, op, mp, cp); };
	auto const make_dense = [&] { return atl::lazy_segment_tree(std::vector<node>(n, node{1, 1}), node{0, 0}, aff{1, 0}, op, mp, cp); };
	REQUIRE(compute_checksum(make_library) == compute_checksum(make_dense));
	BENCHMARK("library -- fixed-size apply/fold") { return compute_checksum(make_library); };
	BENCHMARK("by lazy segment tree -- fixed-size apply/fold") { return compute_checksum(make_dense); };
}
