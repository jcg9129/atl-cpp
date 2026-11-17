/**
 * @file
 * @brief Sparse point-update and range-fold throughput.
 *
 * Materialized paths allocate a path for every touched position; offline compression uses a dense tree; a treap map stores only touched keys; two-child folding visits absent or disjoint children. Mixed affine streams use a 1e9 domain; short and uniform affine/sum folds use 1e9 and 1e18 domains where size_t permits.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, -O2 and -O2 -march=native, 2026-10-15: mixed library 263/271 ms, materialized paths 430/427 ms, offline compression 174/189 ms, treap map 604/633 ms.
 * Full measurements: docs/dynamic-segment-tree.md.
 */
#include <atl/ds/dynamic_segment_tree.hpp>
#include <atl/ds/segment_tree.hpp>
#include <atl/ds/treap_map.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <concepts>

#include <ranges>

using atl::uint32;
using ll = long long;

namespace {
	constexpr ll mod = 998244353;

	struct aff {
		ll a, b;
	};

	/** @brief Sparse affine composition with a full root-to-leaf path per touched position. */
	template<atl::binary_operation<aff> Op>
	struct dynamic_segment_tree_by_materialized_paths {
		struct node {
			aff sum;
			uint32 left = 0, right = 0;
		};

		Op op;
		std::size_t n;
		std::vector<node> nodes{1, node{{1, 0}, 0, 0}};
		uint32 root = 0;

		uint32 point(uint32 x, std::size_t lo, std::size_t hi, std::size_t i, aff const &v) {
			if (x == 0) {
				atl_assert(nodes.size() < 0xFFFFFFFFull);
				nodes.push_back({{1, 0}, 0, 0});
				x = uint32(nodes.size() - 1);
			}
			if (hi - lo == 1) {
				nodes[x].sum = v;
				return x;
			}
			std::size_t const mid = lo + (hi - lo) / 2;
			if (i < mid) {
				uint32 const c = point(nodes[x].left, lo, mid, i, v);
				nodes[x].left = c;
			} else {
				uint32 const c = point(nodes[x].right, mid, hi, i, v);
				nodes[x].right = c;
			}
			node &nd = nodes[x];
			nd.sum = op(nodes[nd.left].sum, nodes[nd.right].sum);
			return x;
		}

		aff fold(uint32 x, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r) const {
			if (x == 0 || r <= lo || hi <= l) return {1, 0};
			node const &nd = nodes[x];
			if (l <= lo && hi <= r) return nd.sum;
			std::size_t const mid = lo + (hi - lo) / 2;
			return op(fold(nd.left, lo, mid, l, r), fold(nd.right, mid, hi, l, r));
		}

		void set(std::size_t i, aff const &v) { root = point(root, 0, n, i, v); }

		aff fold(std::size_t l, std::size_t r) const { return fold(root, 0, n, l, r); }
	};
}

// Non-commutative affine composition makes ordering errors observable.
TEST_CASE("dynamic_segment_tree benchmarks", "[dynamic_segment_tree]") {
	constexpr std::size_t domain = 1'000'000'000, q = 500'000;

	auto op = [](aff const &x, aff const &y) { return aff{y.a * x.a % mod, (y.a * x.b + y.b) % mod}; };

	std::mt19937_64 rng(1);

	struct step {
		bool is_set;
		std::size_t i, j;
		aff f;
	};

	std::vector<step> steps(q);
	std::vector<std::size_t> positions;
	for (auto &s : steps) {
		s.is_set = rng() % 2 == 0;
		if (s.is_set) {
			s.i = rng() % domain;
			s.f = {1 + ll(rng() % (mod - 1)), ll(rng() % mod)};
			positions.push_back(s.i);
		} else {
			s.i = rng() % domain;
			s.j = s.i + 1 + rng() % (domain - s.i);
		}
	}

	{
		atl::dynamic_segment_tree probe(domain, aff{1, 0}, op);
		dynamic_segment_tree_by_materialized_paths<decltype(op)> nprobe{op, domain};
		atl::treap_map<std::size_t, aff, std::less<std::size_t>, decltype(op)> keyed(aff{1, 0}, op);
		for (auto const &s : steps)
			if (s.is_set) {
				probe.set(s.i, s.f);
				nprobe.set(s.i, s.f);
				keyed.insert_or_assign(s.i, s.f);
			} else {
				auto const expected = probe.fold(s.i, s.j), actual = nprobe.fold(s.i, s.j);
				REQUIRE(actual.a == expected.a);
				REQUIRE(actual.b == expected.b);
				auto const keyed_actual = keyed.fold(s.i, s.j);
				REQUIRE(keyed_actual.a == expected.a);
				REQUIRE(keyed_actual.b == expected.b);
			}
		std::printf("nodes: naive %zu, compressed %zu (%.1fx fewer)\n", nprobe.nodes.size(),
					probe.node_count(), double(nprobe.nodes.size()) / double(probe.node_count()));
	}

	BENCHMARK("by materialized paths") {
		dynamic_segment_tree_by_materialized_paths<decltype(op)> t{op, domain};
		t.nodes.reserve(q * 16);
		ll acc = 0;
		for (auto const &s : steps)
			if (s.is_set)
				t.set(s.i, s.f);
			else
				acc += t.fold(s.i, s.j).a;
		return acc;
	};

	BENCHMARK("library") {
		atl::dynamic_segment_tree t(domain, aff{1, 0}, op);
		t.reserve(q / 2);
		ll acc = 0;
		for (auto const &s : steps)
			if (s.is_set)
				t.set(s.i, s.f);
			else
				acc += t.fold(s.i, s.j).a;
		return acc;
	};

	/** @brief Fold sparse affine values by keeping only touched positions in a keyed treap. */
	auto const fold_by_treap_map = [&] {
		atl::treap_map<std::size_t, aff, std::less<std::size_t>, decltype(op)> t(aff{1, 0}, op);
		ll acc = 0;
		for (auto const &s : steps)
			if (s.is_set)
				t.insert_or_assign(s.i, s.f);
			else
				acc += t.fold(s.i, s.j).a;
		return acc;
	};
	BENCHMARK("by treap map") { return fold_by_treap_map(); };

	std::vector<std::size_t> sorted(positions);
	std::ranges::sort(sorted);
	sorted.erase(std::ranges::unique(sorted).begin(), sorted.end());
	auto rank_of = [&](std::size_t p) { return std::ranges::lower_bound(sorted, p) - sorted.begin(); };
	BENCHMARK("by offline compression") {
		atl::segment_tree t(sorted.size(), aff{1, 0}, op);
		ll acc = 0;
		for (auto const &s : steps)
			if (s.is_set)
				t.set(rank_of(s.i), s.f);
			else
				acc += t.fold(rank_of(s.i), rank_of(s.j)).a;
		return acc;
	};
}

namespace {
	/** @brief Compressed leaves with fold recursion into both children, even when absent or disjoint. */
	template<std::copyable T, atl::binary_operation<T> Op>
	struct dynamic_segment_tree_by_two_child_fold {
		struct node {
			T aggregate;
			std::size_t pos;
			uint32 left = 0, right = 0;
			bool is_pending = false;
		};

		Op op;
		T e;
		std::size_t n;
		std::vector<node> nodes{node{e, 0, 0, 0, false}};
		uint32 root = 0;

		uint32 make(T a, std::size_t pos, bool pending) {
			atl_assert(nodes.size() < 0xFFFFFFFFull);
			nodes.push_back({std::move(a), pos, 0, 0, pending});
			return uint32(nodes.size() - 1);
		}

		uint32 point(uint32 x, std::size_t lo, std::size_t hi, std::size_t i, T const &v) {
			if (x == 0) return make(v, i, true);
			if (nodes[x].is_pending) {
				if (nodes[x].pos == i) {
					nodes[x].aggregate = v;
					return x;
				}
				std::size_t const mid = lo + (hi - lo) / 2;
				uint32 const c = make(nodes[x].aggregate, nodes[x].pos, true);
				nodes[x].is_pending = false;
				(nodes[c].pos < mid ? nodes[x].left : nodes[x].right) = c;
			}
			std::size_t const mid = lo + (hi - lo) / 2;
			if (i < mid) {
				uint32 const c = point(nodes[x].left, lo, mid, i, v);
				nodes[x].left = c;
			} else {
				uint32 const c = point(nodes[x].right, mid, hi, i, v);
				nodes[x].right = c;
			}
			nodes[x].aggregate = op(nodes[nodes[x].left].aggregate, nodes[nodes[x].right].aggregate);
			return x;
		}

		T fold(uint32 x, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r) const {
			if (x == 0 || r <= lo || hi <= l) return e;
			node const &nd = nodes[x];
			if (nd.is_pending) return l <= nd.pos && nd.pos < r ? nd.aggregate : e;
			if (l <= lo && hi <= r) return nd.aggregate;
			std::size_t const mid = lo + (hi - lo) / 2;
			return op(fold(nd.left, lo, mid, l, r), fold(nd.right, mid, hi, l, r));
		}

		void set(std::size_t i, T const &v) { root = point(root, 0, n, i, v); }

		T fold(std::size_t l, std::size_t r) const { return fold(root, 0, n, l, r); }
	};
}

TEST_CASE("dynamic_segment_tree overlapping-child fold comparison", "[dynamic_segment_tree][race]") {
	auto op = [](aff const &x, aff const &y) { return aff{y.a * x.a % mod, (y.a * x.b + y.b) % mod}; };
	auto plus = [](ll x, ll y) { return x + y; };
	for (std::size_t const domain : {std::size_t(1'000'000'000), std::size_t((std::min)(1'000'000'000'000'000'000ull, static_cast<unsigned long long>((std::numeric_limits<std::size_t>::max)())))}) {
		std::mt19937_64 rng(2);
		std::size_t const touched = 250'000, q = 1'000'000;
		std::vector<std::size_t> pos(touched);
		for (auto &p : pos) p = rng() % domain;
		atl::dynamic_segment_tree t(domain, aff{1, 0}, op);
		dynamic_segment_tree_by_two_child_fold<aff, decltype(op)> lt{op, aff{1, 0}, domain};
		atl::dynamic_segment_tree ts(domain, ll(0), plus);
		dynamic_segment_tree_by_two_child_fold<ll, decltype(plus)> lts{plus, 0, domain};
		for (std::size_t k = 0; k < touched; ++k) {
			aff const f{1 + ll(rng() % (mod - 1)), ll(rng() % mod)};
			t.set(pos[k], f), lt.set(pos[k], f), ts.set(pos[k], ll(k)), lts.set(pos[k], ll(k));
		}
		std::ranges::sort(pos);
		std::vector<std::pair<std::size_t, std::size_t>> shorts(q), wide(q);
		for (auto &[l, r] : shorts) { // around touched positions, length 1..64
			std::size_t const p = pos[rng() % touched];
			l = p - std::min<std::size_t>(p, rng() % 32);
			r = l + 1 + (rng() % (std::min)(std::size_t(64), domain - l));
		}
		for (auto &[l, r] : wide) {
			std::size_t a = rng() % domain, b = rng() % domain;
			if (a > b) std::swap(a, b);
			l = a, r = b + 1;
		}
		for (std::size_t k = 0; k < 20000; ++k) {
			auto const [l, r] = k % 2 ? shorts[k] : wide[k];
			aff const x = t.fold(l, r), y = lt.fold(l, r);
			if (x.a != y.a || x.b != y.b || ts.fold(l, r) != lts.fold(l, r)) FAIL("folds disagree");
		}
		std::string const d = domain == 1'000'000'000 ? "1e9" : (sizeof(std::size_t) >= 8 ? "1e18" : "size_t max");
		auto race = [&]<std::ranges::input_range Queries>(std::string const &shape, Queries const &qs)
			requires requires(std::ranges::range_reference_t<Queries const> query) { { query.first } -> std::convertible_to<std::size_t>; { query.second } -> std::convertible_to<std::size_t>; }
		{
			BENCHMARK(shape + " over " + d + ", affine (library)") {
				ll acc = 0;
				for (auto const &[l, r] : qs) acc += t.fold(l, r).b;
				return acc;
			};
			BENCHMARK(shape + " over " + d + ", affine (by two-child fold)") {
				ll acc = 0;
				for (auto const &[l, r] : qs) acc += lt.fold(l, r).b;
				return acc;
			};
			BENCHMARK(shape + " over " + d + ", ll sum (library)") {
				ll acc = 0;
				for (auto const &[l, r] : qs) acc += ts.fold(l, r);
				return acc;
			};
			BENCHMARK(shape + " over " + d + ", ll sum (by two-child fold)") {
				ll acc = 0;
				for (auto const &[l, r] : qs) acc += lts.fold(l, r);
				return acc;
			};
		};
		race("1e6 folds len 1-64", shorts);
		race("1e6 folds uniform", wide);
	}
}
