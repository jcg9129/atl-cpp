/**
 * @file
 * @brief Sparse range-action and range-fold throughput.
 *
 * Materialized push creates both children; offline compression uses width-carrying dense cells; point reads compare a single tag walk against a singleton fold. Fresh trees replay 1e5 affine steps on a 1e9 domain, then independent 1e6 point reads.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, -O2 and -O2 -march=native, 2026-10-15: library 225/229 ms, materialized push 222/226 ms, offline compression 123/120 ms; point walk 395/396 ms versus singleton fold 549/548 ms.
 * Full measurements: docs/dynamic-lazy-segment-tree.md.
 */
#include <atl/ds/dynamic_lazy_segment_tree.hpp>
#include <atl/ds/lazy_segment_tree.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

#include <utility>

using ll = long long;

// The online tax for range updates: dynamic lazy tree over [0, 1e9) vs a dense
// lazy_segment_tree over the operation boundaries compressed offline (each
// dense slot represents a segment, its width carried in S).
TEST_CASE("dynamic_lazy_segment_tree benchmarks", "[dynamic_lazy_segment_tree]") {
	constexpr ll mod = 998244353;
	constexpr std::size_t domain = 1'000'000'000, q = 100'000;

	struct item {
		ll s, w;
	};

	struct aff {
		ll a, b;

		bool operator==(aff const &) const = default;
	};

	auto op = [](item const &x, item const &y) { return item{(x.s + y.s) % mod, x.w + y.w}; };
	auto mp = [](aff const &f, item const &s) { return item{(f.a * s.s + f.b % mod * (s.w % mod)) % mod, s.w}; };
	auto cp = [](aff const &f, aff const &g) { return aff{f.a * g.a % mod, (f.a * g.b + f.b) % mod}; };
	auto fill = [](std::size_t w) { return item{0, ll(w)}; };

	std::mt19937_64 rng(1);

	struct step {
		bool is_apply;
		std::size_t l, r;
		aff f;
	};

	std::vector<step> steps(q);
	for (auto &s : steps) {
		s.is_apply = rng() % 2 == 0;
		s.l = rng() % domain;
		s.r = s.l + 1 + rng() % (domain - s.l);
		if (s.is_apply) s.f = {1 + ll(rng() % (mod - 1)), ll(rng() % mod)};
	}

	{
		atl::dynamic_lazy_segment_tree probe(domain, item{0, 0}, aff{1, 0}, op, mp, cp, fill);
		for (auto const &s : steps)
			if (s.is_apply) probe.apply(s.l, s.r, s.f);
		std::printf("nodes materialized: %zu (%.1f/apply)\n", probe.node_count(),
					double(probe.node_count()) / double(q / 2));
	}

	// hand-written baseline: hardcoded affine-sum algebra, materialize both
	// children on every push (the textbook formulation)
	/** @brief Dynamic affine sums with both children materialized on every push. */
	struct dynamic_lazy_segment_tree_by_materialized_push {
		struct nd {
			ll s, w, a, b;
			atl::uint32 l = 0, r = 0;
		};

		std::vector<nd> ns;
		std::size_t n;

		explicit dynamic_lazy_segment_tree_by_materialized_push(std::size_t n_) : n(n_) {
			ns.reserve(1 << 22);
			ns.push_back({0, 0, 1, 0, 0, 0});
			ns.push_back({0, ll(n_), 1, 0, 0, 0});
		}

		atl::uint32 mk(ll w) {
			atl_assert(ns.size() < 0xFFFFFFFFull);
			ns.push_back({0, w, 1, 0, 0, 0});
			return atl::uint32(ns.size() - 1);
		}

		void push(atl::uint32 x, std::size_t lo, std::size_t mid, std::size_t hi) {
			if (ns[x].l == 0) {
				auto c = mk(ll(mid - lo));
				ns[x].l = c;
			}
			if (ns[x].r == 0) {
				auto c = mk(ll(hi - mid));
				ns[x].r = c;
			}
			nd &d = ns[x];
			if (d.a == 1 && d.b == 0) return;
			for (auto c : {d.l, d.r}) {
				nd &ch = ns[c];
				ch.s = (d.a * ch.s + d.b % mod * (ch.w % mod)) % mod;
				ch.a = d.a * ch.a % mod;
				ch.b = (d.a * ch.b + d.b) % mod;
			}
			d.a = 1;
			d.b = 0;
		}

		void apply(atl::uint32 x, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r, ll a, ll b) {
			if (l <= lo && hi <= r) {
				nd &d = ns[x];
				d.s = (a * d.s + b % mod * (d.w % mod)) % mod;
				d.a = a * d.a % mod;
				d.b = (a * d.b + b) % mod;
				return;
			}
			std::size_t const mid = lo + (hi - lo) / 2;
			push(x, lo, mid, hi);
			if (l < mid) apply(ns[x].l, lo, mid, l, r, a, b);
			if (r > mid) apply(ns[x].r, mid, hi, l, r, a, b);
			ns[x].s = (ns[ns[x].l].s + ns[ns[x].r].s) % mod;
		}

		ll fold(atl::uint32 x, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r, ll aa, ll ab) const {
			if (r <= lo || hi <= l) return 0;
			std::size_t const olo = lo < l ? l : lo, ohi = hi < r ? hi : r;
			if (x == 0) return ab % mod * (ll(ohi - olo) % mod) % mod;
			nd const &d = ns[x];
			if (l <= lo && hi <= r) return (aa * d.s + ab % mod * (d.w % mod)) % mod;
			ll const na = aa * d.a % mod, nb = (aa * d.b + ab) % mod;
			std::size_t const mid = lo + (hi - lo) / 2;
			return (fold(d.l, lo, mid, l, r, na, nb) + fold(d.r, mid, hi, l, r, na, nb)) % mod;
		}
	};

	BENCHMARK("by materialized push") {
		dynamic_lazy_segment_tree_by_materialized_push t(domain);
		ll acc = 0;
		for (auto const &s : steps)
			if (s.is_apply)
				t.apply(1, 0, domain, s.l, s.r, s.f.a, s.f.b);
			else
				acc += t.fold(1, 0, domain, s.l, s.r, 1, 0);
		return acc;
	};

	BENCHMARK("library over [0, 1e9)") {
		atl::dynamic_lazy_segment_tree t(domain, item{0, 0}, aff{1, 0}, op, mp, cp, fill);
		t.reserve(q / 4);
		ll acc = 0;
		for (auto const &s : steps)
			if (s.is_apply)
				t.apply(s.l, s.r, s.f);
			else
				acc += t.fold(s.l, s.r).s;
		return acc;
	};

	// offline alternative: compress all operation boundaries into segments
	std::vector<std::size_t> cuts{0, domain};
	for (auto const &s : steps) {
		cuts.push_back(s.l);
		cuts.push_back(s.r);
	}
	std::ranges::sort(cuts);
	cuts.erase(std::ranges::unique(cuts).begin(), cuts.end());
	auto seg_of = [&](std::size_t p) { return std::ranges::lower_bound(cuts, p) - cuts.begin(); };
	BENCHMARK("by offline compression") {
		std::vector<item> init(cuts.size() - 1);
		for (std::size_t i = 0; i + 1 < cuts.size(); ++i) init[i] = {0, ll(cuts[i + 1] - cuts[i])};
		atl::lazy_segment_tree t(std::move(init), item{0, 0}, aff{1, 0}, op, mp, cp);
		ll acc = 0;
		for (auto const &s : steps)
			if (s.is_apply)
				t.apply(seg_of(s.l), seg_of(s.r), s.f);
			else
				acc += t.fold(seg_of(s.l), seg_of(s.r)).s;
		return acc;
	};

	{
		atl::dynamic_lazy_segment_tree t(domain, item{0, 0}, aff{1, 0}, op, mp, cp, fill);
		for (auto const &s : steps)
			if (s.is_apply) t.apply(s.l, s.r, s.f);
		std::vector<std::size_t> reads(1'000'000);
		for (auto &r : reads) r = rng() % domain;
		for (std::size_t k = 0; k < 10000; ++k)
			if (t.at(reads[k]).s != t.fold(reads[k], reads[k] + 1).s) FAIL("at disagrees with single-element fold");
		BENCHMARK("1e6 at() over [0, 1e9) after 5e4 applies (library)") {
			ll acc = 0;
			for (auto r : reads) acc += t.at(r).s;
			return acc;
		};
		BENCHMARK("1e6 at() over [0, 1e9) after 5e4 applies (by singleton fold)") {
			ll acc = 0;
			for (auto r : reads) acc += t.fold(r, r + 1).s;
			return acc;
		};
	}
}
