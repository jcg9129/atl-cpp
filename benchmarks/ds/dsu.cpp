/**
 * @file
 * Race between `atl::dsu` (the library, full path compression),
 * `dsu_by_path_halving` (relinks to grandparents), `atl::rollback_dsu`
 * (the library, logs both roots), and `rollback_dsu_by_compact_log`
 * (recovers one root from the child during undo). `atl::persistent_dsu`
 * uses shared path-copying arrays. Loops measure throughput of linear union
 * histories and queries, random and binomial merges, batched rollback, and
 * version-tree traversal; construction is included where performed inside a loop.
 * Result (Xeon Platinum 8280, GCC 15.2, `-O2` and `-O2 -march=native`,
 * 2026-10-15): full compression takes 23.1/20.8 ms on binomial trees versus
 * 25.4/26.6 ms by path halving; compact rollback takes 298/301 ms versus
 * 296/271 ms for the library. Numbers: docs/dsu.md.
 */
#include <atl/ds/dsu.hpp>
#include <atl/ds/persistent_dsu.hpp>
#include <atl/ds/rollback_dsu.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <format>
#include <limits>
#include <random>
#include <utility>
#include <vector>

using atl::uint32;
using atl::uint64;

TEST_CASE("dsu benchmarks: rollback vs persistent", "[dsu]") {
	auto seed = GENERATE(take(1, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);

	std::size_t const n = 200000;
	std::size_t const unions = 200000;
	std::size_t const queries = 1000000;

	std::vector<std::pair<uint32, uint32>> ops(unions), probes(queries);
	for (auto &[a, b] : ops) a = uint32(rng() % n), b = uint32(rng() % n);
	for (auto &[a, b] : probes) a = uint32(rng() % n), b = uint32(rng() % n);

	auto label = std::format("n={}, unions={}", n, unions);

	// A fresh structure each sample: the unite chain is what we are timing.
	BENCHMARK("dsu -- unite chain -- " + label) {
		atl::dsu d(n);
		for (auto [a, b] : ops) d.unite(a, b);
		return d.count();
	};
	BENCHMARK("rollback_dsu -- unite chain -- " + label) {
		atl::rollback_dsu d(n);
		for (auto [a, b] : ops) d.unite(a, b);
		return d.count();
	};
	BENCHMARK("persistent_dsu -- unite chain -- " + label) {
		atl::persistent_dsu<> d(n);
		for (auto [a, b] : ops) d = d.unite(a, b);
		return d.count();
	};

	// Queries on the fully-merged structure (built once, read-only in the body).
	atl::rollback_dsu rb(n);
	for (auto [a, b] : ops) rb.unite(a, b);
	atl::persistent_dsu<> pd(n);
	for (auto [a, b] : ops) pd = pd.unite(a, b);
	{
		atl::dsu ordinary(n);
		for (auto [a, b] : ops) ordinary.unite(a, b);
		REQUIRE(ordinary.count() == rb.count());
		REQUIRE(rb.count() == pd.count());
		for (auto [a, b] : probes) {
			REQUIRE(rb.same(a, b) == pd.same(a, b));
			REQUIRE(ordinary.same(a, b) == rb.same(a, b));
		}
	}
	BENCHMARK(std::format("rollback_dsu -- {} same() queries", queries)) {
		uint32 acc = 0;
		for (auto [a, b] : probes) acc += rb.same(a, b);
		return acc;
	};
	BENCHMARK(std::format("persistent_dsu -- {} same() queries", queries)) {
		uint32 acc = 0;
		for (auto [a, b] : probes) acc += pd.same(a, b);
		return acc;
	};
}

namespace {
	/** @brief Compact-log contender: rollback_dsu with an 8-byte log entry, the surviving root
	 *         re-read from the child's parent slot in undo. */
	class rollback_dsu_by_compact_log {
		struct entry {
			uint32 child, child_size;
		};

		std::vector<atl::int32> p_;
		std::vector<entry> history_;

	public:
		explicit rollback_dsu_by_compact_log(std::size_t n) : p_(n, -1) { history_.reserve(n); }

		std::size_t find(std::size_t x) const {
			while (p_[x] >= 0) x = std::size_t(p_[x]);
			return x;
		}

		bool same(std::size_t a, std::size_t b) const { return find(a) == find(b); }

		bool unite(std::size_t a, std::size_t b) {
			std::size_t ra = find(a), rb = find(b);
			if (ra == rb) return false;
			atl::int32 sa = -p_[ra], sb = -p_[rb];
			if (sa < sb) std::swap(ra, rb), std::swap(sa, sb);
			history_.push_back({uint32(rb), uint32(sb)});
			p_[rb] = atl::int32(ra);
			p_[ra] = -(sa + sb);
			return true;
		}

		std::size_t time() const { return history_.size(); }

		void rollback(std::size_t t) {
			while (history_.size() > t) {
				auto const [child, child_size] = history_.back();
				history_.pop_back();
				p_[std::size_t(p_[child])] += atl::int32(child_size);
				p_[child] = -atl::int32(child_size);
			}
		}
	};

	/** @brief Find representatives by path halving instead of full compression. */
	class dsu_by_path_halving {
		std::vector<atl::int32> p_;

	public:
		explicit dsu_by_path_halving(std::size_t n) : p_(n, -1) {}

		std::size_t find(std::size_t x) {
			while (p_[x] >= 0) {
				std::size_t const p = std::size_t(p_[x]);
				if (p_[p] >= 0) p_[x] = p_[p];
				x = std::size_t(p_[x]);
			}
			return x;
		}

		bool same(std::size_t a, std::size_t b) { return find(a) == find(b); }

		bool unite(std::size_t a, std::size_t b) {
			std::size_t ra = find(a), rb = find(b);
			if (ra == rb) return false;
			if (-p_[ra] < -p_[rb]) std::swap(ra, rb);
			p_[ra] += p_[rb];
			p_[rb] = atl::int32(ra);
			return true;
		}
	};

	/**
	 * @brief Version-tree DFS: node v adds edge (a_v, b_v) on top of its parent's version;
	 *        a rollback DSU walks the tree depth-first, undoing on the way back, and
	 *        answers one same() probe per node.
	 */
	template<class Dsu>
		requires std::constructible_from<Dsu, std::size_t> && requires(Dsu &d, std::size_t i) {
			{ d.time() } -> std::convertible_to<std::size_t>;
			d.rollback(i);
			{ d.unite(i, i) } -> std::convertible_to<bool>;
			{ d.same(i, i) } -> std::convertible_to<bool>;
		}
	uint64 sum_version_connectivity_by_rollback(std::size_t n, std::vector<uint32> const &parent, std::vector<std::pair<uint32, uint32>> const &edge,
												std::vector<std::pair<uint32, uint32>> const &probe) {
		std::size_t const q = parent.size();
		std::vector<uint32> head(q + 1, uint32(-1)), next(q);
		for (std::size_t v = q; v-- > 1;) next[v] = head[parent[v]], head[parent[v]] = uint32(v);
		Dsu d(n);
		uint64 acc = 0;
		std::vector<std::pair<uint32, std::size_t>> stack{{0, 0}};
		while (!stack.empty()) {
			auto &[v, t] = stack.back();
			if (t == 0) { // entering v: its own edge, then its probe
				t = d.time() + 1;
				d.unite(edge[v].first, edge[v].second);
				acc += d.same(probe[v].first, probe[v].second);
			}
			uint32 const c = head[v];
			if (c != uint32(-1)) {
				head[v] = next[c];
				stack.push_back({c, 0});
			} else {
				d.rollback(t - 1);
				stack.pop_back();
			}
		}
		return acc;
	}

	/** @brief The same version tree with persistent_dsu, one version per node. */
	uint64 sum_version_connectivity_by_persistent_versions(std::size_t n, std::vector<uint32> const &parent, std::vector<std::pair<uint32, uint32>> const &edge,
														   std::vector<std::pair<uint32, uint32>> const &probe) {
		std::vector<atl::persistent_dsu<>> ver;
		ver.reserve(parent.size());
		uint64 acc = 0;
		for (std::size_t v = 0; v < parent.size(); ++v) {
			ver.push_back((v == 0 ? atl::persistent_dsu<>(n) : ver[parent[v]]).unite(edge[v].first, edge[v].second));
			acc += ver[v].same(probe[v].first, probe[v].second);
		}
		return acc;
	}
}

// Recovering the surviving root adds a dependent load: batched rollback costs
// 298/301 ms by compact log versus 296/271 ms for the library (file record).
TEST_CASE("dsu compression and rollback log comparisons", "[dsu][race]") {
	std::mt19937_64 rng(31);
	{
		constexpr std::size_t n = 1'000'000, q = 1'000'000;
		std::vector<std::pair<uint32, uint32>> random_unions(n), probes(q);
		for (auto &[a, b] : random_unions) a = uint32(rng() % n), b = uint32(rng() % n);
		for (auto &[a, b] : probes) a = uint32(rng() % n), b = uint32(rng() % n);
		{
			atl::dsu reference(n);
			dsu_by_path_halving contender(n);
			for (auto [a, b] : random_unions) REQUIRE(reference.unite(a, b) == contender.unite(a, b));
			for (auto [a, b] : probes) REQUIRE(reference.same(a, b) == contender.same(a, b));
			atl::dsu tree_reference(std::size_t(1) << 20);
			dsu_by_path_halving tree_contender(std::size_t(1) << 20);
			for (std::size_t step = 1; step < (std::size_t(1) << 20); step *= 2)
				for (std::size_t i = 0; i + step < (std::size_t(1) << 20); i += 2 * step)
					REQUIRE(tree_reference.unite(i, i + step) == tree_contender.unite(i, i + step));
			for (auto [a, b] : probes) REQUIRE(tree_reference.same(a & 0xFFFFF, b & 0xFFFFF) == tree_contender.same(a & 0xFFFFF, b & 0xFFFFF));
		}
		BENCHMARK("n=1e6: 1e6 random unite + 1e6 same, full compression (library)") {
			atl::dsu d(n);
			uint64 acc = 0;
			for (auto [a, b] : random_unions) d.unite(a, b);
			for (auto [a, b] : probes) acc += d.same(a, b);
			return acc;
		};
		BENCHMARK("n=1e6: 1e6 random unite + 1e6 same, by path halving") {
			dsu_by_path_halving d(n);
			uint64 acc = 0;
			for (auto [a, b] : random_unions) d.unite(a, b);
			for (auto [a, b] : probes) acc += d.same(a, b);
			return acc;
		};
		// pairs, then pairs of pairs, ...: the deepest trees union by size allows
		BENCHMARK("n=2^20: binomial-merge chain + 1e6 same, full compression (library)") {
			atl::dsu d(std::size_t(1) << 20);
			uint64 acc = 0;
			for (std::size_t s = 1; s < (std::size_t(1) << 20); s *= 2)
				for (std::size_t i = 0; i + s < (std::size_t(1) << 20); i += 2 * s) d.unite(i, i + s);
			for (auto [a, b] : probes) acc += d.same(a & 0xFFFFF, b & 0xFFFFF);
			return acc;
		};
		BENCHMARK("n=2^20: binomial-merge chain + 1e6 same, by path halving") {
			dsu_by_path_halving d(std::size_t(1) << 20);
			uint64 acc = 0;
			for (std::size_t s = 1; s < (std::size_t(1) << 20); s *= 2)
				for (std::size_t i = 0; i + s < (std::size_t(1) << 20); i += 2 * s) d.unite(i, i + s);
			for (auto [a, b] : probes) acc += d.same(a & 0xFFFFF, b & 0xFFFFF);
			return acc;
		};
	}
	{
		// tight unite/rollback loop: 1e7 unite attempts in batches of 64, each batch rolled back
		constexpr std::size_t n = 200'000, m = 10'000'000;
		std::vector<std::pair<uint32, uint32>> ops(m);
		for (auto &[a, b] : ops) a = uint32(rng() % n), b = uint32(rng() % n);
		std::vector<std::pair<uint32, uint32>> base(n / 2);
		for (auto &[a, b] : base) a = uint32(rng() % n), b = uint32(rng() % n);
		{
			atl::rollback_dsu reference(n);
			rollback_dsu_by_compact_log contender(n);
			for (auto [a, b] : base) REQUIRE(reference.unite(a, b) == contender.unite(a, b));
			auto const saved = reference.time();
			for (std::size_t i = 0; i < m; i += 64) {
				for (std::size_t j = i; j < i + 64; ++j) REQUIRE(reference.unite(ops[j].first, ops[j].second) == contender.unite(ops[j].first, ops[j].second));
				reference.rollback(saved);
				contender.rollback(saved);
				REQUIRE(reference.time() == contender.time());
				for (std::size_t j = i; j < i + 64; ++j) REQUIRE(reference.same(ops[j].first, ops[j].second) == contender.same(ops[j].first, ops[j].second));
			}
		}
		BENCHMARK("1e7 unite in batches of 64 + rollback, 12-byte entry (library)") {
			atl::rollback_dsu d(n);
			for (auto [a, b] : base) d.unite(a, b);
			std::size_t const t = d.time();
			for (std::size_t i = 0; i < m; i += 64) {
				for (std::size_t j = i; j < i + 64; ++j) d.unite(ops[j].first, ops[j].second);
				d.rollback(t);
			}
			return d.count();
		};
		BENCHMARK("1e7 unite in batches of 64 + rollback, by compact log") {
			rollback_dsu_by_compact_log d(n);
			for (auto [a, b] : base) d.unite(a, b);
			std::size_t const t = d.time();
			for (std::size_t i = 0; i < m; i += 64) {
				for (std::size_t j = i; j < i + 64; ++j) d.unite(ops[j].first, ops[j].second);
				d.rollback(t);
			}
			return d.time();
		};
	}
	{
		// version tree, n = q = 2e5: each node's parent is a random earlier node
		constexpr std::size_t n = 200'000, q = 200'000;
		std::vector<uint32> parent(q);
		std::vector<std::pair<uint32, uint32>> edge(q), probe(q);
		for (std::size_t v = 1; v < q; ++v) parent[v] = uint32(v - 1 - rng() % std::min<std::size_t>(v, 8));
		for (auto &[a, b] : edge) a = uint32(rng() % n), b = uint32(rng() % n);
		for (auto &[a, b] : probe) a = uint32(rng() % n), b = uint32(rng() % n);
		REQUIRE(sum_version_connectivity_by_rollback<atl::rollback_dsu>(n, parent, edge, probe) == sum_version_connectivity_by_persistent_versions(n, parent, edge, probe));
		REQUIRE(sum_version_connectivity_by_rollback<atl::rollback_dsu>(n, parent, edge, probe) == sum_version_connectivity_by_rollback<rollback_dsu_by_compact_log>(n, parent, edge, probe));
		BENCHMARK("version-tree DFS n=q=2e5, rollback_dsu 12-byte entry (library)") {
			return sum_version_connectivity_by_rollback<atl::rollback_dsu>(n, parent, edge, probe);
		};
		BENCHMARK("version-tree DFS n=q=2e5, rollback_dsu by compact log") {
			return sum_version_connectivity_by_rollback<rollback_dsu_by_compact_log>(n, parent, edge, probe);
		};
		BENCHMARK("version tree n=q=2e5, persistent_dsu") { return sum_version_connectivity_by_persistent_versions(n, parent, edge, probe); };
	}
}
