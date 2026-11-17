/**
 * @file
 * @brief Conditional range-action stream throughput, including construction.
 *
 * Compact recursion removes padding; explicit stacks replace recursive failure descent; fixed tags fuse checks; three adds split value classes; aligned or separate arrays change layout; optional mapping copies summaries; unconditional push and no-collapse remove fast paths. Five shapes at n=q=2e5 and 5e5 replay from fresh trees.
 * Xeon Platinum 8358, GCC 15.2, RelWithDebInfo, -O2 (2026-11-02) and -O2 -march=native (2026-11-03): balanced mix n=q=2e5 library 190/194 ms versus generic engine 341/356 ms.
 * Full measurements: docs/segment-tree-beats.md.
 */
#include <atl/ds/segment_tree_beats.hpp>
#include <atl/experimental/ds/segment_tree_beats_variants.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <concepts>

using atl::int64;
using atl::uint32;

namespace {
	/** @brief One operation of a stream. */
	struct beats_op {
		enum kind_t : std::uint8_t { chmin,
									 chmax,
									 add,
									 sum,
									 max,
									 chmin_under_max } kind;

		uint32 l, r;
		int64 x;
	};

	/** @brief A shape: the initial array and the operation stream. */
	struct beats_shape {
		std::string name;
		std::vector<int64> initial;
		std::vector<beats_op> ops;
	};

	/** @brief A uniform integer in `[lo, hi]`. */
	int64 uniform(std::mt19937_64 &rng, int64 lo, int64 hi) { return lo + int64(rng() % std::uint64_t(hi - lo + 1)); }

	/** @brief A random nonempty range of `[0, n)`. */
	std::pair<uint32, uint32> random_range(std::mt19937_64 &rng, std::size_t n) {
		atl_assert(n > 1 && n < (atl::uint64(1) << 32));
		uint32 l = uint32(rng() % n), r = uint32(rng() % n);
		if (l > r) std::swap(l, r);
		return {l, r + 1};
	}

	/**
	 * @brief Balanced operation mix: chmin / chmax / add / sum a
	 *        quarter each, values and clamps in `[-1e12, 1e12]`, adds in `[-1e9, 1e9]` (so
	 *        int64 sums cannot overflow at 5e5).
	 */
	beats_shape balanced_mix(std::size_t n) {
		atl_assert(n > 1 && n < (atl::uint64(1) << 32));
		std::mt19937_64 rng(1);
		beats_shape s{"balanced mix chmin/chmax/add/sum", std::vector<int64>(n), std::vector<beats_op>(n)};
		for (auto &x : s.initial) x = uniform(rng, -1'000'000'000'000, 1'000'000'000'000);
		for (auto &o : s.ops) {
			auto const [l, r] = random_range(rng, n);
			auto const kind = beats_op::kind_t(rng() % 4);
			int64 const x = kind == beats_op::add ? uniform(rng, -1'000'000'000, 1'000'000'000) : uniform(rng, -1'000'000'000'000, 1'000'000'000'000);
			o = {kind, l, r, x};
		}
		return s;
	}

	/** @brief Clamping operation mix: chmin / max / sum a third each, no add (the O(log n) regime). */
	beats_shape clamp_only(std::size_t n) {
		atl_assert(n > 1 && n < (atl::uint64(1) << 32));
		std::mt19937_64 rng(2);
		beats_shape s{"chmin-only chmin/max/sum", std::vector<int64>(n), std::vector<beats_op>(n)};
		for (auto &x : s.initial) x = uniform(rng, 0, (int64(1) << 31) - 1);
		for (auto &o : s.ops) {
			auto const [l, r] = random_range(rng, n);
			auto const roll = rng() % 3;
			o = {roll == 0 ? beats_op::chmin : roll == 1 ? beats_op::max
														 : beats_op::sum,
				 l, r, uniform(rng, 0, (int64(1) << 31) - 1)};
		}
		return s;
	}

	/**
	 * @brief The add-recharge adversary: rounds of add on `[0, m)` and on `[m, n)` at a
	 *        random boundary, a whole-range max, then a whole-range chmin just under it
	 *        (`x = 0` in the op means "max - 1"); the adds keep splitting the classes the
	 *        chmin merges, which forces the log² descents.
	 */
	beats_shape add_recharge(std::size_t n) {
		atl_assert(n > 1 && n < (atl::uint64(1) << 32));
		std::mt19937_64 rng(3);
		beats_shape s{"add-recharge adversary", std::vector<int64>(n), {}};
		for (auto &x : s.initial) x = uniform(rng, 0, 1'000'000);
		while (s.ops.size() + 4 <= n) {
			uint32 const m = uint32(1 + rng() % (n - 1));
			s.ops.push_back({beats_op::add, 0, m, uniform(rng, 1, 3)});
			s.ops.push_back({beats_op::add, m, uint32(n), uniform(rng, 0, 2)});
			s.ops.push_back({beats_op::max, 0, uint32(n), 0});
			s.ops.push_back({beats_op::chmin_under_max, 0, uint32(n), 0});
		}
		return s;
	}

	/**
	 * @brief Collapse: distinct `a[i] = i`, then alternating whole-range chmin from the top
	 *        and chmax from the bottom that meet halfway (deep beats early), after which
	 *        every node is single-valued (the flatten fast path); a whole-range sum after
	 *        each.
	 */
	beats_shape collapse(std::size_t n) {
		atl_assert(n > 1 && n < (atl::uint64(1) << 32));
		beats_shape s{"collapse whole-range chmin/chmax", std::vector<int64>(n), {}};
		for (std::size_t i = 0; i < n; ++i) s.initial[i] = int64(i);
		std::size_t const q = n;
		for (std::size_t t = 0; s.ops.size() + 2 <= q; ++t) {
			int64 const shift = int64(t * n / q);
			if (t % 2 == 0)
				s.ops.push_back({beats_op::chmin, 0, uint32(n), int64(n) - 1 - shift});
			else
				s.ops.push_back({beats_op::chmax, 0, uint32(n), shift});
			s.ops.push_back({beats_op::sum, 0, uint32(n), 0});
		}
		return s;
	}

	/** @brief Query-heavy: 90% sums over ranges of at most 16, 10% balanced-mix updates. */
	beats_shape query_heavy(std::size_t n) {
		atl_assert(n > 1 && n < (atl::uint64(1) << 32));
		std::mt19937_64 rng(5);
		beats_shape s{"query-heavy 90% short sums", std::vector<int64>(n), std::vector<beats_op>(n)};
		for (auto &x : s.initial) x = uniform(rng, -1'000'000'000'000, 1'000'000'000'000);
		for (auto &o : s.ops) {
			if (rng() % 10 != 0) {
				uint32 const l = uint32(rng() % n), r = uint32(std::min<std::size_t>(n, l + 1 + rng() % 16));
				o = {beats_op::sum, l, r, 0};
			} else {
				auto const [l, r] = random_range(rng, n);
				auto const kind = beats_op::kind_t(rng() % 3);
				int64 const x = kind == beats_op::add ? uniform(rng, -1'000'000'000, 1'000'000'000) : uniform(rng, -1'000'000'000'000, 1'000'000'000'000);
				o = {kind, l, r, x};
			}
		}
		return s;
	}

	/** @brief Builds a `Tree` from the shape and runs its stream; returns a checksum of the queries. */
	template<typename Factory>
		requires std::invocable<Factory &, std::vector<int64> const &> && requires(std::invoke_result_t<Factory &, std::vector<int64> const &> &tree, int64 value) {
			tree.chmin(0, 1, value);
			tree.chmax(0, 1, value);
			tree.add(0, 1, value);
			{ tree.sum(0, 1) } -> std::convertible_to<int64>;
			{ tree.max(0, 1) } -> std::convertible_to<int64>;
		}
	atl::uint64 replay_beats(beats_shape const &s, Factory factory) {
		auto tree = std::invoke(factory, s.initial);
		atl::uint64 acc = 0;
		int64 last_max = 0;
		for (auto const &o : s.ops) switch (o.kind) {
			case beats_op::chmin:
				tree.chmin(o.l, o.r, o.x);
				break;
			case beats_op::chmin_under_max:
				tree.chmin(o.l, o.r, last_max - 1);
				break;
			case beats_op::chmax:
				tree.chmax(o.l, o.r, o.x);
				break;
			case beats_op::add:
				tree.add(o.l, o.r, o.x);
				break;
			case beats_op::sum:
				acc += static_cast<atl::uint64>(tree.sum(o.l, o.r));
				break;
			case beats_op::max:
				last_max = tree.max(o.l, o.r);
				acc ^= static_cast<atl::uint64>(last_max);
				break;
			}
		return acc;
	}

	/** @brief Every contender on one shape: checksums must agree, then one row each. */
	void race(beats_shape const &s, std::string const &size) {
		namespace ex = atl::experimental;
		using ex::beats_layout;
		using engine = atl::segment_tree_beats<atl::beats::extremes<int64>, atl::beats::clamp_add<int64>>;
		atl::uint64 const want = replay_beats(s, [](std::vector<int64> const &initial) { return atl::beats_segment_tree<int64>(initial); });
		REQUIRE(replay_beats(s, [](std::vector<int64> const &initial) { return engine(initial); }) == want);
		REQUIRE(replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_recursive_compact_nodes<int64>(initial); }) == want);
		REQUIRE(replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_explicit_stack<int64>(initial); }) == want);
		REQUIRE(replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_fixed_clamp_add<int64>(initial); }) == want);
		REQUIRE(replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_tagless_layout<int64, int64, beats_layout::aos_aligned>(initial); }) == want);
		REQUIRE(replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_tagless_layout<int64, int64, beats_layout::soa>(initial); }) == want);
		REQUIRE(replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_three_adds<int64>(initial); }) == want);
		REQUIRE(replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_optional_mapping<int64>(initial); }) == want);
		REQUIRE(replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_unconditional_push<int64>(initial); }) == want);
		REQUIRE(replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_no_collapse<int64>(initial); }) == want);
		std::string const prefix = s.name + " n=q=" + size + ": ";
		BENCHMARK(prefix + "generic engine") {
			return replay_beats(s, [](std::vector<int64> const &initial) { return engine(initial); });
		};
		BENCHMARK(prefix + "by compact recursion") {
			return replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_recursive_compact_nodes<int64>(initial); });
		};
		BENCHMARK(prefix + "by explicit stack") {
			return replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_explicit_stack<int64>(initial); });
		};
		BENCHMARK(prefix + "by fixed clamp-add") {
			return replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_fixed_clamp_add<int64>(initial); });
		};
		BENCHMARK(prefix + "library") {
			return replay_beats(s, [](std::vector<int64> const &initial) { return atl::beats_segment_tree<int64>(initial); });
		};
		BENCHMARK(prefix + "by aligned nodes") {
			return replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_tagless_layout<int64, int64, beats_layout::aos_aligned>(initial); });
		};
		BENCHMARK(prefix + "by separate arrays") {
			return replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_tagless_layout<int64, int64, beats_layout::soa>(initial); });
		};
		BENCHMARK(prefix + "by three adds") {
			return replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_three_adds<int64>(initial); });
		};
		BENCHMARK(prefix + "by optional mapping") {
			return replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_optional_mapping<int64>(initial); });
		};
		BENCHMARK(prefix + "by unconditional push") {
			return replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_unconditional_push<int64>(initial); });
		};
		BENCHMARK(prefix + "by no collapse") {
			return replay_beats(s, [](std::vector<int64> const &initial) { return ex::beats_segment_tree_by_no_collapse<int64>(initial); });
		};
	}
}

TEST_CASE("segment_tree_beats race n=2e5", "[segment_tree_beats][race]") {
	std::size_t const n = 200'000;
	for (auto const &s : {balanced_mix(n), clamp_only(n), add_recharge(n), collapse(n), query_heavy(n)}) race(s, "2e5");
}

TEST_CASE("segment_tree_beats race n=5e5", "[segment_tree_beats][race]") {
	std::size_t const n = 500'000;
	for (auto const &s : {balanced_mix(n), clamp_only(n), add_recharge(n), collapse(n), query_heavy(n)}) race(s, "5e5");
}
