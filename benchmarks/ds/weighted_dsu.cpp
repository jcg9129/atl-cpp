/**
 * @file
 * Race between `atl::weighted_dsu` (the library, recursive compression and layout
 * selected by potential width) and `weighted_dsu_by_compression_and_layout`
 * (two-pass buffer, two-pass inverse, halving, splitting, or recursion; split or
 * interleaved nodes; optional root-child exit). Rollback compares the library's
 * width-selected layout with `rollback_weighted_dsu_by_interleaved_nodes`.
 * Plain DSU gives a connectivity-only baseline. Loops measure throughput of
 * constraint/query streams on modular-additive, SL2, random integer, binomial-tree,
 * and parity shapes.
 * Result (Xeon Platinum 8280, GCC 15.2, `-O2` and `-O2 -march=native`,
 * 2026-11-01): wide-potential random rollback takes 103/103 ms by interleaved
 * nodes versus 84.1/86.2 ms by split nodes. The current library uses split nodes
 * for this shape. The 2026-10-24 modular/SL2 compression race measured the library
 * at 3.6/3.7 and 7.5/7.4 ms (plain/native), respectively.
 * Numbers and record-to-current mapping: docs/weighted-dsu.md.
 */
#include <atl/ds/dsu.hpp>
#include <atl/ds/rollback_weighted_dsu.hpp>
#include <atl/ds/weighted_dsu.hpp>
#include <atl/experimental/ds/weighted_dsu_variants.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <vector>

using atl::int64;
using atl::uint32;
using atl::uint64;

namespace {
	using mint = atl::mod_int32<998244353>;

	/** @brief SL(2) over Z/998244353, a non-commutative group. */
	struct sl2 {
		mint a, b, c, d;
		constexpr bool operator==(sl2 const &) const = default;
	};

	/** @brief The matrix product. */
	struct sl2_multiply {
		constexpr sl2 operator()(sl2 const &x, sl2 const &y) const {
			return {x.a * y.a + x.b * y.c, x.a * y.b + x.b * y.d, x.c * y.a + x.d * y.c, x.c * y.b + x.d * y.d};
		}
	};

	/** @brief The adjugate, which is the inverse at determinant 1. */
	struct sl2_inverse {
		constexpr sl2 operator()(sl2 const &x) const { return {x.d, -x.b, -x.c, x.a}; }
	};

	/** @brief Z/2 addition on uint32. */
	struct bit_xor_u32 {
		constexpr uint32 operator()(uint32 a, uint32 b) const { return a ^ b; }
	};

	/** @brief Z/2 is its own inverse. */
	struct self_inverse {
		constexpr uint32 operator()(uint32 x) const { return x; }
	};

	/** @brief Folds an answer into the checksum that keeps the replay from being optimized out. */
	constexpr uint64 digest(int64 x) { return uint64(x); }

	constexpr uint64 digest(uint32 x) { return x; }

	constexpr uint64 digest(mint x) { return x.value(); }

	constexpr uint64 digest(sl2 const &x) { return x.a.value() ^ x.d.value(); }

	/** @brief One operation of a stream: a constraint `unite(a, b, w)` or a `difference(a, b)` query. */
	template<std::copyable Potential>
	struct operation {
		bool is_unite;
		uint32 a, b;
		Potential w;
	};

	template<std::copyable Potential>
	struct workload {
		std::size_t n;
		Potential identity;
		std::vector<operation<Potential>> ops;
	};

	/**
	 * @brief The random potential-constraint workload: random pairs, half constraints (9 in 10
	 *        drawn from a hidden assignment, so most of the late ones are consistent), half queries.
	 */
	template<std::semiregular Potential, atl::binary_operation<Potential> Op, atl::inverse_operation<Potential> Inverse,
			 typename Sampler>
		requires std::invocable<Sampler &, std::mt19937_64 &>
			  && std::convertible_to<std::invoke_result_t<Sampler &, std::mt19937_64 &>, Potential>
	workload<Potential> random_mix(
		std::size_t n, std::size_t q, Potential identity, Op op, Inverse inverse, Sampler sample,
		std::mt19937_64 &rng) {
		std::vector<Potential> hidden(n);
		for (auto &h : hidden) h = sample(rng);
		workload<Potential> w{n, identity, {}};
		w.ops.reserve(q);
		for (std::size_t i = 0; i < q; ++i) {
			uint32 const a = uint32(rng() % n), b = uint32(rng() % n);
			if (rng() & 1)
				w.ops.push_back(
					{true, a, b,
					 rng() % 10
						 ? Potential(std::invoke(op, std::invoke(inverse, hidden[a]), hidden[b]))
						 : sample(rng)});
			else
				w.ops.push_back({false, a, b, identity});
		}
		return w;
	}

	/** @brief Pairs, then pairs of pairs, ... (the deepest trees union by size allows), then queries. */
	workload<int64> binomial(std::size_t log_n, std::size_t q, std::mt19937_64 &rng) {
		std::size_t const n = std::size_t(1) << log_n;
		workload<int64> w{n, 0, {}};
		for (std::size_t s = 1; s < n; s *= 2)
			for (std::size_t i = 0; i + s < n; i += 2 * s)
				w.ops.push_back({true, uint32(i), uint32(i + s), int64(rng() % 1000)});
		for (std::size_t i = 0; i < q; ++i) w.ops.push_back({false, uint32(rng() % n), uint32(rng() % n), 0});
		return w;
	}

	template<typename Dsu, std::copyable Potential>
		requires requires(Dsu &d, Potential const &w) {
			{ d.unite(0, 0, w) } -> std::same_as<atl::unite_outcome>;
			{ d.difference(0, 0) } -> std::same_as<std::optional<Potential>>;
			{ digest(w) } -> std::convertible_to<uint64>;
		}
	uint64 replay(Dsu d, workload<Potential> const &w) {
		uint64 acc = 0;
		for (auto const &op : w.ops) {
			if (op.is_unite)
				acc += uint64(d.unite(op.a, op.b, op.w));
			else if (auto const r = d.difference(op.a, op.b))
				acc += digest(*r);
		}
		return acc;
	}

	/** @brief The baseline: plain dsu on the same unites, same() for each query. */
	template<std::copyable Potential>
	uint64 replay_plain(workload<Potential> const &w) {
		atl::dsu d(w.n);
		uint64 acc = 0;
		for (auto const &op : w.ops) acc += op.is_unite ? uint64(d.unite(op.a, op.b)) : uint64(d.same(op.a, op.b));
		return acc;
	}

	using atl::experimental::weighted_dsu_by_compression_and_layout;
	using atl::experimental::weighted_dsu_compression;
	using atl::experimental::weighted_dsu_layout;

	/** @brief Every contender and the plain-dsu baseline on one workload. */
	template<std::copyable Potential, atl::binary_operation<Potential> Op, atl::inverse_operation<Potential> Inverse>
		requires std::equality_comparable<Potential> && std::copy_constructible<Op> && std::copy_constructible<Inverse>
			  && requires(Potential const &p) { { digest(p) } -> std::convertible_to<uint64>; }
	void benchmark_weighted_differences(
		std::string const &shape, workload<Potential> const &w, Op op, Inverse inverse) {
		{
			atl::dsu connectivity(w.n);
			atl::weighted_dsu<Potential, Op, Inverse> reference(w.n, w.identity, op, inverse);
			for (auto const &item : w.ops) {
				if (item.is_unite) {
					connectivity.unite(item.a, item.b);
					reference.unite(item.a, item.b, item.w);
				} else
					REQUIRE(connectivity.same(item.a, item.b) == reference.same(item.a, item.b));
			}
		}
		auto const benchmark_method = [&]<weighted_dsu_compression Compression, weighted_dsu_layout Layout,
										  bool HasRootChildExit = true>(std::string const &name) {
			using contender = weighted_dsu_by_compression_and_layout<
				Compression, Layout, Potential, Op, Inverse, HasRootChildExit>;
			REQUIRE(replay(contender(w.n, w.identity, op, inverse), w)
					== replay(atl::weighted_dsu<Potential, Op, Inverse>(w.n, w.identity, op, inverse), w));
			BENCHMARK(shape + " -- " + name) {
				return replay(contender(w.n, w.identity, op, inverse), w);
			};
		};
		using enum weighted_dsu_compression;
		constexpr auto interleaved = weighted_dsu_layout::interleaved, split = weighted_dsu_layout::split;
		benchmark_method.template operator()<two_pass_buffer, interleaved>(
			"by compression and layout (two-pass buffer), interleaved");
		benchmark_method.template operator()<two_pass_buffer, split>(
			"by compression and layout (two-pass buffer), split");
		benchmark_method.template operator()<two_pass_inverse, interleaved>(
			"by compression and layout (two-pass inverse), interleaved");
		benchmark_method.template operator()<two_pass_inverse, split>(
			"by compression and layout (two-pass inverse), split");
		benchmark_method.template operator()<halving, interleaved>(
			"by compression and layout (halving), interleaved");
		benchmark_method.template operator()<halving, split>(
			"by compression and layout (halving), split");
		benchmark_method.template operator()<recursive, interleaved>(
			"by compression and layout (recursive), interleaved");
		benchmark_method.template operator()<recursive, split>(
			"by compression and layout (recursive), split");
		benchmark_method.template operator()<splitting, interleaved>(
			"by compression and layout (splitting), interleaved");
		benchmark_method.template operator()<splitting, split>(
			"by compression and layout (splitting), split");
		benchmark_method.template operator()<two_pass_buffer, interleaved, false>(
			"by compression and layout (two-pass buffer), interleaved, no root-child exit");
		benchmark_method.template operator()<two_pass_inverse, interleaved, false>(
			"by compression and layout (two-pass inverse), interleaved, no root-child exit");
		benchmark_method.template operator()<halving, interleaved, false>(
			"by compression and layout (halving), interleaved, no root-child exit");
		benchmark_method.template operator()<splitting, interleaved, false>(
			"by compression and layout (splitting), interleaved, no root-child exit");
		benchmark_method.template operator()<recursive, interleaved, false>(
			"by compression and layout (recursive), interleaved, no root-child exit");
		BENCHMARK(shape + " -- weighted_dsu (library, layout by potential width)") {
			return replay(atl::weighted_dsu<Potential, Op, Inverse>(w.n, w.identity, op, inverse), w);
		};
		BENCHMARK(shape + " -- baseline: plain dsu, same unites") { return replay_plain(w); };
	}

	/** @brief The library rollback class against by interleaved nodes, on one workload. */
	template<std::copyable Potential, atl::binary_operation<Potential> Op, atl::inverse_operation<Potential> Inverse>
		requires std::equality_comparable<Potential> && std::copy_constructible<Op> && std::copy_constructible<Inverse>
			  && requires(Potential const &p) { { digest(p) } -> std::convertible_to<uint64>; }
	void benchmark_rollback_differences(
		std::string const &shape, workload<Potential> const &w, Op op, Inverse inverse) {
		using contender = atl::experimental::rollback_weighted_dsu_by_interleaved_nodes<Potential, Op, Inverse>;
		REQUIRE(replay(contender(w.n, w.identity, op, inverse), w)
				== replay(atl::rollback_weighted_dsu<Potential, Op, Inverse>(w.n, w.identity, op, inverse), w));
		BENCHMARK(shape + " -- rollback_weighted_dsu (library: layout by width)") {
			return replay(atl::rollback_weighted_dsu<Potential, Op, Inverse>(w.n, w.identity, op, inverse), w);
		};
		BENCHMARK(shape + " -- rollback_weighted_dsu by interleaved nodes") {
			return replay(contender(w.n, w.identity, op, inverse), w);
		};
	}
}

namespace {
	/**
	 * @brief Builds the five shapes of the benchmark_weighted_differences (fixed seeds) and hands each to
	 * `run_race(name, workload, op, inverse)`.
	 */
	template<typename Race>
		requires std::invocable<Race const &, std::string const &, workload<int64> const &, std::plus<>, std::negate<>>
			  && std::invocable<Race const &, std::string const &, workload<uint32> const &, bit_xor_u32, self_inverse>
			  && std::invocable<Race const &, std::string const &, workload<mint> const &, std::plus<>, std::negate<>>
			  && std::invocable<Race const &, std::string const &, workload<sl2> const &, sl2_multiply, sl2_inverse>
	void for_each_shape(Race const &run_race) {
		std::mt19937_64 rng(2026'10'24);
		{
			auto const w = random_mix<mint>(
				200'000, 200'000, mint(0), std::plus<>{}, std::negate<>{},
				[](std::mt19937_64 &g) { return mint(g()); }, rng);
			run_race("random constraints n=q=2e5, mod 998244353 +", w, std::plus<>{}, std::negate<>{});
		}

		{
			auto const w = random_mix<int64>(
				1'000'000, 2'000'000, 0, std::plus<>{}, std::negate<>{},
				[](std::mt19937_64 &g) { return int64(g() % 1'000'000'000); }, rng);
			run_race("random mix n=1e6, 2e6 ops, int64 +", w, std::plus<>{}, std::negate<>{});
		}
		{
			auto const w = binomial(20, 1'000'000, rng);
			run_race("binomial trees n=2^20 + 1e6 queries, int64 +", w, std::plus<>{}, std::negate<>{});
		}

		{
			auto const w = random_mix<sl2>(
				200'000, 200'000, sl2{1, 0, 0, 1}, sl2_multiply{}, sl2_inverse{},
				[](std::mt19937_64 &g) {
					mint const s(g()), t(g());
					return sl2_multiply{}(sl2{1, s, 0, 1}, sl2{1, 0, t, 1});
				},
				rng);
			run_race("random non-commutative constraints n=q=2e5, SL2 mod 998244353", w, sl2_multiply{}, sl2_inverse{});
		}

		{
			// bipartiteness: every constraint says "different sides", so cycles contradict when odd
			auto w = random_mix<uint32>(
				1'000'000, 2'000'000, 0, bit_xor_u32{}, self_inverse{},
				[](std::mt19937_64 &g) { return uint32(g() & 1); }, rng);
			for (auto &op : w.ops) op.w = op.is_unite ? 1 : 0;
			run_race("Z/2 parity n=1e6, 2e6 ops", w, bit_xor_u32{}, self_inverse{});
		}
	}
}

TEST_CASE("weighted_dsu race", "[weighted_dsu][race]") {
	for_each_shape([]<typename... Args>(Args const &...args)
					   requires requires { benchmark_weighted_differences(args...); }
				   { benchmark_weighted_differences(args...); });
}

TEST_CASE("rollback_weighted_dsu race: layout", "[weighted_dsu][race][rollback]") {
	for_each_shape([]<typename... Args>(Args const &...args)
					   requires requires { benchmark_rollback_differences(args...); }
				   { benchmark_rollback_differences(args...); });
}
