/** @file
 * @brief Batch throughput of Pohlig-Hellman versus whole-group baby-step
 * giant-step, each with Barrett32, Barrett64 or Montgomery products. Shapes:
 * 100 smooth-prime queries near 1e9 and 4e18, 20 safe-prime queries near 1e9,
 * five semi-smooth-prime queries near 1e18. Whole-group tables near 1e18 are
 * infeasible, so those shapes race Pohlig-Hellman engines only.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, 2026-10-15, -O2 / -O2
 * -march=native: library smooth 998244353 1.1/1.1 ms versus Barrett32 BSGS
 * 76.3/79.8 ms. Full table: docs/discrete-log.md.
 */
#include <atl/experimental/nt/discrete_log_variants.hpp>
#include <atl/nt/discrete_log.hpp>
#include <atl/nt/modular.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using atl::uint64;

namespace {
	struct instance {
		uint64 a, b;
	};

	std::vector<instance> make_instances(uint64 p, int count) {
		atl_assert(count >= 0);
		std::mt19937_64 rng(p);
		uint64 const g = atl::primitive_root(p);
		std::vector<instance> v(static_cast<std::size_t>(count));
		for (auto &q : v) q = {g, atl::pow_mod(g, rng() % (p - 1), p)};
		return v;
	}

	/** @brief Adapt a fixed strategy and engine to the non-coprime reduction signature. */
	template<atl::detail::modular_multiplier Engine, bool UsesPohligHellman>
		requires std::constructible_from<Engine, uint64>
	std::optional<uint64> discrete_log_with_engine(uint64 a, uint64 b, uint64 m) {
		return atl::detail::discrete_log_reduced(a, b, m, [](uint64 base, uint64 target, uint64 mod) {
			Engine const engine(mod);
			if constexpr (UsesPohligHellman)
				return atl::detail::pohlig_hellman(engine, engine.to_form(base), engine.to_form(target));
			else
				return atl::experimental::discrete_log_by_baby_step_giant_step(
					engine, engine.to_form(base), engine.to_form(target));
		});
	}

	// a prime near 1e18 whose p - 1 = 2·q·k has a prime q near 1e12: the giant steps
	// are sqrt(1e12) = 1e6 and BSGS over the whole group (1e9) is out of reach
	uint64 semi_smooth_prime() {
		uint64 q = 1'000'000'000'000;
		while (!atl::is_prime(q)) ++q;
		for (uint64 k = 500'000;; ++k)
			if (atl::is_prime(2 * q * k + 1)) return 2 * q * k + 1;
	}
}

TEST_CASE("discrete_log: strategy and engine", "[discrete_log]") {
	using barrett32 = atl::detail::barrett_multiplier<atl::barrett>;
	using barrett64 = atl::detail::barrett_multiplier<atl::barrett64>;
	using montgomery = atl::detail::montgomery64;

	auto const race = [](std::string const &label, uint64 p, int count, bool is_bsgs_feasible) {
		auto const instances = make_instances(p, count);
		for (auto const &q : instances) REQUIRE(atl::pow_mod(q.a, *atl::discrete_log(q.a, q.b, p), p) == q.b);
		auto const run = [&]<typename Solver>(Solver solver)
			requires std::is_invocable_r_v<std::optional<uint64>, Solver &, uint64, uint64, uint64>
		{
			uint64 acc = 0;
			for (auto const &q : instances) acc += *solver(q.a, q.b, p);
			return acc;
		};
		for (auto const &q : instances) {
			auto const want = atl::discrete_log(q.a, q.b, p);
			REQUIRE(discrete_log_with_engine<barrett64, true>(q.a, q.b, p) == want);
			REQUIRE(discrete_log_with_engine<montgomery, true>(q.a, q.b, p) == want);
			if (p < (uint64(1) << 32)) REQUIRE(discrete_log_with_engine<barrett32, true>(q.a, q.b, p) == want);
			if (is_bsgs_feasible) {
				REQUIRE(discrete_log_with_engine<barrett64, false>(q.a, q.b, p) == want);
				REQUIRE(discrete_log_with_engine<montgomery, false>(q.a, q.b, p) == want);
				if (p < (uint64(1) << 32)) REQUIRE(discrete_log_with_engine<barrett32, false>(q.a, q.b, p) == want);
			}
		}

		BENCHMARK("library " + label) {
			return run([](uint64 a, uint64 b, uint64 m) { return atl::discrete_log(a, b, m); });
		};
		if (p < (uint64(1) << 32)) {
			BENCHMARK("by pohlig hellman barrett32 " + label) {
				return run([](uint64 a, uint64 b, uint64 m) {
					return discrete_log_with_engine<barrett32, true>(a, b, m);
				});
			};
			if (is_bsgs_feasible) BENCHMARK("by baby step giant step barrett32 " + label) {
					return run([](uint64 a, uint64 b, uint64 m) {
						return discrete_log_with_engine<barrett32, false>(a, b, m);
					});
				};
		}
		BENCHMARK("by pohlig hellman barrett64 " + label) {
			return run([](uint64 a, uint64 b, uint64 m) { return discrete_log_with_engine<barrett64, true>(a, b, m); });
		};
		BENCHMARK("by pohlig hellman montgomery " + label) {
			return run([](uint64 a, uint64 b, uint64 m) {
				return discrete_log_with_engine<montgomery, true>(a, b, m);
			});
		};
		if (is_bsgs_feasible) {
			BENCHMARK("by baby step giant step barrett64 " + label) {
				return run([](uint64 a, uint64 b, uint64 m) {
					return discrete_log_with_engine<barrett64, false>(a, b, m);
				});
			};
			BENCHMARK("by baby step giant step montgomery " + label) {
				return run([](uint64 a, uint64 b, uint64 m) {
					return discrete_log_with_engine<montgomery, false>(a, b, m);
				});
			};
		}
	};
	race("smooth 998244353 x100", 998244353, 100, true);
	race("safe 1e9+7 x20", 1'000'000'007, 20, true);
	race("smooth 29*2^57+1 x100", 4179340454199820289ull, 100, false);
	race("semi-smooth ~1e18 (q~1e12) x5", semi_smooth_prime(), 5, false);
}
