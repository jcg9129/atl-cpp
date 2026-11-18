#pragma once

/**
 * @file
 * @brief The race loser for the coprime core of atl::discrete_log: plain baby-step
 *        giant-step over the whole residue range. Experimental, not part of the
 *        public atl API.
 *
 * Race (benchmarks/nt/discrete_log.cpp, measured 2026-10-15, x86-64 Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, -O2,
 * same engine
 * on both sides): Pohlig–Hellman, the library strategy, solved 100 logarithms modulo the
 * smooth-order prime 998244353 in 1.1 ms against 76.3 ms for BSGS (69.4x), and 20
 * modulo the safe prime 1e9+7 in 13.1 ms against 15.4 ms (1.18x). In the second case
 * the order of the base is 2q with q prime, so Pohlig–Hellman's work is one BSGS
 * over q ≈ m/2. Near 1e18, BSGS needs a table of 1e9 entries and was not run;
 * Pohlig–Hellman took 3.8 ms for 100 logarithms modulo 29·2^57 + 1 and 377 ms for 5
 * modulo a prime whose p − 1 has a prime factor near 1e12.
 */

#include <atl/config.hpp>
#include <atl/ds/flat_hash_map.hpp>
#include <atl/nt/discrete_log.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <cstddef>
#include <limits>
#include <optional>

namespace atl::experimental {
	/**
	 * @brief Least `y >= 0` with `a^y = t` for unit forms `a`, `t` of @p engine, or
	 *        none, by baby-step giant-step over `[0, m)`.
	 *
	 * The babies `t·a^j` (`j < n = ceil(sqrt(m))`) map to their largest `j`; the
	 * giants `a^(n·i)` for `i = 1..n` probe them, and the first hit gives
	 * `y = n·i − j`, the least one. No inverse is needed. Drop-in for
	 * `atl::detail::pohlig_hellman` as the solver behind
	 * `atl::detail::discrete_log_reduced`.
	 *
	 * Lost to Pohlig–Hellman on every shape (see the file comment): 69.4x on a
	 * smooth-order prime near 1e9, 1.18x on the safe prime 1e9+7.
	 *
	 * @tparam Engine Modular multiplier satisfying the internal engine contract.
	 * @param engine Arithmetic engine and modulus.
	 * @param a Base in engine form.
	 * @param t Target in engine form.
	 * @return The least exponent, or std::nullopt if unreachable.
	 * @pre `a` and `t` are unit forms of @p engine.
	 * @note Not constexpr (hash tables have a runtime seed).
	 * @complexity Expected O(sqrt(m)) time and storage; hashes do not affect the result.
	 */
	template<atl::detail::modular_multiplier Engine>
	std::optional<uint64> discrete_log_by_baby_step_giant_step(Engine const &engine, uint64 a, uint64 t) {
		if (t == engine.one()) return uint64(0);
		uint64 const step = atl::detail::ceil_sqrt(engine.mod());
		atl_assert(step <= (uint64(1) << 32)); // baby indices are uint32
		atl_assert(step <= (std::numeric_limits<std::size_t>::max)());
		flat_hash_map<uint64, uint32> baby(static_cast<std::size_t>(step));
		uint64 power = t;
		for (uint64 j = 0; j < step; ++j) {
			baby.insert_or_assign(power, uint32(j)); // the largest j wins: least y
			power = engine.mul(power, a);
		}
		uint64 const giant = atl::detail::engine_power(engine, a, step);
		power = engine.one();
		for (uint64 i = 1; i <= step; ++i) {
			power = engine.mul(power, giant);
			auto const it = baby.find(power);
			if (it != baby.end()) return uint64(uint128(i) * step - it->second);
		}
		return std::nullopt;
	}
}
