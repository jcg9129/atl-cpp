#pragma once

/** @file
 * @brief The shared process-wide hash seed and integer mixing primitive.
 */

#include <atl/types/alias.hpp>
#include <chrono>
#include <random>

namespace atl {
	namespace detail {
		/**
		 * @brief The splitmix64 step and output mix (the SplitMix generator of
		 *        Steele, Lea and Flood, 2014, with Stafford's Mix13 constants):
		 *        a bijection of `uint64` with full avalanche.
		 */
		constexpr uint64 splitmix64(uint64 x) noexcept {
			x += 0x9e3779b97f4a7c15ULL;
			x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
			x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
			return x ^ (x >> 31);
		}

		// Entropy failure is fatal during static initialization; make that policy explicit.
		inline uint64 initial_hash_seed() noexcept {
			uint64 s = uint64(std::chrono::steady_clock::now().time_since_epoch().count());
			std::random_device rd;
			s ^= (uint64(rd()) << 32) ^ rd();
			return splitmix64(s);
		}

		/**
		 * @brief The per-process hash seed, drawn once (clock + random_device,
		 *        mixed) so hashing is unpredictable, which defeats anti-hash
		 *        (hash-flooding) attacks. A namespace-scope variable initialised
		 *        during static initialisation, not a function-local static: a
		 *        hash then reads it with a plain load and no guard check
		 *        (5-8% faster flat_hash_set lookups on x86-64,
		 *        benchmarks/ds/hash.cpp).
		 *
		 * The price: a hash computed during static initialisation, before this
		 * variable is initialised, uses seed 0, and the value changes once the
		 * initialiser runs. Hence the rule on set_hash_seed and on tables built
		 * during static initialisation.
		 */
		inline uint64 hash_seed_value = initial_hash_seed();

		inline uint64 hash_seed() noexcept { return hash_seed_value; }

	}

	/**
	 * @brief Pin the process-wide hash seed, for reproducible runs.
	 *
	 * @pre Call from `main` (or later), before building any table whose keys use
	 * this seed. No thread may hash or call set_hash_seed concurrently with this
	 * call. Changing the seed invalidates all previously computed seeded hashes
	 * and table placement.
	 * @warning Not during static initialisation: the seed is itself initialised
	 * then, so an earlier call can be overwritten, and a table filled earlier is
	 * invalidated when the seed changes. Build seeded tables after `main` starts.
	 * @param seed Seed to mix and install for subsequent hash calls.
	 * @complexity O(1).
	 * @note Not constexpr: changes process-wide runtime state.
	 */
	inline void set_hash_seed(uint64 seed) noexcept { detail::hash_seed_value = detail::splitmix64(seed); }

}
