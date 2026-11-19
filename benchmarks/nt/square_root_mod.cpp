/**
 * @file
 * @brief Race of `atl::square_root_mod` (Tonelli-Shanks) against `atl::experimental::square_root_mod_by_cipolla`.
 *
 * The one difference: Tonelli-Shanks corrects an approximate root through the 2-power subgroup of the multiplicative
 * group, so its work grows with the exponent s of 2 in `prime - 1`; Cipolla raises `a + u` to `(prime + 1) / 2` in a
 * quadratic extension, independent of s. Both use the same full-width Montgomery arithmetic.
 *
 * What the loop measures: the throughput of 256 root computations per Catch2 sample (checksummed so the results
 * are used), over alternating residues and non-residues of fixed seeded roots, after REQUIRE-checking that the two
 * contenders agree and that every root is a square root of its input. Shapes: the primes 998244353 (s = 23),
 * 1000000007 (3 mod 4), 29 * 2^57 + 1 (62 bits, s = 57) and eight seeded random 64-bit primes.
 *
 * Result: not measured yet; the race is pending, so Tonelli-Shanks is only the provisional default.
 */
#include <atl/experimental/nt/square_root_mod_variants.hpp>
#include <atl/nt/basic.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/square_root_mod.hpp>
#include <atl/utils/assert.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
	using std::uint64_t;

	/** @brief Alternating nonzero residues and non-residues, paired with their prime moduli. */
	std::vector<std::pair<uint64_t, uint64_t>> make_instances(std::vector<uint64_t> const &primes) {
		std::mt19937_64 random(42);
		std::vector<std::pair<uint64_t, uint64_t>> instances;
		for (auto prime : primes) {
			uint64_t non_residue = 2;
			while (atl::pow_mod(non_residue, (prime - 1) / 2, prime) == 1) {
				atl_assert(non_residue < prime - 1);
				++non_residue;
			}
			for (int i = 0; i < 16; ++i) {
				uint64_t const root = random() % (prime - 1) + 1;
				uint64_t const square = atl::mul_mod_unchecked(root, root, prime);
				instances.emplace_back(square, prime);
				instances.emplace_back(atl::mul_mod_unchecked(square, non_residue, prime), prime);
			}
		}
		return instances;
	}

	/** @brief XOR of every root over the instances, so the optimizer cannot drop the work. */
	template<typename Contender>
		requires std::is_invocable_r_v<std::optional<uint64_t>, Contender const &, uint64_t, uint64_t>
	uint64_t checksum_of_roots(std::vector<std::pair<uint64_t, uint64_t>> const &instances, Contender const &contender) {
		uint64_t checksum = 0;
		for (auto const &[value, prime] : instances) checksum ^= contender(value, prime).value_or(0);
		return checksum;
	}

	/** @brief Check that the contenders agree on every instance, then time both. */
	void race(std::string const &shape, std::vector<uint64_t> const &primes) {
		auto const instances = make_instances(primes);
		for (std::size_t i = 0; i < instances.size(); ++i) {
			auto const [value, prime] = instances[i];
			auto const expected = atl::square_root_mod(value, prime);
			REQUIRE(expected == atl::experimental::square_root_mod_by_cipolla(value, prime));
			REQUIRE(expected.has_value() == (i % 2 == 0));
			if (expected) {
				REQUIRE(*expected <= prime / 2);
				REQUIRE(atl::mul_mod_unchecked(*expected, *expected, prime) == value);
			}
		}
		auto const library = [](uint64_t value, uint64_t prime) { return atl::square_root_mod(value, prime); };
		auto const by_cipolla = [](uint64_t value, uint64_t prime) { return atl::experimental::square_root_mod_by_cipolla(value, prime); };
		BENCHMARK("square_root_mod (library) | " + shape) { return checksum_of_roots(instances, library); };
		BENCHMARK("square_root_mod_by_cipolla | " + shape) { return checksum_of_roots(instances, by_cipolla); };
	}
}

TEST_CASE("square_root_mod: Tonelli-Shanks versus Cipolla (race pending)", "[square_root_mod]") {
	std::mt19937_64 random(6);
	std::vector<uint64_t> random_primes;
	while (random_primes.size() < 8) {
		uint64_t const candidate = random() | (uint64_t(1) << 63) | uint64_t(1);
		if (atl::is_prime(candidate)) random_primes.push_back(candidate);
	}
	race("998244353, mixed x256", std::vector<uint64_t>(8, 998244353));
	race("1000000007 (3 mod 4), mixed x256", std::vector<uint64_t>(8, 1000000007));
	race("29*2^57+1 (62-bit), mixed x256", std::vector<uint64_t>(8, 4179340454199820289ull));
	race("random 64-bit primes, mixed x256", random_primes);
}
