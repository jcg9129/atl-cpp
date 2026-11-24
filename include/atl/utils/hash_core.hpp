#pragma once

/** @file
 * @brief Shared hash seed and scalar integer, enum and identity hashes.
 */

#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/traits/integral.hpp>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <random>
#include <type_traits>

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

		/**
		 * @brief The integer keys of @ref hash and @ref identity_hash: std's
		 *        integral types (with `bool` and the character types) plus
		 *        atl's (with the 128-bit types in every language mode).
		 */
		template<typename T>
		concept hashable_integer = std::integral<T> || atl::integral<T>;

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

	/**
	 * @brief Hash that returns the key unchanged — the fastest hash, for keys already
	 *        uniformly random (e.g. outputs of a well-mixed hash).
	 *
	 * @warning Not seeded and not mixing: it clusters on structured keys and is **not**
	 *          anti-hash safe. Use only when the keys are known to be random.
	 *          A key wider than `std::size_t` is truncated to its low bits.
	 */
	struct identity_hash {
		/**
		 * @brief Return the integer key reduced to `std::size_t`, without mixing.
		 * @tparam T An integer type, including `bool` and character types.
		 * @param x Key whose low bits are returned.
		 * @return The key converted to `std::size_t`.
		 * @complexity O(1).
		 */
		template<detail::hashable_integer T>
		constexpr std::size_t operator()(T x) const noexcept {
			// A 32-bit size_t keeps the low bits intentionally: the table consumes low bits.
			return static_cast<std::size_t>(x);
		}
	};

	/**
	 * @brief Default strong, seeded hash functor.
	 *
	 * Specialized here for integers and enums; `hash_extensions.hpp` adds floating-point, strings, pointers,
	 * `std::pair`, `std::tuple`, `std::vector` and `std::array`; every path mixes in
	 * a per-process random seed (see @ref set_hash_seed), so it resists anti-hash
	 * attacks. Equal keys hash equally within one process run; hashes differ between
	 * runs unless the seed is pinned. The primary template is left undefined on
	 * purpose — specialize `atl::hash<YourType>` for a custom key, or pass
	 * `std::hash<YourType>` (or any functor) as the container's `Hash` parameter.
	 *
	 * A user specialization must be default constructible and callable as
	 * `hash<K>{}(value)` on the element reference used by its composite. The
	 * result must be implicitly convertible to `uint64`; construction, call and
	 * conversion must all be noexcept. Equal values must produce equal hashes
	 * while the seed is unchanged. Pair/array/vector use the declared element
	 * type (including cv); tuple uses the decay of its const element reference.
	 * Integer packing and byte-buffer paths do not require element hashers.
	 * All composites are noexcept and reject throwing customizations.
	 *
	 * @par Integer keys
	 * Standard integer types (including `bool` and character types) and atl's
	 * 128-bit integers are supported. Keys up to 64 bits use one seeded
	 * splitmix64 round; wider keys chain the low and high halves through the
	 * seeded state. The result is reduced to `std::size_t` on narrower targets.
	 *
	 * @par Floating-point keys
	 * Equal values of the same key type hash equally: `-0.0 == +0.0`, so both
	 * zeros are normalized before hashing. All NaNs are canonicalized to one hash
	 * per seed, although NaN values still compare unequal. Known x87 layouts hash
	 * only the ten value bytes, excluding padding in 12- or 16-byte objects;
	 * known binary128 layouts use all sixteen bytes. Unknown wide-float layouts
	 * and forced-fallback builds use value decomposition into sign, exponent and
	 * significand bits, never object padding. Hashes are not stable across ABIs.
	 *
	 * @tparam T Key type; an unsupported type requires a user specialization.
	 * @note Not constexpr: every specialization reads the run-time seed.
	 */
	template<typename T>
	struct hash;

	/**
	 * @brief Integers (128-bit ones included): one seeded splitmix64 round per 64 bits.
	 * @copydetails atl::hash
	 */
	template<detail::hashable_integer T>
	struct hash<T> {
		/**
		 * @brief Hash an integer key with one seeded mix per 64-bit word.
		 * @param x Integer key; both halves participate when wider than 64 bits.
		 * @return Seeded hash reduced to `std::size_t`.
		 * @complexity O(1).
		 */
		std::size_t operator()(T x) const noexcept {
			if constexpr (sizeof(T) > 8) {
				// Convert the bit pattern before shifting so signed keys use a logical shift.
				auto const u = traits::make_unsigned_t<T>(x);
				// chained, not xor-combined: the high half must pass through
				// the seeded state, or keys could be built whose hashes agree
				// in their low bits under every seed
				return static_cast<std::size_t>(detail::splitmix64(uint64(u >> 64) ^ detail::splitmix64(uint64(u) + detail::hash_seed())));
			} else {
				return static_cast<std::size_t>(detail::splitmix64(uint64(x) + detail::hash_seed()));
			}
		}
	};

	/**
	 * @brief Enums hash as their underlying value (`hash<E>{}(e) ==
	 *        hash<std::underlying_type_t<E>>{}(e)`). The conversion is harmless
	 *        here, although atl::traits::make_signed/make_unsigned reject enums
	 *        in arithmetic contexts.
	 */
	template<typename E>
		requires std::is_enum_v<E>
	struct hash<E> {
		/**
		 * @brief Hash an enum through its underlying integer type.
		 * @param e Enum value to convert and hash.
		 * @return The seeded hash of the underlying value.
		 * @complexity O(1).
		 */
		std::size_t operator()(E e) const noexcept {
			using underlying = std::underlying_type_t<E>;
			return hash<underlying>{}(static_cast<underlying>(e));
		}
	};

}
