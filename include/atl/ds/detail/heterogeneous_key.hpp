#pragma once
#include <atl/types/concepts.hpp>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace atl::detail {
	/**
	 * @brief Two arithmetic types, which are never compared directly as a lookup:
	 *        a mixed comparison applies the usual arithmetic conversions
	 *        (`-1 < 5u` is false, `-1 == 2^64 - 1`), not the conversion to `Key`.
	 */
	template<typename Lookup, typename Key>
	concept arithmetic_pair = (std::is_arithmetic_v<Lookup> || atl::integral<Lookup>) && (std::is_arithmetic_v<Key> || atl::integral<Key>);

	/** @brief A comparator that opts into heterogeneous lookup (`Compare::is_transparent`). */
	template<typename Compare>
	concept transparent_comparator = requires { typename Compare::is_transparent; };

	/**
	 * @brief A lookup argument @p Compare orders against stored @p Key values
	 *        directly, without building a `Key`: the comparator is transparent
	 *        and a strict weak order over the mixed pair, and the pair is not
	 *        an @ref arithmetic_pair.
	 */
	template<typename Lookup, typename Key, typename Compare>
	concept heterogeneous_key_for =
		transparent_comparator<Compare> && std::strict_weak_order<Compare const &, Key const &, Lookup const &>
		&& !arithmetic_pair<Lookup, Key>;

	/**
	 * @brief A lookup argument an ordered container accepts: compared directly when
	 *        @ref heterogeneous_key_for holds, otherwise converted to `Key` once.
	 */
	template<typename Lookup, typename Key, typename Compare>
	concept ordered_lookup_key = heterogeneous_key_for<Lookup, Key, Compare> || std::convertible_to<Lookup const &, Key>;

	/**
	 * @brief A lookup argument a hash table probes with directly, without building a `Key`:
	 *        @p Hash and @p Eq both opt in with `is_transparent` (as the standard's
	 *        transparent lookup), `hash(k)` and `eq(key, k)` are valid, and the pair is
	 *        not an @ref arithmetic_pair. The hasher must give @p Lookup and the equal @p Key the same hash;
	 *        that is the caller's contract.
	 */
	template<typename Lookup, typename Key, typename Hash, typename Eq>
	concept hashed_heterogeneous_key_for =
		requires { typename Hash::is_transparent; typename Eq::is_transparent; }
		&& requires(Hash const &h, Eq const &e, Lookup const &k, Key const &key) {
			   { h(k) } -> std::convertible_to<std::size_t>;
			   { e(key, k) } -> std::convertible_to<bool>;
		   } && !arithmetic_pair<Lookup, Key>;
}
