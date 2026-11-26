#pragma once
#include <atl/types/alias.hpp>
#include <concepts>

/**
 * @brief Ready-made cluster algebras for @ref atl::top_tree : library `Value`,
 *        `Path`, `Point` and `Tag` types with their merges, identities
 *        given by the types' default values. A @ref atl::top_tree over one of
 *        them gets the set's named members (`path_sum`, `diameter`,
 *        `marked_distance_sum`, ...), provided by its cluster header, and
 *        constructs from a vertex count (or values) alone; each set has an
 *        alias spelling every slot (`atl::sum_top_tree<T>`, ...).
 */
namespace atl::top_clusters {
	namespace detail {
		/**
		 * @brief A value the summing sets can hold: regular, closed under `+`
		 *        and `*`, and constructible from a vertex count (the add tags
		 *        multiply by the number of vertices they cover).
		 */
		template<typename T>
		concept summable = std::regular<T> && std::constructible_from<T, uint32> && requires(T const &a, T const &b) {
			{ a + b } -> std::convertible_to<T>;
			{ a * b } -> std::convertible_to<T>;
		};

		/** @brief An edge length of the distance sets: `summable` and ordered. */
		template<typename W>
		concept length = summable<W> && std::totally_ordered<W>;
	}

	/**
	 * @brief Whether @p Path is the path type of a ready-made set: then a
	 *        @ref atl::top_tree over it takes every identity from the types'
	 *        default values and constructs from a vertex count alone.
	 */
	template<typename Path>
	inline constexpr bool is_ready_made = false;

	namespace detail {
		/** @brief A value with a per-vertex starting value (`Value::of_vertex(v)`). */
		template<typename Value>
		concept per_vertex_value = requires(uint32 v) {
			{ Value::of_vertex(v) } -> std::convertible_to<Value>;
		};
	}
}
