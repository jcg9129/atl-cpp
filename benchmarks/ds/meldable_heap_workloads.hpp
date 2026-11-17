#pragma once

/**
 * @file
 * @brief Workloads shared by the meldable-heap tuning and race benchmarks:
 *        bottom-up melds over trees, a slope-trick-like pattern and random
 *        push / pop / meld mixes. Inputs are generated outside the timed
 *        region; each run returns a checksum of the popped values.
 */

#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace meldable_heap_bench {
	using atl::int64;
	using atl::uint32;

	/** @brief A forest of int64 heaps the workloads can drive. */
	template<typename Forest>
	concept int64_forest = std::constructible_from<Forest, std::size_t> && requires(Forest &forest, uint32 h, int64 x) {
		forest.push(h, x);
		{ forest.pop(h) } -> std::convertible_to<int64>;
		forest.meld(h, h);
		{ forest.size(h) } -> std::convertible_to<std::size_t>;
		forest.reserve(h);
	};

	/** @brief An @ref int64_forest with a lazy add of an int64. */
	template<typename Forest>
	concept shifting_int64_forest = int64_forest<Forest> && requires(Forest &forest, uint32 h, int64 x) {
		forest.add_to_all(h, x);
	};

	/** @brief The tree shapes: a path, a star, a caterpillar (half spine, a leaf per spine vertex), random. */
	enum class tree_shape { path,
							star,
							caterpillar,
							random };

	inline std::string name(tree_shape shape) {
		switch (shape) {
		case tree_shape::path:
			return "path";
		case tree_shape::star:
			return "star";
		case tree_shape::caterpillar:
			return "caterpillar";
		default:
			return "random";
		}
	}

	/** @brief A tree on `n` vertices rooted at 0 with `parent[v] < v`, so `v = n-1 .. 1` is a bottom-up order. */
	inline std::vector<uint32> tree_parents(tree_shape shape, uint32 n, std::mt19937_64 &rng) {
		std::vector<uint32> parent(n, 0);
		for (uint32 v = 1; v < n; ++v) switch (shape) {
			case tree_shape::path:
				parent[v] = v - 1;
				break;
			case tree_shape::star:
				parent[v] = 0;
				break;
			case tree_shape::caterpillar:
				parent[v] = v < n / 2 ? v - 1 : v - n / 2;
				break;
			default:
				parent[v] = uint32(rng() % v);
				break;
			}
		return parent;
	}

	/** @brief `count` uniform values in `[0, 2^40)`. */
	inline std::vector<int64> random_values(std::size_t count, std::mt19937_64 &rng) {
		std::vector<int64> values(count);
		for (int64 &x : values) x = int64(rng() >> 24);
		return values;
	}

	/**
	 * @brief Bottom-up over the tree, one heap per vertex: each vertex (its
	 *        children already melded in) pushes 3 values, pops 2, then melds
	 *        into its parent. `values` holds `3 n` values.
	 */
	template<int64_forest Forest>
	int64 tree_melds(std::vector<uint32> const &parent, std::vector<int64> const &values, bool should_reserve) {
		std::size_t const n = parent.size();
		atl_assert(n < (std::numeric_limits<uint32>::max)());
		Forest forest(n);
		if (should_reserve) forest.reserve(n + 2);
		int64 checksum = 0;
		for (std::size_t v = n; v-- > 0;) {
			forest.push(uint32(v), values[3 * v]);
			forest.push(uint32(v), values[3 * v + 1]);
			forest.push(uint32(v), values[3 * v + 2]);
			checksum += forest.pop(uint32(v));
			checksum ^= forest.pop(uint32(v));
			if (v != 0) forest.meld(parent[v], uint32(v));
		}
		return checksum + int64(forest.size(0));
	}

	/**
	 * @brief The slope-trick pattern on min-heaps of breakpoints: bottom-up,
	 *        each vertex (children melded in) pops one top per child, pushes
	 *        2 values, shifts its whole heap, then melds into its parent.
	 *        `values` holds `3 n` values (the third is the shift).
	 */
	template<shifting_int64_forest Forest>
	int64 slope_trick_pattern(std::vector<uint32> const &parent, std::vector<int64> const &values) {
		std::size_t const n = parent.size();
		atl_assert(n < (std::numeric_limits<uint32>::max)());
		std::vector<uint32> children(n, 0);
		for (std::size_t v = 1; v < n; ++v) ++children[parent[v]];
		Forest forest(n);
		int64 checksum = 0;
		for (std::size_t v = n; v-- > 0;) {
			for (uint32 i = 0; i < children[v] && forest.size(uint32(v)) > 1; ++i) checksum += forest.pop(uint32(v));
			forest.push(uint32(v), values[3 * v]);
			forest.push(uint32(v), values[3 * v + 1]);
			forest.add_to_all(uint32(v), values[3 * v + 2] % 1000);
			if (v != 0) forest.meld(parent[v], uint32(v));
		}
		return checksum + int64(forest.size(0));
	}

	/** @brief One step of a random mix: push `value` onto `a`, pop `a`, or meld `b` into `a`. */
	struct operation {
		enum : uint32 { push,
						pop,
						meld } kind;

		uint32 a, b;
		int64 value;
	};

	/**
	 * @brief `count` random operations over `heap_count` heaps, push / pop /
	 *        meld with weights `push_weight` / `pop_weight` / the rest of 100;
	 *        a pop of an empty heap is replayed as a push.
	 */
	inline std::vector<operation> random_mix(std::size_t count, uint32 heap_count, uint32 push_weight, uint32 pop_weight,
											 std::mt19937_64 &rng) {
		std::vector<operation> ops(count);
		for (operation &op : ops) {
			uint32 const roll = uint32(rng() % 100);
			op.kind = roll < push_weight			  ? operation::push
					: roll < push_weight + pop_weight ? operation::pop
													  : operation::meld;
			op.a = uint32(rng() % heap_count);
			op.b = uint32(rng() % heap_count);
			op.value = int64(rng() >> 24);
		}
		return ops;
	}

	/** @brief Replay `ops` on a fresh forest of `heap_count` heaps. */
	template<int64_forest Forest>
	int64 replay(std::vector<operation> const &ops, uint32 heap_count) {
		Forest forest(heap_count);
		int64 checksum = 0;
		for (operation const &op : ops) {
			if (op.kind == operation::push || (op.kind == operation::pop && forest.size(op.a) == 0))
				forest.push(op.a, op.value);
			else if (op.kind == operation::pop)
				checksum += forest.pop(op.a);
			else
				forest.meld(op.a, op.b);
		}
		return checksum;
	}
}
