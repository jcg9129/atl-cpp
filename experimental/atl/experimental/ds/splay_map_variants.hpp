#pragma once

/**
 * @file
 * @brief Splay-map contenders varying top-down versus bottom-up descent and
 * split versus interleaved node storage. Repush is identical to bottom-up on
 * a push-free keyed store; semi-splay changes only positional sequence reads.
 *
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-24,
 * benchmarks/ds/splay_map.cpp, -O2 -march=native, n=1e6: mixed operations take
 * 1065 ms with top-down split storage (library), 1058 ms top-down interleaved,
 * 1258 ms bottom-up interleaved and 1283 ms bottom-up split. Successor chains
 * take 82.3, 97.4, 101 and 84.0 ms respectively. Split storage improves the
 * successor workload while top-down descent reduces random-workload cost.
 * Complete tables and both flag sets: docs/splay-map.md.
 */

#include <atl/ds/splay_map.hpp>
#include <atl/experimental/ds/splay_engine_variants.hpp>
#include <concepts>
#include <functional>

namespace atl::experimental {
	/**
	 * @brief `atl::splay_map` over any engine and node layout, the full
	 *        interface (so the oracle tests and the race run every contender).
	 */
	template<typename Engine, typename Layout, std::semiregular Key, typename Value = void, typename Compare = std::ranges::less,
			 typename Op = void>
		requires atl::detail::splay_map_layout<Layout> && atl::detail::splay_map_parameters<Key, Value, Compare, Op>
			  && atl::detail::splay_map_engine<Engine, atl::detail::splay_map_pool<Layout, Key, Value, Compare, Op>>
	class splay_map_by_strategy : public atl::detail::splay_map_impl<splay_map_by_strategy<Engine, Layout, Key, Value, Compare, Op>, Engine,
																	 Layout, Key, Value, Compare, Op> {
		using base = atl::detail::splay_map_impl<splay_map_by_strategy, Engine, Layout, Key, Value, Compare, Op>;

	public:
		/** @brief Inherit the tier-specific constructors. */
		using base::base;
	};

	/** @brief Top-down descent through the library engine. */
	using splay_by_top_down = splay_engine_by_strategy<splay_shape::top_down, splay_path_push::ancestor_stack>;

	/** @brief Bottom-up descent through an already-pushed path. */
	using splay_by_bottom_up = splay_engine_by_strategy<splay_shape::bottom_up, splay_path_push::ancestor_stack>;
}
