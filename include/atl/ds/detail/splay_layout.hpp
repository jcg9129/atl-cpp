#pragma once
#include <concepts>

namespace atl {
	/** @brief Splay node layout: links, size and payload in one record per node. */
	struct splay_layout_interleaved {};

	/** @brief Splay node layout: a 16-byte links record (`child[2]`, parent, size) apart from the payload. */
	struct splay_layout_split {};

	namespace detail {
		using atl::splay_layout_interleaved;
		using atl::splay_layout_split;

		/**
		 * @brief Columnar storage separates every field. Mutation-heavy folds pay
		 * for the extra arrays: n=1e6, uniform surgery 3750 vs 1922 ms with
		 * interleaved storage, reverse/fold 8814 vs 3023 ms.
		 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, -O2,
		 * 2026-10-24; benchmarks/ds/splay_sequence.cpp.
		 */
		struct splay_layout_columnar {};

		/**
		 * @brief One of the three node layouts, shared by the splay-engine
		 *        clients (@ref atl::splay_sequence, @ref atl::link_cut_tree), so one
		 *        selector names a layout everywhere.
		 */
		template<typename Layout>
		concept splay_layout = std::same_as<Layout, splay_layout_interleaved> || std::same_as<Layout, splay_layout_split>
							|| std::same_as<Layout, splay_layout_columnar>;

		/** @brief Accept the layouts implemented by the sequence pool. */
		template<typename Layout>
		concept splay_sequence_layout = splay_layout<Layout>;

		/** @brief The layouts public splay containers accept: interleaved or split. */
		template<typename Layout>
		concept splay_selectable_layout = std::same_as<Layout, splay_layout_interleaved> || std::same_as<Layout, splay_layout_split>;
	}
}
