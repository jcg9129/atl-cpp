#pragma once

/**
 * @file
 * @brief Contenders for @ref atl::link_cut_tree, every one the full class
 *        over the shared implementation with one axis changed:
 *  - path form: two-access (library: `access(u)`, then exposing v meets at
 *    the lca and cuts off u's side) vs evert-operate-evert-back
 *    (`IsEvertForm`: find the root, re-root at u, access v, re-root back);
 *  - size field on (library) or off (`IsSized = false`: no `depth`,
 *    `distance`, `ancestor`, `path_vertex`);
 *  - handle-splay push: the engine's ancestor stack (library) vs the
 *    experimental engine's parity walk (reversal-only tiers) or pointer
 *    reversal;
 *  - node layout: interleaved (library), split links/payload, columnar;
 *  - commutative fold: the public `FoldOrder = commutative_fold_t` mode,
 *    raced against the default directly.
 *
 * benchmarks/ds/link_cut_tree.cpp, Xeon Platinum 8280, GCC 15.2,
 * RelWithDebInfo, assertions off, 2026-10-25, -O2 -march=native,
 * N=Q=5e5: by rerooting takes 531 vs library 268 ms on random-tree path
 * sums and 1823 vs 913 ms with lazy affine updates. Omitting sizes takes
 * 250 vs 268 ms on random trees and 434 vs 463 ms on structure-only churn,
 * but drops depth and indexed path queries. Parity-walk push takes 473 vs
 * 433 ms on a path; split layout 404 vs 433 ms on a path and 817 vs 753 ms
 * on edge-tier MST. Columnar layout takes 1157 vs 753 ms on edge-tier MST.
 * The public commutative-fold mode takes 227 vs 268 ms on random trees and
 * 736 vs 913 ms on lazy affine updates. The library retains size fields,
 * interleaved storage and ancestor-stack pushing. Full tables in the guide.
 */

#include <atl/ds/detail/splay_layout.hpp>
#include <atl/ds/link_cut_tree.hpp>
#include <atl/experimental/ds/splay_engine_variants.hpp>
#include <atl/types/alias.hpp>
#include <concepts>
#include <type_traits>

namespace atl::experimental {
	/**
	 * @brief @ref atl::link_cut_tree with the engine, layout, size field and
	 *        path form chosen at compile time (the template arguments after
	 *        them as in the library class).
	 * @tparam Value Default-initializable when non-void.
	 * @tparam Tag Default-initializable when non-void.
	 */
	template<typename Engine, atl::detail::splay_layout Layout, bool IsSized, bool IsEvertForm, typename Value = void,
			 typename Op = void, typename Tag = void, typename Mapping = void, typename Compose = void,
			 typename FoldOrder = void, typename ValuePlacement = void>
		requires atl::detail::link_cut_algebra<Value, Op, Tag, Mapping, Compose, FoldOrder, ValuePlacement> && (std::is_void_v<Value> || std::default_initializable<Value>) && (std::is_void_v<Tag> || std::default_initializable<Tag>)
			  && atl::detail::link_cut_engine_for<Engine, atl::detail::link_cut_store<Layout, IsSized, Value, Op, Tag, Mapping,
																					  Compose, FoldOrder>>
	class link_cut_tree_by_configuration : public atl::detail::link_cut_tree_impl<Engine, Layout, IsSized, IsEvertForm, Value, Op, Tag,
																				  Mapping, Compose, FoldOrder, ValuePlacement> {
		using base = atl::detail::link_cut_tree_impl<Engine, Layout, IsSized, IsEvertForm, Value, Op, Tag, Mapping, Compose,
													 FoldOrder, ValuePlacement>;

	public:
		using base::base;
	};

	/** @brief The library engine's splay shape with the parity-walk push (reversal-only tiers). */
	using link_cut_parity_engine = splay_engine_by_strategy<splay_shape::top_down, splay_path_push::parity_walk>;

	/** @brief The library engine's splay shape with the buffer-free pointer-reversal push. */
	using link_cut_pointer_reversal_engine = splay_engine_by_strategy<splay_shape::top_down, splay_path_push::pointer_reversal>;
}
