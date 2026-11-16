/**
 * @file
 * @brief The race losers behind @ref atl::time_divide_and_conquer using a dense time tree,
 *        same signature as the library engine. Dense CSR uses an iterative walk
 *        and pruning by `should_visit`:
 * - @ref time_divide_and_conquer_by_recursive_vectors: recursive walk over a midpoint
 *   segment tree, one `std::vector<uint32>` of items per node;
 * - @ref time_divide_and_conquer_by_unpruned_csr: Dense CSR without pruning, every subtree
 *   inside `[0, T)` is walked and `should_visit` is checked at the leaves;
 * - @ref time_divide_and_conquer_by_configurable_csr, CSR with its tunings switchable, to measure them.
 *
 * Results and why each lost: docs/time-divide-and-conquer.md and
 * benchmarks/ds/time_divide_and_conquer.cpp.
 *
 * Sparse-query comparison (2026-11-01, Xeon 8280, plain -O2 / -march=native):
 * Rank compression ships automatically at Q <= T / 10. Retained losers:
 * - @ref time_divide_and_conquer_by_frame_skipping: Dense CSR without the frames whose
 *   inserts an enclosing frame's pop discards anyway;
 * - @ref time_divide_and_conquer_by_ranked_frame_skipping: rank compression
 *   plus skipping; no copy-mode gain and within noise on plain -O2;
 * - @ref time_divide_and_conquer_by_dense_csr: Dense CSR without automatic compression.
 */
#pragma once
#include <atl/ds/time_divide_and_conquer.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <ranges>
#include <utility>
#include <vector>

namespace atl::experimental {
	namespace detail {

		template<typename Structure, typename Mode>
		struct frames_of {
			using type = atl::detail::copy_frames<Structure>;
		};

		template<typename Structure, std::move_constructible Save, std::move_constructible Restore>
			requires restore_mode<by_rollback<Save, Restore>, Structure>
		struct frames_of<Structure, by_rollback<Save, Restore>> {
			using type = atl::detail::rollback_frames<Structure, Save, Restore>;
		};

		template<typename Structure, typename Mode>
		using frames_of_t = typename frames_of<Structure, Mode>::type;

		/** @brief The frame stack of a restore mode, as the library engine builds it. */
		template<typename Structure, restore_mode<Structure> Mode, typename Body>
			requires std::invocable<Body &, frames_of_t<Structure, Mode> &>
		constexpr void with_frames(Structure &structure, Mode &mode, std::size_t depth, Body &&body) {
			if constexpr (std::same_as<Mode, by_copy_t>) {
				atl::detail::copy_frames<Structure> frames(structure, depth);
				body(frames);
			} else {
				atl::detail::rollback_frames frames(structure, mode, depth);
				body(frames);
			}
		}

		/**
		 * @brief @ref atl::detail::walk_time_tree that opens no frame for a node
		 *        whose inserts the innermost open frame's pop discards anyway
		 *        (such a node inserts into that frame and closes nothing).
		 *
		 * Call a node *last* when it is a right child or a left child whose right
		 * sibling is not live: after its subtree the walk visits nothing more
		 * below its parent. Let `F` be the innermost open frame when the walk
		 * enters node `c`, opened (or absorbed, recursively) at an ancestor `w`.
		 * If every node on the path below `w` down to `c` is last, the walk
		 * leaves `c`'s subtree and then closes ancestors up to `w` without
		 * entering any other node, so `F` is popped right after `c`'s subtree.
		 * Inserting `c`'s items into `F` instead of a frame of its own is then
		 * invisible: everything the walk does inside `c`'s subtree sees the same
		 * state (deeper frames copy or save it), and `F`'s pop
		 * discards `c`'s items with the rest (a rollback restore undoes
		 * everything since its save; a copy is dropped whole), leaving the
		 * state the walk had before entering `w`, as if `c` had pushed and
		 * popped. The condition propagates top-down: `c` may absorb iff `c` is
		 * last and its parent either has items (it pushed or absorbed, so some
		 * frame closes right after the parent's subtree) or may itself absorb.
		 * The root never absorbs (no frame is open above it), so rollback mode
		 * still ends with `structure` as it started and copy mode never touches
		 * `structure`. Per depth two bits remember whether the node on the
		 * current path absorbed (no pop on the way up) and whether a child of it
		 * may absorb; the depth is at most 30.
		 *
		 * A frame is a full copy in copy mode and a save plus
		 * restore in rollback mode; chains of last children are common (every
		 * right child is last, and so is every left child of a pruned sibling),
		 * so a large share of the frames disappears, most for copy mode where a
		 * frame costs O(structure) or an allocation.
		 */
		template<atl::detail::restore_frames Frames, item_insertion<atl::detail::frames_structure_t<Frames>> Insert,
				 time_visit<atl::detail::frames_structure_t<Frames>> Visit>
		constexpr void visit_time_tree_by_frame_skipping(Frames &frames, uint32 leaves, std::vector<uint32> const &offsets,
														 std::vector<uint32> const &items, std::vector<uint8> const &is_live,
														 Insert &insert, Visit &visit) {
			uint64 is_absorbed = 0, is_absorbing = 0; // bit d: the path node at depth d
			uint32 v = 1, depth = 0;
			bool can_absorb = false; // whether v, being entered, may insert into the open frame
			while (true) {
				uint64 const bit = uint64(1) << depth;
				bool const has_items = offsets[v] != offsets[v + 1];
				is_absorbed = has_items && can_absorb ? is_absorbed | bit : is_absorbed & ~bit;
				is_absorbing = has_items || can_absorb ? is_absorbing | bit : is_absorbing & ~bit;
				if (has_items) {
					if (!can_absorb) frames.push();
					auto &s = frames.top();
					for (uint32 k = offsets[v]; k < offsets[v + 1]; ++k) std::invoke(insert, s, items[k]);
				}
				if (v < leaves) {
					bool const is_right_live = is_live[2 * std::size_t(v) + 1];
					v = is_live[2 * std::size_t(v)] ? 2 * v : 2 * v + 1;
					can_absorb = ((v & 1) || !is_right_live) && (is_absorbing & bit) != 0;
					++depth;
					continue;
				}
				std::invoke(visit, frames.top(), v - leaves);
				while (true) {
					if (offsets[v] != offsets[v + 1] && (is_absorbed & (uint64(1) << depth)) == 0) frames.pop();
					if (v == 1) return;
					if (!(v & 1) && is_live[v + 1]) {
						++v;
						can_absorb = (is_absorbing & (uint64(1) << (depth - 1))) != 0; // a right child is last
						break;
					}
					v >>= 1;
					--depth;
				}
			}
		}

		/** @brief Recursive-vector tree: per node the items whose cover contains it. */
		struct recursive_time_tree {
			std::vector<std::vector<uint32>> items; ///< heap-numbered, root 1, midpoint split
			uint32 time_count;						///< `T`

			/** @brief Put item `i` on the cover nodes of `[begin, end)` below `node`, which spans `[l, r)`. */
			constexpr void add(uint32 node, uint32 l, uint32 r, uint32 begin, uint32 end, uint32 i) {
				if (begin <= l && r <= end) {
					items[node].push_back(i);
					return;
				}
				uint32 const mid = l + (r - l) / 2;
				if (begin < mid) add(2 * node, l, mid, begin, end, i);
				if (mid < end) add(2 * node + 1, mid, r, begin, end, i);
			}

			/** @brief Visit selected times below `node`, spanning `[l, r)`. */
			template<atl::detail::restore_frames Frames, item_insertion<atl::detail::frames_structure_t<Frames>> Insert,
					 time_visit<atl::detail::frames_structure_t<Frames>> Visit, time_predicate ShouldVisit>
			constexpr void visit_time_subtree(Frames &frames, uint32 node, uint32 l, uint32 r, Insert &insert, Visit &visit,
											  ShouldVisit &should_visit) const {
				bool const has_items = !items[node].empty();
				if (has_items) {
					frames.push();
					for (uint32 i : items[node]) std::invoke(insert, frames.top(), i);
				}
				if (r - l == 1) {
					if (std::invoke(should_visit, l)) std::invoke(visit, frames.top(), l);
				} else {
					uint32 const mid = l + (r - l) / 2;
					visit_time_subtree(frames, 2 * node, l, mid, insert, visit, should_visit);
					visit_time_subtree(frames, 2 * node + 1, mid, r, insert, visit, should_visit);
				}
				if (has_items) frames.pop();
			}
		};
	}

	/**
	 * @brief Recursive vectors: midpoint tree of `4T` nodes with per-node item vectors;
	 * should_visit is checked at leaves. This shape is not measured yet.
	 * @tparam Structure Insertable represented state.
	 * @tparam Lifetimes Const sized forward range of item intervals.
	 * @tparam Insert Item-index callback. @tparam Visit Time-index callback.
	 * @tparam ShouldVisit Time predicate. @tparam Mode Restoration policy.
	 * @param structure Initial state, restored on normal completion.
	 * @param time_count Number of times.
	 * @param lifetimes Half-open interval of each item.
	 * @param insert Callback applying an item index.
	 * @param visit Callback observing a selected time.
	 * @param should_visit Predicate selecting times.
	 * @param mode Rollback callbacks or copy tag.
	 * @pre The bounds and callback laws of atl::time_divide_and_conquer apply;
	 *      time_divide_and_conquer_by_recursive_vectors additionally requires T <= 2^29.
	 * @complexity O(T + m log(T + 1)) engine storage and setup, plus insertion,
	 *             visit and restoration work; rank compression uses Q leaves.
	 */
	template<typename Structure, lifetime_range Lifetimes, item_insertion<Structure> Insert,
			 time_visit<Structure> Visit, time_predicate ShouldVisit = every_time,
			 restore_mode<Structure> Mode = default_restore_t<Structure>>
	constexpr void time_divide_and_conquer_by_recursive_vectors(Structure &structure, std::size_t time_count,
																Lifetimes const &lifetimes, Insert insert, Visit visit,
																ShouldVisit should_visit = {},
																Mode mode = default_restore<Structure>()) {
		atl_assert(time_count <= (std::size_t(1) << 29)); // heap numbering of the midpoint tree reaches 4T in uint32
		atl_assert(uint64(std::ranges::size(lifetimes)) < uint64((std::numeric_limits<uint32>::max)()));
		if (time_count == 0) return;
		auto const t_count = uint32(time_count);
		detail::recursive_time_tree tree{std::vector<std::vector<uint32>>(4 * std::size_t(t_count)), t_count};
		uint32 i = 0;
		for (auto const &lifetime : lifetimes) {
			auto const [begin, end] = atl::detail::lifetime_bounds(lifetime);
			atl_assert(begin <= end && end <= time_count);
			if (begin < end) tree.add(1, 0, t_count, uint32(begin), uint32(end), i);
			++i;
		}
		std::size_t const depth = std::size_t(std::bit_width(t_count)) + 1;
		detail::with_frames(structure, mode, depth,
							[&]<atl::detail::restore_frames Frames>(Frames &frames) { tree.visit_time_subtree(frames, 1, 0, t_count, insert, visit, should_visit); });
	}

	/**
	 * @brief CSR with independent pruning, suffix stretching and frame-skipping switches.
	 * Pruning omits subtrees with no query; stretching extends lifetimes ending at T
	 * over padding leaves. The unpruned and unstretched shapes are not measured yet.
	 * `IsSkipping` walks with @ref detail::visit_time_tree_by_frame_skipping.
	 * @tparam IsPruning skip subtrees holding no visited time
	 * @tparam IsStretching cover `[begin, leaves)` instead of `[begin, T)` when `end == T`
	 * @tparam IsSkipping open no frame for a node whose inserts the enclosing frame's pop discards
	 * @tparam Structure Insertable represented state.
	 * @tparam Lifetimes Const sized forward range of item intervals.
	 * @tparam Insert Item-index callback. @tparam Visit Time-index callback.
	 * @tparam ShouldVisit Time predicate. @tparam Mode Restoration policy.
	 * @param structure Initial state, restored on normal completion.
	 * @param time_count Number of times.
	 * @param lifetimes Half-open interval of each item.
	 * @param insert Callback applying an item index.
	 * @param visit Callback observing a selected time.
	 * @param should_visit Predicate selecting times.
	 * @param mode Rollback callbacks or copy tag.
	 * @pre The bounds and callback laws of atl::time_divide_and_conquer apply;
	 *      time_divide_and_conquer_by_recursive_vectors additionally requires T <= 2^29.
	 * @complexity O(T + m log(T + 1)) engine storage and setup, plus insertion,
	 *             visit and restoration work; rank compression uses Q leaves.
	 */
	template<bool IsPruning, bool IsStretching, bool IsSkipping = false, typename Structure, lifetime_range Lifetimes,
			 item_insertion<Structure> Insert, time_visit<Structure> Visit, time_predicate ShouldVisit = every_time,
			 restore_mode<Structure> Mode = default_restore_t<Structure>>
	constexpr void time_divide_and_conquer_by_configurable_csr(Structure &structure, std::size_t time_count, Lifetimes const &lifetimes,
															   Insert insert, Visit visit, ShouldVisit should_visit = {},
															   Mode mode = default_restore<Structure>()) {
		atl_assert(time_count <= (std::size_t(1) << 30));
		atl_assert(uint64(std::ranges::size(lifetimes)) < uint64((std::numeric_limits<uint32>::max)()));
		if (time_count == 0) return;
		uint32 const leaves = std::bit_ceil(uint32(time_count));
		uint32 const height = uint32(std::countr_zero(leaves));
		std::vector<uint8> is_live(2 * std::size_t(leaves), 0);
		for (uint32 t = 0; t < uint32(time_count); ++t)
			is_live[leaves + t] = IsPruning ? uint8(bool(std::invoke(should_visit, t))) : uint8(1);
		for (std::size_t v = leaves; v-- > 1;) is_live[v] = uint8(is_live[2 * v] | is_live[2 * v + 1]);
		if (!is_live[1]) return;
		auto const cover_end = [&](std::size_t end) {
			return IsStretching && end == time_count ? leaves : uint32(end);
		};
		std::vector<uint32> offsets(2 * std::size_t(leaves) + 1, 0);
		uint64 entries = 0;
		for (auto const &lifetime : lifetimes) {
			auto const [begin, end] = atl::detail::lifetime_bounds(lifetime);
			atl_assert(begin <= end && end <= time_count);
			if (begin == end) continue;
			atl::detail::for_each_cover_node(uint32(begin), cover_end(end), leaves, [&](uint32 v) {
				if (is_live[v]) ++offsets[v], ++entries;
			});
		}
		atl_assert(entries < uint64((std::numeric_limits<uint32>::max)()));
		uint32 running = 0;
		for (auto &o : offsets) o = running += o;
		std::vector<uint32> items(static_cast<std::size_t>(entries));
		{
			uint32 i = 0;
			for (auto const &lifetime : lifetimes) {
				auto const [begin, end] = atl::detail::lifetime_bounds(lifetime);
				if (begin != end) {
					atl::detail::for_each_cover_node(uint32(begin), cover_end(end), leaves, [&](uint32 v) {
						if (is_live[v]) items[--offsets[v]] = i;
					});
				}
				++i;
			}
		}
		auto checked_visit = [&](Structure &s, uint32 t) {
			if (IsPruning || std::invoke(should_visit, t)) std::invoke(visit, s, t);
		};
		detail::with_frames(structure, mode, height + 1, [&]<atl::detail::restore_frames Frames>(Frames &frames) {
			if constexpr (IsSkipping)
				detail::visit_time_tree_by_frame_skipping(frames, leaves, offsets, items, is_live, insert, checked_visit);
			else
				atl::detail::walk_time_tree(frames, leaves, offsets, items, is_live, insert, checked_visit);
		});
	}

	/** @brief Unpruned CSR, @ref time_divide_and_conquer_by_configurable_csr without pruning.  * @tparam Structure Insertable represented state.
	 * @tparam Lifetimes Const sized forward range of item intervals.
	 * @tparam Insert Item-index callback. @tparam Visit Time-index callback.
	 * @tparam ShouldVisit Time predicate. @tparam Mode Restoration policy.
	 * @param structure Initial state, restored on normal completion.
	 * @param time_count Number of times.
	 * @param lifetimes Half-open interval of each item.
	 * @param insert Callback applying an item index.
	 * @param visit Callback observing a selected time.
	 * @param should_visit Predicate selecting times.
	 * @param mode Rollback callbacks or copy tag.
	 * @pre The bounds and callback laws of atl::time_divide_and_conquer apply;
	 *      time_divide_and_conquer_by_recursive_vectors additionally requires T <= 2^29.
	 * @complexity O(T + m log(T + 1)) engine storage and setup, plus insertion,
	 *             visit and restoration work; rank compression uses Q leaves.
	 */
	template<typename Structure, lifetime_range Lifetimes, item_insertion<Structure> Insert,
			 time_visit<Structure> Visit, time_predicate ShouldVisit = every_time,
			 restore_mode<Structure> Mode = default_restore_t<Structure>>
	constexpr void time_divide_and_conquer_by_unpruned_csr(Structure &structure, std::size_t time_count,
														   Lifetimes const &lifetimes, Insert insert, Visit visit,
														   ShouldVisit should_visit = {},
														   Mode mode = default_restore<Structure>()) {
		time_divide_and_conquer_by_configurable_csr<false, true>(structure, time_count, lifetimes, std::move(insert), std::move(visit),
																 std::move(should_visit), std::move(mode));
	}

	/**
	 * @brief Dense CSR with pruning and suffix stretching, without rank compression. Lost on churn
	 *        n=q=3e5 at 1% queries: rollback 34.6/32.9 vs 13.8/13.9 ms;
	 *        at 10%: 29.6/28.5 vs 17.6/17.7 ms. Copy mode at 1%: 548/552
	 *        vs 510/517 ms; at 10%: 573/577 vs 550/557 ms. Kept as the
	 *        dense fallback because ranking every time costs an extra pass and
	 *        buffer: edge toggles n=q=1e5, 11.1/10.7 vs 10.8/11.4 ms.
	 *        Numbers plain -O2 / -march=native, Xeon 8280, 2026-11-01,
	 *        benchmarks/ds/time_divide_and_conquer.cpp.
	 * @tparam Structure Insertable represented state.
	 * @tparam Lifetimes Const sized forward range of item intervals.
	 * @tparam Insert Item-index callback. @tparam Visit Time-index callback.
	 * @tparam ShouldVisit Time predicate. @tparam Mode Restoration policy.
	 * @param structure Initial state, restored on normal completion.
	 * @param time_count Number of times.
	 * @param lifetimes Half-open interval of each item.
	 * @param insert Callback applying an item index.
	 * @param visit Callback observing a selected time.
	 * @param should_visit Predicate selecting times.
	 * @param mode Rollback callbacks or copy tag.
	 * @pre The bounds and callback laws of atl::time_divide_and_conquer apply;
	 *      time_divide_and_conquer_by_recursive_vectors additionally requires T <= 2^29.
	 * @complexity O(T + m log(T + 1)) engine storage and setup, plus insertion,
	 *             visit and restoration work; rank compression uses Q leaves.
	 */
	template<typename Structure, lifetime_range Lifetimes, item_insertion<Structure> Insert,
			 time_visit<Structure> Visit, time_predicate ShouldVisit = every_time,
			 restore_mode<Structure> Mode = default_restore_t<Structure>>
	constexpr void time_divide_and_conquer_by_dense_csr(Structure &structure, std::size_t time_count,
														Lifetimes const &lifetimes, Insert insert, Visit visit,
														ShouldVisit should_visit = {},
														Mode mode = default_restore<Structure>()) {
		time_divide_and_conquer_by_configurable_csr<true, true>(structure, time_count, lifetimes, std::move(insert), std::move(visit),
																std::move(should_visit), std::move(mode));
	}

	/**
	 * @brief Frame skipping: lost on churn n=q=3e5 at 50% queries,
	 *        rollback 21.6/20.8 vs dense CSR's 20.6/19.4 ms.
	 *        No by_copy gain: persistent_dsu 490/500 vs 487/497 ms at 50%.
	 *        Numbers plain -O2 / -march=native, Xeon 8280, 2026-11-01,
	 *        benchmarks/ds/time_divide_and_conquer.cpp.
	 *        Opens no frame when its inserts can be discarded by an enclosing
	 *        frame's pop; the proof is in @ref detail::visit_time_tree_by_frame_skipping.
	 * @tparam Structure Insertable represented state.
	 * @tparam Lifetimes Const sized forward range of item intervals.
	 * @tparam Insert Item-index callback. @tparam Visit Time-index callback.
	 * @tparam ShouldVisit Time predicate. @tparam Mode Restoration policy.
	 * @param structure Initial state, restored on normal completion.
	 * @param time_count Number of times.
	 * @param lifetimes Half-open interval of each item.
	 * @param insert Callback applying an item index.
	 * @param visit Callback observing a selected time.
	 * @param should_visit Predicate selecting times.
	 * @param mode Rollback callbacks or copy tag.
	 * @pre The bounds and callback laws of atl::time_divide_and_conquer apply;
	 *      time_divide_and_conquer_by_recursive_vectors additionally requires T <= 2^29.
	 * @complexity O(T + m log(T + 1)) engine storage and setup, plus insertion,
	 *             visit and restoration work; rank compression uses Q leaves.
	 */
	template<typename Structure, lifetime_range Lifetimes, item_insertion<Structure> Insert,
			 time_visit<Structure> Visit, time_predicate ShouldVisit = every_time,
			 restore_mode<Structure> Mode = default_restore_t<Structure>>
	constexpr void time_divide_and_conquer_by_frame_skipping(Structure &structure, std::size_t time_count,
															 Lifetimes const &lifetimes, Insert insert, Visit visit,
															 ShouldVisit should_visit = {},
															 Mode mode = default_restore<Structure>()) {
		time_divide_and_conquer_by_configurable_csr<true, true, true>(structure, time_count, lifetimes, std::move(insert),
																	  std::move(visit), std::move(should_visit), std::move(mode));
	}

	/**
	 * @brief Rank compression plus frame skipping: lost to rank compression
	 *        alone. No by_copy gain (churn n=q=3e5 at 1% queries, 509/518 vs
	 *        510/517 ms), and rollback is within noise on plain -O2 (50%:
	 *        19.2 vs 19.3 ms; 1%: 13.5 vs 13.8 ms). Numbers plain -O2 /
	 *        -march=native, Xeon 8280, 2026-11-01,
	 *        benchmarks/ds/time_divide_and_conquer.cpp.
	 *        The tree uses Q visited times: [b,e) becomes [rank[b],rank[e]),
	 *        and leaf k visits the original time times[k]. Ranking costs an
	 *        extra O(T) prefix pass and buffer, even on dense workloads.
	 * @pre as @ref atl::time_divide_and_conquer (asserted)
	 * @tparam Structure Insertable represented state.
	 * @tparam Lifetimes Const sized forward range of item intervals.
	 * @tparam Insert Item-index callback. @tparam Visit Time-index callback.
	 * @tparam ShouldVisit Time predicate. @tparam Mode Restoration policy.
	 * @param structure Initial state, restored on normal completion.
	 * @param time_count Number of times.
	 * @param lifetimes Half-open interval of each item.
	 * @param insert Callback applying an item index.
	 * @param visit Callback observing a selected time.
	 * @param should_visit Predicate selecting times.
	 * @param mode Rollback callbacks or copy tag.
	 * @pre The bounds and callback laws of atl::time_divide_and_conquer apply;
	 *      time_divide_and_conquer_by_recursive_vectors additionally requires T <= 2^29.
	 * @complexity O(T + m log(T + 1)) engine storage and setup, plus insertion,
	 *             visit and restoration work; rank compression uses Q leaves.
	 */
	template<typename Structure, lifetime_range Lifetimes, item_insertion<Structure> Insert,
			 time_visit<Structure> Visit, time_predicate ShouldVisit = every_time,
			 restore_mode<Structure> Mode = default_restore_t<Structure>>
	constexpr void time_divide_and_conquer_by_ranked_frame_skipping(Structure &structure, std::size_t time_count,
																	Lifetimes const &lifetimes, Insert insert, Visit visit,
																	ShouldVisit should_visit = {},
																	Mode mode = default_restore<Structure>()) {
		atl_assert(time_count <= (std::size_t(1) << 30));
		atl_assert(uint64(std::ranges::size(lifetimes)) < uint64((std::numeric_limits<uint32>::max)()));
		std::vector<uint32> rank(time_count + 1, 0), times;
		for (uint32 t = 0; t < uint32(time_count); ++t) {
			bool const is_visited = std::invoke(should_visit, t);
			if (is_visited) times.push_back(t);
			rank[t + 1] = rank[t] + uint32(is_visited);
		}
		if (times.empty()) return;
		auto const ranked = std::views::transform(std::ranges::ref_view(lifetimes), [&]<typename Lifetime>(Lifetime const &lifetime)
													  requires(atl::detail::member_lifetime<Lifetime> || atl::detail::pair_lifetime<Lifetime>)
												  {
													  auto const [begin, end] = atl::detail::lifetime_bounds(lifetime);
													  atl_assert(begin <= end && end <= time_count);
													  return item_lifetime{rank[begin], rank[end]};
												  });
		time_divide_and_conquer_by_configurable_csr<true, true, true>(
			structure, times.size(), ranked, std::move(insert),
			[&](Structure &s, uint32 k) { std::invoke(visit, s, times[k]); }, every_time{}, std::move(mode));
	}
}
