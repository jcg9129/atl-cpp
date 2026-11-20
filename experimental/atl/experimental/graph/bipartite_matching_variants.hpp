#pragma once
#include <atl/graph/bipartite_matching.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/graph/detail/edge_count.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstddef>
#include <ranges>
#include <vector>

/**
 * @file
 * @brief Reference bipartite-matching implementations benchmarked against
 *        the library auction + Hopcroft–Karp hybrid
 *        (atl::graph::bipartite_matching).
 *
 * Same contract: a @ref atl::graph::bipartite_matching_result over two
 * separate 0-based vertex ranges.
 *
 * Measured on random, banded, line, augmented-cycle, skewed and dense graphs (benchmarks/graph/bipartite_matching.cpp,
 * record 2026-10-15, x86-64 Xeon Platinum 8358, GCC 15.2,
 * RelWithDebInfo, assertions off, -O2 -g -DNDEBUG -std=c++23): the auction phase
 * is what wins augmented cycle — 33.6 ms against pure Hopcroft–Karp's 404,
 * the pass-structured Kuhn's 345 and Kuhn's 1385 — while all three are ahead
 * of it on the easy and dense shapes (dense 3e3+3e3 m=3e6: 17.8-18.9 ms
 * against 55.7). The per-root-epoch Kuhn also detonates on skewed sides
 * (692 ms against 3.8 for the pass-structured one).
 */
namespace atl::experimental {
	/**
	 * @brief Pure Hopcroft–Karp with a greedy warm start: BFS layers the
	 *        left side from its free vertices (labeling everything reachable
	 *        — the textbook first-free-level cutoff was measured far worse),
	 *        then one DFS pass augments along vertex-disjoint
	 *        label-increasing paths.
	 *
	 * On augmented cycles this takes 404 ms versus the auction hybrid's 33.6 ms.
	 * @tparam EdgeList Const forward edge range, with stable iteration.
	 * @param left Number of left vertices.
	 * @param right Number of right vertices.
	 * @param edges Edges between the separate ranges; parallel copies permitted.
	 * @return Owning maximum matching; ties are unspecified.
	 * @pre left, right < graph::none, |edges| < 2^32, endpoints in range (asserted).
	 * @note Not constexpr (runtime contender). Uses explicit DFS stacks and O(left + right + m) heap storage; O(1) call stack.
	 * @complexity O(left + right + m sqrt(left + right))
	 */
	template<graph::edge_list EdgeList>
	[[nodiscard]] graph::bipartite_matching_result bipartite_matching_by_hopcroft_karp(std::size_t left, std::size_t right,
																					   EdgeList const &edges) {
		auto const m = graph::detail::checked_edge_count(edges, graph::none);
		atl_assert(left < graph::none && right < graph::none && m < (uint64(1) << 32)); // uint32 ids and offsets
		std::vector<uint32> offset(left + 1, 0), to(m);
		for (auto const &e : edges) {
			atl_assert(graph::source(e) < left && graph::target(e) < right);
			++offset[graph::source(e) + 1];
		}
		for (std::size_t v = 0; v < left; ++v) offset[v + 1] += offset[v];
		{
			std::vector<uint32> cursor(offset.begin(), offset.end() - 1);
			for (auto const &e : edges) to[cursor[graph::source(e)]++] = graph::target(e);
		}

		graph::bipartite_matching_result r{0, std::vector<uint32>(left, graph::none), std::vector<uint32>(right, graph::none)};
		auto &ml = r.left_pair;
		auto &mr = r.right_pair;
		for (uint32 l = 0; l < left; ++l)
			for (uint32 k = offset[l]; k < offset[l + 1] && ml[l] == graph::none; ++k)
				if (mr[to[k]] == graph::none) {
					ml[l] = to[k];
					mr[to[k]] = l;
				}

		std::vector<uint32> dist(left), it(left), queue, stack, chosen;
		queue.reserve(left);
		for (;;) {
			std::ranges::fill(dist, graph::none);
			queue.clear();
			for (uint32 l = 0; l < left; ++l)
				if (ml[l] == graph::none) {
					dist[l] = 0;
					queue.push_back(l);
				}
			bool is_free_right_reachable = false;
			for (std::size_t at = 0; at < queue.size(); ++at) {
				uint32 const l = queue[at];
				for (uint32 k = offset[l]; k < offset[l + 1]; ++k) {
					uint32 const l2 = mr[to[k]];
					if (l2 == graph::none)
						is_free_right_reachable = true;
					else if (dist[l2] == graph::none) {
						dist[l2] = dist[l] + 1;
						queue.push_back(l2);
					}
				}
			}
			if (!is_free_right_reachable) break;
			std::ranges::copy(offset.begin(), offset.end() - 1, it.begin());
			for (uint32 root = 0; root < left; ++root) {
				if (ml[root] != graph::none) continue;
				stack.assign(1, root); // chosen[i] is the right vertex stack[i] picked
				chosen.clear();
				while (!stack.empty()) {
					uint32 const l = stack.back();
					bool has_advanced = false;
					while (it[l] < offset[l + 1]) {
						uint32 const w = to[it[l]++];
						uint32 const l2 = mr[w];
						if (l2 == graph::none) {
							chosen.push_back(w);
							for (std::size_t i = stack.size(); i-- > 0;) {
								ml[stack[i]] = chosen[i];
								mr[chosen[i]] = stack[i];
							}
							stack.clear();
							chosen.clear();
							has_advanced = true;
							break;
						}
						if (dist[l2] == dist[l] + 1) {
							chosen.push_back(w);
							stack.push_back(l2);
							has_advanced = true;
							break;
						}
					}
					if (!has_advanced) {
						dist[l] = graph::none;
						stack.pop_back();
						if (!chosen.empty()) chosen.pop_back();
					}
				}
			}
		}
		for (uint32 l = 0; l < left; ++l) r.size += ml[l] != graph::none;
		return r;
	}

	/**
	 * @brief Kuhn's algorithm: one augmenting DFS per left vertex over the
	 *        current matching, with a greedy warm start.
	 *
	 * Augmented cycles take 1385 ms; skewed sides take 692 ms, versus
	 * 33.6 and 4.8 ms for the library (Xeon Platinum 8358, GCC 15.2,
	 * RelWithDebInfo, -O2, assertions off, 2026-10-15). The
	 * family's designated oracle is the certified max_flow reduction in the
	 * tests, not this one.
	 * @tparam EdgeList Const forward edge range, with stable iteration.
	 * @param left Number of left vertices.
	 * @param right Number of right vertices.
	 * @param edges Edges between the separate ranges; parallel copies permitted.
	 * @return Owning maximum matching; ties are unspecified.
	 * @pre left, right < graph::none, |edges| < 2^32, endpoints in range (asserted).
	 * @note Not constexpr (runtime contender). Uses explicit DFS stacks and O(left + right + m) heap storage; O(1) call stack.
	 * @complexity O(n m) worst case
	 */
	template<graph::edge_list EdgeList>
	[[nodiscard]] graph::bipartite_matching_result bipartite_matching_by_kuhn(std::size_t left, std::size_t right,
																			  EdgeList const &edges) {
		auto const m = graph::detail::checked_edge_count(edges, graph::none);
		atl_assert(left < graph::none && right < graph::none && m < (uint64(1) << 32)); // uint32 ids and offsets
		std::vector<uint32> offset(left + 1, 0), to(m);
		for (auto const &e : edges) {
			atl_assert(graph::source(e) < left && graph::target(e) < right);
			++offset[graph::source(e) + 1];
		}
		for (std::size_t v = 0; v < left; ++v) offset[v + 1] += offset[v];
		{
			std::vector<uint32> cursor(offset.begin(), offset.end() - 1);
			for (auto const &e : edges) to[cursor[graph::source(e)]++] = graph::target(e);
		}

		graph::bipartite_matching_result r{0, std::vector<uint32>(left, graph::none), std::vector<uint32>(right, graph::none)};
		auto &ml = r.left_pair;
		auto &mr = r.right_pair;
		for (uint32 l = 0; l < left; ++l)
			for (uint32 k = offset[l]; k < offset[l + 1] && ml[l] == graph::none; ++k)
				if (mr[to[k]] == graph::none) {
					ml[l] = to[k];
					mr[to[k]] = l;
				}

		std::vector<uint32> stamp(left, 0), it(left), stack, chosen;
		uint32 round = 0;
		for (uint32 root = 0; root < left; ++root) {
			if (ml[root] != graph::none) continue;
			++round;
			it[root] = offset[root];
			stamp[root] = round;
			stack.assign(1, root); // chosen[i] is the right vertex stack[i] picked
			chosen.clear();
			while (!stack.empty()) {
				uint32 const l = stack.back();
				bool has_advanced = false;
				while (it[l] < offset[l + 1]) {
					uint32 const w = to[it[l]++];
					uint32 const l2 = mr[w];
					if (l2 == graph::none) {
						chosen.push_back(w);
						for (std::size_t i = stack.size(); i-- > 0;) {
							ml[stack[i]] = chosen[i];
							mr[chosen[i]] = stack[i];
						}
						stack.clear();
						chosen.clear();
						has_advanced = true;
						break;
					}
					if (stamp[l2] != round) {
						stamp[l2] = round;
						it[l2] = offset[l2];
						chosen.push_back(w);
						stack.push_back(l2);
						has_advanced = true;
						break;
					}
				}
				if (!has_advanced) {
					stack.pop_back();
					if (!chosen.empty()) chosen.pop_back();
				}
			}
		}
		for (uint32 l = 0; l < left; ++l) r.size += ml[l] != graph::none;
		return r;
	}

	/**
	 * @brief Kuhn's algorithm in passes: each pass tries every free left once
	 *        with ONE visited stamp shared by all its searches, and passes
	 *        repeat until one augments nothing.
	 *
	 * Within a pass, a left a failed search marked stays dead until the
	 * matching changes, and keeping it marked after an augmentation only
	 * defers it to the next pass, so a pass costs O(m) and the last,
	 * augmentation-free pass (with fresh marks) proves maximality. This
	 * removes the per-root re-exploration that makes the per-root-epoch
	 * @ref atl::experimental::bipartite_matching_by_kuhn quadratic on skewed sides.
	 *
	 * Measured (record 2026-10-15, x86-64 Xeon Platinum 8358, plain
	 * -O2): skewed 1e5 to 1e3 3.8 ms (Kuhn 692, library 4.8), but augmented
	 * cycle 345 ms against the library auction's 33.6, and random 18.4
	 * against 15.9.
	 * @tparam EdgeList Const forward edge range, with stable iteration.
	 * @param left Number of left vertices.
	 * @param right Number of right vertices.
	 * @param edges Edges between the separate ranges; parallel copies permitted.
	 * @return Owning maximum matching; ties are unspecified.
	 * @pre left, right < graph::none, |edges| < 2^32, endpoints in range (asserted).
	 * @note Not constexpr (runtime contender). Uses explicit DFS stacks and O(left + right + m) heap storage; O(1) call stack.
	 * @complexity O(n m) worst case (O(m) per pass)
	 */
	template<graph::edge_list EdgeList>
	[[nodiscard]] graph::bipartite_matching_result bipartite_matching_by_kuhn_passes(std::size_t left, std::size_t right,
																					 EdgeList const &edges) {
		auto const m = graph::detail::checked_edge_count(edges, graph::none);
		atl_assert(left < graph::none && right < graph::none && m < (uint64(1) << 32)); // uint32 ids and offsets
		std::vector<uint32> offset(left + 1, 0), to(m);
		for (auto const &e : edges) {
			atl_assert(graph::source(e) < left && graph::target(e) < right);
			++offset[graph::source(e) + 1];
		}
		for (std::size_t v = 0; v < left; ++v) offset[v + 1] += offset[v];
		{
			std::vector<uint32> cursor(offset.begin(), offset.end() - 1);
			for (auto const &e : edges) to[cursor[graph::source(e)]++] = graph::target(e);
		}

		graph::bipartite_matching_result r{0, std::vector<uint32>(left, graph::none), std::vector<uint32>(right, graph::none)};
		auto &ml = r.left_pair;
		auto &mr = r.right_pair;
		for (uint32 l = 0; l < left; ++l)
			for (uint32 k = offset[l]; k < offset[l + 1] && ml[l] == graph::none; ++k)
				if (mr[to[k]] == graph::none) {
					ml[l] = to[k];
					mr[to[k]] = l;
				}

		std::vector<uint32> stamp(left, 0), it(left), stack, chosen;
		uint32 round = 0;
		for (bool has_augmented = true; has_augmented;) {
			has_augmented = false;
			++round;
			for (uint32 root = 0; root < left; ++root) {
				if (ml[root] != graph::none || stamp[root] == round) continue;
				it[root] = offset[root];
				stamp[root] = round;
				stack.assign(1, root); // chosen[i] is the right vertex stack[i] picked
				chosen.clear();
				while (!stack.empty()) {
					uint32 const l = stack.back();
					bool has_advanced = false;
					while (it[l] < offset[l + 1]) {
						uint32 const w = to[it[l]++];
						uint32 const l2 = mr[w];
						if (l2 == graph::none) {
							chosen.push_back(w);
							for (std::size_t i = stack.size(); i-- > 0;) {
								ml[stack[i]] = chosen[i];
								mr[chosen[i]] = stack[i];
							}
							stack.clear();
							chosen.clear();
							has_advanced = has_augmented = true;
							break;
						}
						if (stamp[l2] != round) {
							stamp[l2] = round;
							it[l2] = offset[l2];
							chosen.push_back(w);
							stack.push_back(l2);
							has_advanced = true;
							break;
						}
					}
					if (!has_advanced) {
						stack.pop_back();
						if (!chosen.empty()) chosen.pop_back();
					}
				}
			}
		}
		for (uint32 l = 0; l < left; ++l) r.size += ml[l] != graph::none;
		return r;
	}
}
