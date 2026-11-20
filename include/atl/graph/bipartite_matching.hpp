#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/detail/edge_count.hpp>
#include <atl/graph/partition.hpp>
#include <atl/graph/scc.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <limits>
#include <ranges>
#include <utility>
#include <vector>

namespace atl::graph {
	/**
	 * @brief What @ref atl::graph::bipartite_matching computes: a maximum matching as the two
	 * pair arrays (`none` = unmatched).
	 */
	struct bipartite_matching_result {
		std::size_t size = 0;			///< maximum matching cardinality
		std::vector<uint32> left_pair;	///< matched right vertex per left vertex, `none` if unmatched
		std::vector<uint32> right_pair; ///< matched left vertex per right vertex, `none` if unmatched
	};

	namespace detail {
		/**
		 * @brief The @ref atl::graph::bipartite_matching engine on a prebuilt adjacency in
		 *        both directions: left l's rights are
		 *        `left_target[left_offset[l] .. left_offset[l + 1])`, right
		 *        w's lefts `right_target[right_offset[w] .. right_offset[w + 1])`.
		 *        Lets @ref atl::graph::minimum_chain_cover hand over the closure's CSR
		 *        without materializing it as an edge list.
		 * @pre the two directions hold the same edges, endpoints in range
		 */
		[[nodiscard]] constexpr bipartite_matching_result bipartite_matching_csr(std::size_t left, std::size_t right,
																				 std::vector<uint32> const &left_offset,
																				 std::vector<uint32> const &left_target,
																				 std::vector<uint32> const &right_offset,
																				 std::vector<uint32> const &right_target) {
			atl_assert(left < none && right < none && left_target.size() <= none);
			bipartite_matching_result r{0, std::vector<uint32>(left, none), std::vector<uint32>(right, none)};
			auto &ml = r.left_pair;
			auto &mr = r.right_pair;
			std::size_t const lim = (std::min)(left, right);
			std::size_t size = 0;

			std::vector<uint64> lv(left);
			std::vector<uint32> queue;
			auto constexpr infinite_level = (std::numeric_limits<uint64>::max)();
			queue.reserve(left);
			std::vector<uint32> ring(right + 1); // free rights, each queued at most once
			std::size_t head = 0, tail = 0, live = 0;
			auto const push = [&](uint32 w) {
				ring[tail] = w;
				if (++tail == ring.size()) tail = 0;
				++live;
			};
			auto const pop = [&] {
				uint32 const w = ring[head];
				if (++head == ring.size()) head = 0;
				--live;
				return w;
			};
			for (uint32 w = 0; w < right; ++w) push(w);

			auto const global_relabel = [&] {
				std::ranges::fill(lv, infinite_level);
				queue.clear();
				for (uint32 l = 0; l < left; ++l)
					if (ml[l] == none) {
						lv[l] = 0;
						queue.push_back(l);
					}
				for (std::size_t at = 0; at < queue.size(); ++at) {
					uint32 const l = queue[at];
					for (uint32 k = left_offset[l]; k < left_offset[l + 1]; ++k) {
						uint32 const u = mr[left_target[k]];
						if (u != none && lv[u] > lv[l] + 2) {
							lv[u] = lv[l] + 2;
							queue.push_back(u);
						}
					}
				}
			};

			uint64 const period = uint64(left) + right;
			uint64 ticks = 0;
			while (live > 0 && size < lim) {
				if (ticks == 0) global_relabel();
				if (++ticks == period) ticks = 0;
				uint32 const w = pop();
				uint32 best = none;
				uint64 best_lv = infinite_level;
				for (uint32 k = right_offset[w]; k < right_offset[w + 1]; ++k)
					if (lv[right_target[k]] < best_lv) {
						best_lv = lv[right_target[k]];
						best = right_target[k];
					}
				if (best_lv == infinite_level) continue; // no left reaches a free one: w rests until HK
				uint32 const old = ml[best];
				if (old != none) {
					mr[old] = none;
					push(old);
				} else {
					++size;
				}
				ml[best] = w;
				mr[w] = best;
				lv[best] = best_lv + 2;
			}

			// Hopcroft-Karp finish: guarantees maximum cardinality. Phases label
			// everything reachable (no first-free-level cutoff): full labels let
			// the DFS harvest longer label-increasing paths too.
			std::vector<uint32> dist(left), it(left), stack, chosen;
			for (;;) {
				std::ranges::fill(dist, none);
				queue.clear();
				for (uint32 l = 0; l < left; ++l)
					if (ml[l] == none) {
						dist[l] = 0;
						queue.push_back(l);
					}
				bool is_free_right_reachable = false;
				for (std::size_t at = 0; at < queue.size(); ++at) {
					uint32 const l = queue[at];
					for (uint32 k = left_offset[l]; k < left_offset[l + 1]; ++k) {
						uint32 const l2 = mr[left_target[k]];
						if (l2 == none)
							is_free_right_reachable = true;
						else if (dist[l2] == none) {
							dist[l2] = dist[l] + 1;
							queue.push_back(l2);
						}
					}
				}
				if (!is_free_right_reachable) break;
				std::ranges::copy(left_offset.begin(), left_offset.end() - 1, it.begin());
				for (uint32 root = 0; root < left; ++root) {
					if (ml[root] != none) continue;
					stack.assign(1, root); // chosen[i] is the right vertex stack[i] picked
					chosen.clear();
					while (!stack.empty()) {
						uint32 const l = stack.back();
						bool has_advanced = false;
						while (it[l] < left_offset[l + 1]) {
							uint32 const w = left_target[it[l]++];
							uint32 const l2 = mr[w];
							if (l2 == none) {
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
							dist[l] = none;
							stack.pop_back();
							if (!chosen.empty()) chosen.pop_back();
						}
					}
				}
			}
			for (uint32 l = 0; l < left; ++l) r.size += ml[l] != none;
			return r;
		}
	}

	/**
	 * @brief Maximum matching over separate left and right vertex ranges.
	 * @tparam EdgeList Const forward edge range; parallel edges are permitted.
	 * @param left Number of left vertices.
	 * @param right Number of right vertices.
	 * @param edges Edges with source in `[0, left)` and target in `[0, right)`.
	 * @return Owning mutual pair arrays with `none` for unmatched vertices and their cardinality.
	 * No particular maximum matching is selected when several exist. Input order
	 * is preserved within adjacency rows; no input references are retained.
	 * @pre Both side counts are less than `none`; the edge count is at most `none`
	 * and endpoints are in range (asserted). Edge iteration is stable during the call.
	 * @complexity O(V^2 (V + m)) conservative worst-case time for the uncapped
	 * auction, V = left + right; the layered finish takes O(V + m sqrt(V)).
	 * O(left + right + m)
	 * heap storage. All searches use queues or explicit stacks, including on
	 * million-vertex alternating paths. Auction levels and relabel periods use
	 * uint64: between relabels levels are below `4 * left + 2 * right + 2`.
	 * Allocation failures propagate.
	 */
	template<edge_list EdgeList>
	[[nodiscard]] constexpr bipartite_matching_result bipartite_matching(std::size_t left, std::size_t right,
																		 EdgeList const &edges) {
		auto const m = detail::checked_edge_count(edges, none);
		atl_assert(left < none && right < none && m <= none);
		std::vector<uint32> left_offset(left + 1, 0), left_target(m), right_offset(right + 1, 0), right_target(m);
		for (auto const &e : edges) {
			atl_assert(graph::source(e) < left && graph::target(e) < right);
			++left_offset[graph::source(e) + 1];
			++right_offset[graph::target(e) + 1];
		}
		for (std::size_t v = 0; v < left; ++v) left_offset[v + 1] += left_offset[v];
		for (std::size_t v = 0; v < right; ++v) right_offset[v + 1] += right_offset[v];
		{
			std::vector<uint32> cl(left_offset.begin(), left_offset.end() - 1), cr(right_offset.begin(), right_offset.end() - 1);
			for (auto const &e : edges) {
				left_target[cl[graph::source(e)]++] = graph::target(e);
				right_target[cr[graph::target(e)]++] = graph::source(e);
			}
		}
		return detail::bipartite_matching_csr(left, right, left_offset, left_target, right_offset, right_target);
	}

	/**
	 * @brief A choice of vertices on both sides of a bipartite graph — what
	 * @ref atl::graph::minimum_vertex_cover and @ref atl::graph::maximum_independent_set produce.
	 */
	struct bipartite_vertex_set {
		std::size_t size = 0;	  ///< number of chosen vertices
		std::vector<uint8> left;  ///< 0/1: left vertex chosen
		std::vector<uint8> right; ///< 0/1: right vertex chosen
	};

	namespace detail {
		/**
		 * @brief The @ref atl::graph::minimum_vertex_cover construction on a prebuilt
		 *        left adjacency `to[offset[l] .. offset[l + 1])`.
		 * @pre @p matching is maximum for these edges (cardinality checked for a valid matching)
		 */
		[[nodiscard]] constexpr bipartite_vertex_set vertex_cover_csr(std::size_t left, std::size_t right,
																	  std::vector<uint32> const &offset, std::vector<uint32> const &to,
																	  bipartite_matching_result const &matching) {
			// alternating BFS from the free lefts: left -> any unmatched edge ->
			// right -> its matched left. (Scanning the one matched edge too is a
			// no-op: its endpoint is already in Z.)
			std::vector<uint8> zl(left, 0), zr(right, 0);
			std::vector<uint32> queue;
			queue.reserve(left);
			for (uint32 l = 0; l < left; ++l)
				if (matching.left_pair[l] == none) {
					zl[l] = 1;
					queue.push_back(l);
				}
			for (std::size_t at = 0; at < queue.size(); ++at) {
				uint32 const l = queue[at];
				for (uint32 k = offset[l]; k < offset[l + 1]; ++k) {
					uint32 const w = to[k];
					if (zr[w]) continue;
					zr[w] = 1;
					uint32 const l2 = matching.right_pair[w];
					if (l2 != none && !zl[l2]) {
						zl[l2] = 1;
						queue.push_back(l2);
					}
				}
			}

			bipartite_vertex_set cover{0, std::move(zl), std::move(zr)};
			for (uint32 l = 0; l < left; ++l) {
				cover.left[l] = !cover.left[l];
				cover.size += cover.left[l];
			}
			for (uint32 w = 0; w < right; ++w) cover.size += cover.right[w];
			atl_assert(cover.size == matching.size); // else the matching was not maximum
			return cover;
		}
	}

	/**
	 * @brief Minimum vertex cover by alternating reachability (König's theorem).
	 * @tparam EdgeList Const forward edge range.
	 * @param left Number of left vertices, less than `none`.
	 * @param right Number of right vertices, less than `none`.
	 * @param edges Stable input edges, at most `none`, with endpoints in range.
	 * @param matching A valid maximum matching of this graph.
	 * @return Owning 0/1 masks with size equal to the maximum matching's size.
	 * @pre Side counts, edge count, endpoints and pair-array sizes are asserted;
	 * validity and mutual consistency of the supplied matching are documented only.
	 * For a valid matching, equality of cover and matching sizes asserts maximum cardinality.
	 * @complexity O(left + right + m) time and heap storage; O(1) call stack.
	 */
	template<edge_list EdgeList>
	[[nodiscard]] constexpr bipartite_vertex_set minimum_vertex_cover(std::size_t left, std::size_t right, EdgeList const &edges,
																	  bipartite_matching_result const &matching) {
		atl_assert(matching.left_pair.size() == left && matching.right_pair.size() == right);
		auto const m = detail::checked_edge_count(edges, none);
		atl_assert(left < none && right < none && m <= none);
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
		return detail::vertex_cover_csr(left, right, offset, to, matching);
	}

	/**
	 * @brief Complement of the minimum vertex cover, with owning 0/1 masks.
	 * @tparam EdgeList Const forward edge range.
	 * @param left Number of left vertices.
	 * @param right Number of right vertices.
	 * @param edges The same graph used to compute the matching.
	 * @param matching A valid maximum matching of this graph.
	 * @return A set of size `left + right - matching.size` containing no edge.
	 * @pre The minimum_vertex_cover preconditions hold, and `left + right` fits
	 * size_t (asserted). Matching validity and stable iteration are documented only.
	 * @complexity O(left + right + m) time and heap storage; O(1) call stack.
	 */
	template<edge_list EdgeList>
	[[nodiscard]] constexpr bipartite_vertex_set maximum_independent_set(std::size_t left, std::size_t right, EdgeList const &edges,
																		 bipartite_matching_result const &matching) {
		atl_assert(right <= (std::numeric_limits<std::size_t>::max)() - left);
		auto s = minimum_vertex_cover(left, right, edges, matching);
		for (auto &c : s.left) c = !c;
		for (auto &c : s.right) c = !c;
		s.size = left + right - s.size;
		return s;
	}

	namespace detail {
		/**
		 * @brief Whether the edge list is a DAG — the assert-only guard of the
		 *        cover functions (Kahn's algorithm; zero cost in release
		 *        builds).
		 * @pre every endpoint is in `[0, n)` (asserted)
		 * @complexity O(n + m)
		 */
		template<edge_list EdgeList>
		[[nodiscard]] constexpr bool acyclic(std::size_t n, EdgeList const &edges) {
			auto const m = detail::checked_edge_count(edges, none);
			atl_assert(n < none && m <= none);
			std::vector<uint32> offset(n + 1, 0), to(m);
			std::vector<uint32> indegree(n, 0);
			for (auto const &e : edges) {
				atl_assert(graph::source(e) < n && graph::target(e) < n);
				++offset[graph::source(e) + 1];
				++indegree[graph::target(e)];
			}
			for (std::size_t v = 0; v < n; ++v) offset[v + 1] += offset[v];
			{
				std::vector<uint32> cursor(offset.begin(), offset.end() - 1);
				for (auto const &e : edges) to[cursor[graph::source(e)]++] = graph::target(e);
			}
			std::vector<uint32> queue;
			queue.reserve(n);
			for (uint32 v = 0; v < n; ++v)
				if (indegree[v] == 0) queue.push_back(v);
			for (std::size_t at = 0; at < queue.size(); ++at)
				for (uint32 k = offset[queue[at]]; k < offset[queue[at] + 1]; ++k)
					if (--indegree[to[k]] == 0) queue.push_back(to[k]);
			return queue.size() == n;
		}

		/**
		 * @brief Transitive closure of a DAG as a CSR, `to[offset[u] ..
		 *        offset[u + 1])` the vertices u reaches in increasing order —
		 *        one bitset row per vertex, filled in reverse topological order
		 *        so every edge to a not-yet-reached target costs n/64 word-ORs (an edge
		 *        to a reached one is skipped: its row is already contained).
		 * @pre @p edges form a DAG on `[0, n)` (both asserted)
		 * @complexity O(n + m + (n + m) ceil(n/64) + q) time and O(n ceil(n/64) + m + q) storage, where q is the closure edge count
		 */
		template<edge_list EdgeList>
		[[nodiscard]] constexpr std::pair<std::vector<uint32>, std::vector<uint32>> closure_csr(std::size_t n,
																								EdgeList const &edges) {
			auto const m = detail::checked_edge_count(edges, none);
			atl_assert(n < none && m <= none);
			std::vector<uint32> offset(n + 1, 0), to(m);
			std::vector<uint32> indegree(n, 0);
			for (auto const &e : edges) {
				atl_assert(graph::source(e) < n && graph::target(e) < n);
				++offset[graph::source(e) + 1];
				++indegree[graph::target(e)];
			}
			for (std::size_t v = 0; v < n; ++v) offset[v + 1] += offset[v];
			{
				std::vector<uint32> cursor(offset.begin(), offset.end() - 1);
				for (auto const &e : edges) to[cursor[graph::source(e)]++] = graph::target(e);
			}
			std::vector<uint32> topo;
			topo.reserve(n);
			for (uint32 v = 0; v < n; ++v)
				if (indegree[v] == 0) topo.push_back(v);
			for (std::size_t at = 0; at < topo.size(); ++at)
				for (uint32 k = offset[topo[at]]; k < offset[topo[at] + 1]; ++k)
					if (--indegree[to[k]] == 0) topo.push_back(to[k]);
			atl_assert(topo.size() == n); // a cycle makes chains meaningless

			std::size_t const words = n / 64 + (n % 64 != 0);
			atl_assert(words == 0 || n <= std::vector<uint64>().max_size() / words);
			std::vector<uint64> reach(n * words, 0);
			for (std::size_t i = topo.size(); i-- > 0;) {
				uint32 const u = topo[i];
				for (uint32 k = offset[u]; k < offset[u + 1]; ++k) {
					uint32 const v = to[k];
					uint64 &word = reach[u * words + v / 64];
					uint64 const bit = uint64(1) << (v % 64);
					// v already reached: its finished row arrived with it
					if (word & bit) continue;
					word |= bit;
					for (std::size_t w = 0; w < words; ++w) reach[u * words + w] |= reach[v * words + w];
				}
			}

			std::vector<uint32> closed_offset(n + 1, 0);
			for (std::size_t u = 0; u < n; ++u) {
				std::size_t bits = 0;
				for (std::size_t w = 0; w < words; ++w) bits += std::size_t(std::popcount(reach[u * words + w]));
				atl_assert(bits <= none - closed_offset[u]);
				closed_offset[u + 1] = closed_offset[u] + uint32(bits);
			}
			std::vector<uint32> closed(closed_offset[n]);
			for (std::size_t u = 0, k = 0; u < n; ++u)
				for (std::size_t w = 0; w < words; ++w)
					for (uint64 row = reach[u * words + w]; row; row &= row - 1)
						closed[k++] = uint32(w * 64) + uint32(std::countr_zero(row));
			return {std::move(closed_offset), std::move(closed)};
		}
	}

	/**
	 * @brief What @ref atl::graph::minimum_path_cover computes: vertex-disjoint paths through a
	 * DAG as successor/predecessor links (`none` at path ends).
	 */
	struct path_cover {
		std::size_t count = 0;	  ///< number of paths
		std::vector<uint32> next; ///< path successor per vertex, `none` at path ends
		std::vector<uint32> prev; ///< path predecessor per vertex, `none` at path starts

		/** @brief Reconstruct paths in increasing start-vertex order. @return Owning vertex lists, each start to end. @pre Links describe a path partition. @complexity O(n) time and heap storage; O(1) call stack. */
		[[nodiscard]] constexpr std::vector<std::vector<vertex_id>> paths() const {
			std::vector<std::vector<vertex_id>> out;
			out.reserve(count);
			for (uint32 v = 0; v < prev.size(); ++v) {
				if (prev[v] != none) continue;
				auto &path = out.emplace_back();
				for (uint32 at = v; at != none; at = next[at]) path.push_back(at);
			}
			return out;
		}
	};

	/**
	 * @brief Minimum path cover of a DAG — the fewest vertex-disjoint paths
	 *        (along edges) covering every vertex: split each vertex into
	 *        predecessor/successor copies, match, and each vertex's matched
	 *        right is its path successor; `n - matching` paths by König.
	 * @pre @p edges form a DAG (assert-checked in debug builds; on a cyclic
	 * input the matched successors close into cycles and no path cover
	 * exists).
	 * @tparam EdgeList Const forward edge range.
	 * @param n Number of vertices, less than `none`.
	 * @param edges Stable DAG edges, at most `none`, with endpoints in `[0, n)`.
	 * @return Owning links and the minimum number of paths, `n - matching.size`.
	 * @pre The size and endpoint bounds of bipartite_matching hold (asserted).
	 * @complexity O(n^2 (n + m)) conservative time, O(n + m) heap storage, O(1) call stack.
	 */
	template<edge_list EdgeList>
	[[nodiscard]] constexpr path_cover minimum_path_cover(std::size_t n, EdgeList const &edges) {
		atl_assert(detail::acyclic(n, edges));
		auto match = bipartite_matching(n, n, edges);
		return {n - match.size, std::move(match.left_pair), std::move(match.right_pair)};
	}

	/**
	 * @brief What @ref atl::graph::minimum_chain_cover computes: the chain partition plus the
	 * maximum antichain that certifies it.
	 */
	struct chain_cover {
		std::size_t count = 0;		  ///< number of chains
		std::vector<uint32> next;	  ///< chain successor per vertex (a reachable vertex), `none` at chain ends
		std::vector<uint32> prev;	  ///< chain predecessor, `none` at chain starts
		std::vector<uint8> antichain; ///< 0/1: in a maximum antichain (exactly `count` vertices)

		/** @brief Reconstruct chains in increasing start-vertex order. @return Owning vertex lists, each start to end. @pre Links describe a chain partition. @complexity O(n) time and heap storage; O(1) call stack. */
		[[nodiscard]] constexpr std::vector<std::vector<vertex_id>> chains() const {
			std::vector<std::vector<vertex_id>> out;
			out.reserve(count);
			for (uint32 v = 0; v < prev.size(); ++v) {
				if (prev[v] != none) continue;
				auto &chain = out.emplace_back();
				for (uint32 at = v; at != none; at = next[at]) chain.push_back(at);
			}
			return out;
		}
	};

	/**
	 * @brief Minimum chain cover and maximum antichain of a DAG's reachability
	 *        poset — Dilworth's theorem, constructively: the transitive
	 *        closure is computed internally (bitset rows in reverse
	 *        topological order), the split-graph matching on the closure walks
	 *        the chains, and the antichain is the vertices whose *both* split
	 *        copies lie in the @ref atl::graph::maximum_independent_set of the closure's
	 *        split graph (König). The two certify each other: any antichain is
	 *        at most any chain cover's size, and these are equal.
	 * @pre @p edges form a DAG (assert-checked in debug builds).
	 * @tparam EdgeList Const forward edge range.
	 * @param n Number of vertices, less than `none`.
	 * @param edges Stable DAG edges, at most `none`, with endpoints in `[0, n)`.
	 * @return Owning chain links and a maximum-antichain mask of `count` vertices.
	 * @pre The bitset allocation `n * ceil(n/64)` fits `vector<uint64>::max_size()`,
	 * and the closure has at most `none` edges (asserted).
	 * @complexity O(n + m + (n + m) ceil(n/64) + n^2 (n + q)) conservative time,
	 * O(n ceil(n/64) + m + q) heap storage, O(1) call stack; q is the closure edge count.
	 */
	template<edge_list EdgeList>
	[[nodiscard]] constexpr chain_cover minimum_chain_cover(std::size_t n, EdgeList const &edges) {
		// one closure CSR serves the matching and the cover; only the right
		// side's adjacency is derived (a transpose, already in left order)
		auto const [offset, to] = detail::closure_csr(n, edges);
		std::vector<uint32> in_offset(n + 1, 0), in_from(to.size());
		for (auto const v : to) ++in_offset[v + 1];
		for (std::size_t v = 0; v < n; ++v) in_offset[v + 1] += in_offset[v];
		{
			std::vector<uint32> cursor(in_offset.begin(), in_offset.end() - 1);
			for (uint32 u = 0; u < n; ++u)
				for (uint32 k = offset[u]; k < offset[u + 1]; ++k) in_from[cursor[to[k]]++] = u;
		}
		auto match = detail::bipartite_matching_csr(n, n, offset, to, in_offset, in_from);
		// the antichain: both split copies outside the cover (König)
		auto const cover = detail::vertex_cover_csr(n, n, offset, to, match);
		chain_cover r{n - match.size, std::move(match.left_pair), std::move(match.right_pair),
					  std::vector<uint8>(n, 0)};
		for (uint32 v = 0; v < n; ++v) r.antichain[v] = !cover.left[v] && !cover.right[v];
		return r;
	}

	/**
	 * @brief Membership of an edge across *all* maximum matchings — what
	 * @ref atl::graph::matching_edge_roles decides per input edge.
	 */
	enum class matching_edge : uint8 {
		forbidden, ///< in no maximum matching
		allowed,   ///< in some maximum matching, but not all
		mandatory, ///< in every maximum matching
	};

	/**
	 * @brief Classify every edge by its role across the maximum matchings —
	 *        the fine (SCC) part of the Dulmage–Mendelsohn decomposition.
	 *
	 * Over the flow-residual digraph of any one maximum matching (unmatched
	 * edges left->right, matched edges right->left, plus source/sink arcs
	 * encoding the free vertices), one rerouting criterion decides
	 * everything: an edge's unit can be moved iff its endpoints share a
	 * strongly connected component. Same SCC = `allowed`; different SCCs =
	 * `mandatory` if matched, `forbidden` if not. Parallel copies classify
	 * themselves (the matched copy's arc and an unmatched copy's arc form a
	 * 2-cycle, so no specific copy is ever mandatory).
	 *
	 * Pay-per-use like the König functions: pass the graph again plus a
	 * **maximum** matching for it.
	 * @pre @p matching is a maximum matching of these edges with pair arrays
	 *      of sizes @p left and @p right (sizes asserted)
	 * @tparam EdgeList Const forward edge range.
	 * @param left Number of left vertices.
	 * @param right Number of right vertices.
	 * @param edges Stable graph edges in output order, including distinct parallel copies.
	 * @param matching A valid maximum matching for this graph (validity documented only).
	 * @return One role per input edge, in input iteration order.
	 * @pre `left + right + 2 < none` and `m + left + right < none` (asserted),
	 * so residual vertices and edges fit CSR ids and size_t on 32-bit targets.
	 * Endpoints and pair-array sizes are asserted; iteration must remain stable.
	 * @complexity O(left + right + m) time and heap storage, O(1) call stack.
	 */
	template<edge_list EdgeList>
	[[nodiscard]] constexpr std::vector<matching_edge> matching_edge_roles(std::size_t left, std::size_t right,
																		   EdgeList const &edges,
																		   bipartite_matching_result const &matching) {
		atl_assert(matching.left_pair.size() == left && matching.right_pair.size() == right);
		auto const m = detail::checked_edge_count(edges, none);
		atl_assert(left < none && right < none && m <= none);
		atl_assert(uint64(left) + right + 2 < none);
		atl_assert(uint64(m) + left + right < none);
		uint32 const s = uint32(left + right), t = s + 1;
		std::vector<arc<>> arcs;
		arcs.reserve(m + left + right);
		std::vector<uint8> matched_copy(left, 0); // one copy per pair plays the matched role
		for (auto const &e : edges) {
			atl_assert(graph::source(e) < left && graph::target(e) < right);
			uint32 const l = graph::source(e), r = uint32(left) + graph::target(e);
			if (matching.left_pair[l] == graph::target(e) && !matched_copy[l]) {
				matched_copy[l] = 1;
				arcs.push_back({r, l});
			} else {
				arcs.push_back({l, r});
			}
		}
		for (uint32 l = 0; l < left; ++l)
			arcs.push_back(matching.left_pair[l] == none ? arc<>{s, l} : arc<>{l, s});
		for (uint32 r = 0; r < right; ++r)
			arcs.push_back(matching.right_pair[r] == none ? arc<>{uint32(left) + r, t}
														  : arc<>{t, uint32(left) + r});

		auto const components = scc(csr(left + right + 2, arcs));
		std::vector<matching_edge> kind(m);
		std::size_t i = 0;
		for (auto const &e : edges) {
			uint32 const l = graph::source(e), r = uint32(left) + graph::target(e);
			if (components.same(l, r))
				kind[i] = matching_edge::allowed;
			else if (matching.left_pair[l] == graph::target(e))
				kind[i] = matching_edge::mandatory;
			else
				kind[i] = matching_edge::forbidden;
			++i;
		}
		return kind;
	}
}
