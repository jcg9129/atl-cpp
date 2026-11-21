#pragma once
#include <atl/strings/comparable_text.hpp>
#include <atl/strings/detail/symbols.hpp>
#include <atl/strings/trie.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief The frozen counterpart of @ref atl::trie: built once from a full word
	 *        list, then queried — no insert/erase. Children live as contiguous
	 *        runs of `(symbol, target)` pairs **sorted by symbol** in one flat
	 *        array, so lookups are cache-local binary searches (linear at small
	 *        degree), memory is ~8 B/edge + ~20 B/node + 4 B/word with zero per-node
	 *        containers, and — unlike the dynamic trie's hashed sibling order —
	 *        lexicographic traversal works: @ref atl::static_trie::children yields a node's edges
	 *        in symbol order, and node ids are **lexicographic preorder**.
	 *
	 * The build never sorts whole words. It is an MSD partition: each node
	 * sorts only its surviving words' *current* symbols, gathered as plain
	 * integer keys (no word comparisons, no LCP scans); a slice whose words
	 * all carry one symbol skips the sort, and a lone word emits its
	 * remaining suffix as a chain. Total work is O(n + L + sum over branching
	 * nodes of slice·log slice), at most O(n + L log(1 + n)), over L input symbols;
	 * peak scratch is O(L + n), including materialized symbols, one word-id
	 * permutation, one key array and a heap stack. Empty words add O(n) work.
	 *
	 * Query semantics match @ref atl::trie exactly (multiset of words: @ref atl::static_trie::count is
	 * a word's multiplicity, @ref atl::static_trie::count_with_prefix the occurrences carrying a
	 * prefix; the same node-handle API). A word list that still changes between
	 * queries needs the dynamic @ref atl::trie instead.
	 *
	 * @tparam Alphabet Exclusive upper bound on symbols, at most 2^32 and representable by size_t.
	 * @pre Total symbols are at most 2^32 - 2; word count fits uint32; all
	 *      projected symbols lie in `[0, Alphabet)` (asserted during materialization).
	 * @note Signed characters are numeric symbols; project to unsigned char for bytes.
	 *       Input ranges may be non-contiguous and single-pass. Queries outside the
	 *       alphabet miss. Build and lookup use loops and heap stacks, without
	 *       recursion per node or character. Allocation/projection exceptions propagate.
	 *       Returned spans survive const operations; assignment and destruction invalidate
	 *       them, and moving transfers their storage. A moved-from trie may only be
	 *       assigned or destroyed.
	 */
	template<std::size_t Alphabet>
		requires(Alphabet <= (uint64(1) << 32))
	class static_trie {
		static constexpr uint32 nil = uint32(-1);

	public:
		/**
		 * @brief One child edge: the symbol and the node it leads to. @ref atl::static_trie::children
		 * spans are sorted ascending by `symbol`.
		 */
		struct edge {
			uint32 symbol; ///< Numeric edge label.
			uint32 target; ///< Child node id.
		};

		/**
		 * @brief A handle to a node (@ref atl::static_trie::root, @ref atl::static_trie::walk, @ref atl::static_trie::children). Ids are
		 * dense `[0, node_count())` in lexicographic preorder: the root is 0
		 * and every subtree occupies the contiguous id interval
		 * `[n, subtree_end(n))`.
		 */
		using node_id = uint32;

	private:
		std::vector<uint32> offset_;	   // node -> first edge; node_count()+1 entries
		std::vector<edge> edges_;		   // all child edges, per-node runs sorted by symbol
		std::vector<uint32> word_count_;   // occurrences ending exactly at the node
		std::vector<uint32> prefix_count_; // occurrences whose word passes through the node
		std::vector<uint32> first_word_;   // first word (in lexicographic order) in the subtree
		std::vector<uint32> subtree_end_;  // one past the last preorder id in the subtree
		std::vector<uint32> word_order_;   // input index of each lexicographic word position

		/**
		 * @brief Child of @p node along @p sym, or @ref atl::static_trie::nil. Runs are usually tiny:
		 * scan short ones linearly, binary-search the rest (a fixed threshold
		 * of eight edges).
		 */
		[[nodiscard]] constexpr uint32 step(uint32 node, uint32 sym) const {
			if (offset_[node] == offset_[node + 1]) return nil;
			edge const *lo = edges_.data() + offset_[node];
			edge const *hi = edges_.data() + offset_[node + 1];
			if (hi - lo <= 8) {
				for (; lo != hi; ++lo)
					if (lo->symbol == sym) return lo->target;
				return nil;
			}
			while (lo < hi) {
				edge const *mid = lo + (hi - lo) / 2;
				if (mid->symbol < sym)
					lo = mid + 1;
				else
					hi = mid;
			}
			return lo != edges_.data() + offset_[node + 1] && lo->symbol == sym ? lo->target : nil;
		}

		template<typename R, typename Proj>
			requires trie_word<R, Proj>
		[[nodiscard]] constexpr uint32 node_of(R &&w, Proj &proj) const {
			uint32 cur = 0;
			for (auto &&e : atl::detail::input_elements(w)) {
				auto const s = std::invoke(proj, std::forward<decltype(e)>(e));
				if (!atl::detail::symbol_below(s, Alphabet)) return nil;
				cur = step(cur, uint32(s));
				if (cur == nil) return nil;
			}
			return cur;
		}

	public:
		/** @brief The empty dictionary: just a root, every count 0. */
		constexpr static_trie()
			: offset_{0, 0}, word_count_{0}, prefix_count_{0}, first_word_{0}, subtree_end_{1} {}

		/**
		 * @brief Build from @p words, each element a word whose symbols are seen
		 *        through @p proj (as in @ref atl::trie::insert). Duplicate words are
		 *        counted with multiplicity.
		 * @complexity O(n + L + sum of slice·log slice over branching nodes), at most O(n + L log(1 + n)), over L total symbols
		 * @tparam R An input range of words.
		 * @tparam Proj Lvalue-invocable projection returning an integral symbol.
		 * @param words The projected pattern collection.
		 * @param proj Symbol projection.
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word_range<R, Proj>
		constexpr explicit static_trie(R &&words, Proj proj = {}) {
			// materialize the symbol sequences once: the level passes below
			// re-read each word once per depth, which arbitrary input ranges
			// (and per-symbol projections) should not pay repeatedly
			std::vector<uint32> sym;
			std::vector<uint32> word_end{0};
			if constexpr (std::ranges::sized_range<R>) {
				auto const count = std::ranges::size(words);
				atl_assert(count <= nil && count < word_end.max_size());
				word_end.reserve(std::size_t(count) + 1);
			}
			for (auto &&w : atl::detail::input_elements(words)) {
				for (auto &&e : atl::detail::input_elements(w)) {
					auto const s = std::invoke(proj, std::forward<decltype(e)>(e));
					atl_assert(atl::detail::symbol_below(s, Alphabet));
					atl_assert(sym.size() < uint64(nil) - 1);
					sym.push_back(uint32(s));
				}
				atl_assert(word_end.size() <= nil && word_end.size() < word_end.max_size());
				word_end.push_back(uint32(sym.size()));
			}
			uint32 const n = uint32(word_end.size() - 1);

			// DFS radix partition. A frame is a slice of `ids` = the words whose
			// prefix reaches this node; the node's number is assigned at pop
			// (preorder), so the edge that led here is patched then. The
			// finished `ids` is the lexicographic word order, kept as
			// word_order_.
			std::vector<uint32> &ids = word_order_;
			ids.resize(n);
			for (uint32 i = 0; i < n; ++i) ids[i] = i;

			struct frame {
				uint32 lo, hi, depth;
				uint32 parent_edge; // index into edges_, nil at the root
			};

			std::vector<frame> stack{{0, n, 0, nil}};
			auto const symbol_at = [&](uint32 w, uint32 d) { return sym[word_end[w] + d]; };
			auto const length = [&](uint32 w) { return word_end[w + 1] - word_end[w]; };
			// (symbol << 32 | word) keys of a slice's survivors: sorting plain
			// integers, gathered once, beats a comparator that reloads each
			// symbol through two indirections
			std::vector<uint64> keys;
			auto const open_node = [&](uint32 lo, uint32 hi, uint32 parent_edge) {
				uint32 const id = uint32(word_count_.size());
				if (parent_edge != nil) edges_[parent_edge].target = id;
				offset_.push_back(uint32(edges_.size()));
				prefix_count_.push_back(hi - lo);
				first_word_.push_back(lo); // the node's slice [lo, hi) IS its word interval
				return id;
			};
			while (!stack.empty()) {
				auto const [lo, hi, depth, parent_edge] = stack.back();
				stack.pop_back();
				uint32 id = open_node(lo, hi, parent_edge);
				if (hi - lo == 1) {
					uint32 const w = ids[lo];
					for (uint32 d = depth; d < length(w); ++d) {
						word_count_.push_back(0);
						edges_.push_back({symbol_at(w, d), id + 1});
						id = open_node(lo, hi, nil);
					}
					word_count_.push_back(1);
					continue;
				}
				auto const rest = std::ranges::partition(ids.begin() + lo, ids.begin() + hi,
														 [&](uint32 w) { return length(w) == depth; });
				uint32 const live = uint32(rest.begin() - ids.begin());
				word_count_.push_back(live - lo);
				if (live == hi) continue;
				keys.clear();
				bool is_one_symbol = true;
				uint32 const s0 = symbol_at(ids[live], depth);
				for (uint32 i = live; i < hi; ++i) {
					uint32 const s = symbol_at(ids[i], depth);
					is_one_symbol = is_one_symbol && s == s0;
					keys.push_back(uint64(s) << 32 | ids[i]);
				}
				if (is_one_symbol) {
					edges_.push_back({s0, nil});
					// The slice's common continuation is a chain of single-child
					// nodes. Measuring it word by word reads each word's symbols
					// sequentially, where one level per pass would jump between
					// words at every depth.
					uint32 const first = ids[live];
					uint32 chain = length(first) - (depth + 1);
					for (uint32 i = live + 1; i < hi && chain != 0; ++i) {
						uint32 const w = ids[i];
						uint32 const limit = (std::min)(chain, length(w) - (depth + 1));
						uint32 k = 0;
						while (k < limit && symbol_at(w, depth + 1 + k) == symbol_at(first, depth + 1 + k)) ++k;
						chain = k;
					}
					for (uint32 d = depth + 1; d < depth + 1 + chain; ++d) {
						open_node(live, hi, uint32(edges_.size() - 1));
						word_count_.push_back(0);
						edges_.push_back({symbol_at(first, d), nil});
					}
					stack.push_back({live, hi, depth + 1 + chain, uint32(edges_.size() - 1)});
					continue;
				}
				std::ranges::sort(keys);
				for (uint32 i = live; i < hi; ++i) ids[i] = uint32(keys[i - live]);
				uint32 const first_edge = uint32(edges_.size());
				for (uint32 a = 0; a < hi - live;) {
					uint32 const s = uint32(keys[a] >> 32);
					uint32 b = a + 1;
					while (b < hi - live && uint32(keys[b] >> 32) == s) ++b;
					edges_.push_back({s, nil}); // target patched when the child pops
					a = b;
				}
				// push runs in reverse so the smallest symbol pops first: node
				// ids come out in lexicographic preorder
				uint32 run_hi = hi;
				for (uint32 e = uint32(edges_.size()); e-- > first_edge;) {
					uint32 run_lo = run_hi;
					while (run_lo > live && uint32(keys[run_lo - 1 - live] >> 32) == edges_[e].symbol) --run_lo;
					stack.push_back({run_lo, run_hi, depth + 1, e});
					run_hi = run_lo;
				}
			}
			offset_.push_back(uint32(edges_.size()));
			// a node's subtree ends where its last (largest-symbol, so
			// largest-id) child's does; children have larger ids, so one
			// reverse preorder pass settles every node
			uint32 const nodes = uint32(word_count_.size());
			subtree_end_.resize(nodes);
			for (uint32 v = nodes; v-- > 0;)
				subtree_end_[v] = offset_[v] == offset_[v + 1] ? v + 1 : subtree_end_[edges_[offset_[v + 1] - 1].target];
		}

		/** @brief Multiplicity of the exact word @p word. @complexity O(len·log degree)
		 * @tparam R An input range of symbols.
		 * @tparam Proj Lvalue-invocable projection returning an integral symbol.
		 * @param word The projected symbol range.
		 * @param proj Symbol projection.
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		[[nodiscard]] constexpr std::size_t count(R &&word, Proj proj = {}) const {
			uint32 const v = node_of(word, proj);
			return v == nil ? 0 : word_count_[v];
		}

		/** @brief Whether @p word occurs at least once.
		 * @tparam R An input range of symbols.
		 * @tparam Proj Lvalue-invocable projection returning an integral symbol.
		 * @param word The projected symbol range.
		 * @param proj Symbol projection.
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		[[nodiscard]] constexpr bool contains(R &&word, Proj proj = {}) const {
			uint32 const v = node_of(word, proj);
			return v != nil && word_count_[v] != 0;
		}

		/**
		 * @brief The node spelling @p word, or `std::nullopt` if its path is absent.
		 *        The empty word always returns root(), even for an empty dictionary.
		 * @complexity O(len·log degree)
		 * @tparam R An input range of symbols.
		 * @tparam Proj Lvalue-invocable projection returning an integral symbol.
		 * @param word The projected symbol range.
		 * @param proj Symbol projection.
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		[[nodiscard]] constexpr std::optional<node_id> find(R &&word, Proj proj = {}) const {
			uint32 const v = node_of(word, proj);
			return v == nil ? std::nullopt : std::optional<node_id>(v);
		}

		/**
		 * @brief Number of stored occurrences whose word begins with @p prefix.
		 * @complexity O(len·log degree)
		 * @tparam R An input range of symbols.
		 * @tparam Proj Lvalue-invocable projection returning an integral symbol.
		 * @param prefix The projected symbol range.
		 * @param proj Symbol projection.
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		[[nodiscard]] constexpr std::size_t count_with_prefix(R &&prefix, Proj proj = {}) const {
			uint32 const v = node_of(prefix, proj);
			return v == nil ? 0 : prefix_count_[v];
		}

		/** @brief The root handle — the empty prefix (id 0). */
		[[nodiscard]] constexpr node_id root() const { return 0; }

		/** @brief The child of @p n along @p sym, or `std::nullopt`.
		 * @tparam Sym Integral symbol type.
		 * @param sym Numeric edge label; out-of-alphabet values miss or reset.
		 * @pre The node handle belongs to this object (asserted).
		 */
		template<std::integral Sym>
		[[nodiscard]] constexpr std::optional<node_id> walk(node_id n, Sym sym) const {
			atl_assert(n < word_count_.size());
			if (!atl::detail::symbol_below(sym, Alphabet)) return std::nullopt;
			uint32 const c = step(n, uint32(sym));
			return c == nil ? std::nullopt : std::optional<node_id>(c);
		}

		/**
		 * @brief @p n's child edges, **sorted ascending by symbol** — the ordered
		 * traversal the dynamic trie cannot offer (k-th word, completions in
		 * lexicographic order, merges over two tries).
		 * @pre The node handle belongs to this object (asserted).
		 */
		[[nodiscard]] constexpr std::span<edge const> children(node_id n) const {
			atl_assert(n < word_count_.size());
			return std::span<edge const>(edges_).subspan(offset_[n], offset_[n + 1] - offset_[n]);
		}

		/** @brief Occurrences of the word ending exactly at @p n.
		 * @pre The node handle belongs to this object (asserted).
		 */
		[[nodiscard]] constexpr std::size_t word_count(node_id n) const {
			atl_assert(n < word_count_.size());
			return word_count_[n];
		}

		/**
		 * @brief The words carrying @p n's prefix, as the half-open interval
		 *        `[begin, end)` of positions in the LEXICOGRAPHIC word order
		 *        (the multiset order, duplicates counted; `end - begin ==`
		 *        @ref atl::static_trie::prefix_count). @ref atl::static_trie::word_order maps a position back to
		 *        its input index. Walking a query prefix with @ref atl::static_trie::walk and
		 *        reading its interval answers "which dictionary words start
		 *        with x" in O(len·log degree) total.
		 * @pre The node handle belongs to this object (asserted).
		 */
		[[nodiscard]] constexpr index_interval word_interval(node_id n) const {
			atl_assert(n < first_word_.size());
			return {first_word_[n], first_word_[n] + prefix_count_[n]};
		}

		/**
		 * @brief The input indices of the words in lexicographic order: entry
		 *        `i` is the input index of the word at position `i` of every
		 *        @ref atl::static_trie::word_interval. Equal words come in unspecified order
		 *        (the build's partition is not stable).
		 */
		[[nodiscard]] constexpr std::span<uint32 const> word_order() const { return word_order_; }

		/**
		 * @brief One past the last node id in @p n's subtree: the subtree is
		 *        exactly the preorder ids `[n, subtree_end(n))`, the interval
		 *        Euler-tour style range structures over node ids index.
		 * @pre The node handle belongs to this object (asserted).
		 */
		[[nodiscard]] constexpr node_id subtree_end(node_id n) const {
			atl_assert(n < subtree_end_.size());
			return subtree_end_[n];
		}

		/** @brief Occurrences whose word passes through @p n (i.e. carries its prefix).
		 * @pre The node handle belongs to this object (asserted).
		 */
		[[nodiscard]] constexpr std::size_t prefix_count(node_id n) const {
			atl_assert(n < prefix_count_.size());
			return prefix_count_[n];
		}

		/** @brief Number of stored words counted with multiplicity. */
		constexpr std::size_t size() const { return prefix_count_[0]; }

		/** @brief Whether no input word was supplied. */
		constexpr bool empty() const { return size() == 0; }

		/** @brief Number of trie nodes, root included. */
		[[nodiscard]] constexpr std::size_t node_count() const { return word_count_.size(); }

		/** @brief Number of edges (= node_count() - 1 on a nonempty dictionary). */
		[[nodiscard]] constexpr std::size_t edge_count() const { return edges_.size(); }
	};
}
