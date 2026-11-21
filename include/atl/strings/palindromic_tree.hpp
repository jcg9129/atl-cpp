#pragma once
#include <atl/ds/detail/hashed_child_pool.hpp>
#include <atl/strings/detail/symbols.hpp>
#include <atl/strings/trie.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <ranges>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief The (generalized) palindromic tree (eertree) over the symbol
	 *        alphabet `[0, Alphabet)`: one node per distinct nonempty
	 *        palindromic substring of the inserted words, laid out for a large
	 *        sparse alphabet.
	 *
	 * Two structures share the nodes. The **build tree**: @ref atl::palindromic_tree::parent of `cPc`
	 * is `P`, with @ref atl::palindromic_tree::symbol the wrapping `c` — rooted at the two classic
	 * roots, @ref atl::palindromic_tree::odd_root (length -1, parent of the 1-char palindromes) and
	 * @ref atl::palindromic_tree::even_root (length 0, parent of the 2-char ones). The **suffix-link
	 * tree**: @ref atl::palindromic_tree::suffix_link points to the longest proper palindromic
	 * suffix. Node ids are creation order — the roots first, then each new
	 * palindrome as it first appears — so nonroot nodes satisfy both `parent(v) < v` and
	 * `suffix_link(v) < v`: iterating ids in reverse is already a valid
	 * propagation order for link-tree DPs.
	 *
	 * Since a palindrome has exactly one parent edge, the node fuses into the
	 * hashed child pool entry exactly like the trie's (symbol +
	 * sibling BST + payload, 40 B), and there are at most `L + 2` nodes on L
	 * input symbols — each position creates at most one new palindrome.
	 *
	 * The build is online and generalized like the suffix automaton's:
	 * @ref atl::palindromic_tree::extend appends a symbol to the word in progress and returns the
	 * node of its longest palindromic suffix; @ref atl::palindromic_tree::finish_word resets for the
	 * next word (palindromes never cross a word boundary, so only the walker
	 * and the per-word text buffer reset — nodes and counts persist), and
	 * @ref atl::palindromic_tree::insert is extend-each + finish_word. Queries are valid at any
	 * point. Duplicate words count with multiplicity in
	 * @ref atl::palindromic_tree::occurrence_counts.
	 *
	 * @tparam Alphabet Exclusive upper bound on symbols, at most 2^32 (and representable in size_t).
	 * @note Not constexpr (the child pool captures a runtime process hash seed).
	 * All traversals are iterative; no recursion per character or node.
	 * Moved-from instances may only be destroyed or assigned to. If a mutating
	 * allocation fails, clear or replace the tree prior to reuse.
	 * @pre At most 2^32 - 1 nodes, including roots; allocation is checked by the pool.
	 * @pre Each word has at most INT32_MAX symbols; occurrence counts fit uint64.
	 */
	template<std::size_t Alphabet>
		requires(Alphabet <= (uint64(1) << 32))
	class palindromic_tree {
		static constexpr uint32 nil = uint32(-1);

		/**
		 * @brief A fused pool entry: the pool manages `symbol`/`left`/`right` (the
		 * parent's sibling BST, keyed by the wrapping symbol); the rest is
		 * the palindrome's payload.
		 */
		struct node {
			uint64 ending_count = 0;
			uint32 symbol = 0;
			uint32 left = nil, right = nil;
			uint32 children = nil;
			uint32 parent = nil;
			uint32 link = nil;
			int32 length = 0;
			uint32 suffix_depth = 0;
		};

		detail::hashed_child_pool<node> pool_;
		std::vector<uint32> word_;
		uint32 last_ = 1;

		/**
		 * @brief Walk suffix links from @p v to the first palindrome that @p sym
		 * extends: `sym` must mirror the symbol just preceding the palindrome's
		 * occurrence ending at the current position. The odd root (length -1)
		 * always matches, so the walk terminates.
		 */
		[[nodiscard]] uint32 extensible(uint32 v, uint32 sym) const {
			auto const pos = uint32(word_.size()) - 1;
			for (;;) {
				int64 const mirror = int64(pos) - pool_[v].length - 1;
				if (mirror >= 0 && word_[uint32(mirror)] == sym) return v;
				v = pool_[v].link;
			}
		}

	public:
		/**
		 * @brief A handle to a node (@ref atl::palindromic_tree::odd_root, @ref atl::palindromic_tree::even_root, @ref atl::palindromic_tree::walk,
		 * @ref atl::palindromic_tree::extend). Stable until clear() or assignment; ids are creation
		 * order, so nonroot nodes satisfy `parent(v) < v` and `suffix_link(v) < v`.
		 */
		using node_id = uint32;

		/** @brief The empty tree: just the two roots.
		 * @note Not constexpr (uses the runtime-seeded tree).
		 */
		palindromic_tree() {
			pool_.add();
			pool_.add();
			pool_[0].length = -1;
			pool_[0].link = 0;
			pool_[1].length = 0;
			pool_[1].link = 0;
		}

		/** @brief The length -1 root: parent of the single-symbol palindromes (id 0).
		 * @note constexpr: reads no tree state.
		 */
		[[nodiscard]] constexpr node_id odd_root() const { return 0; }

		/**
		 * @brief The length 0 root: parent of the two-symbol palindromes (id 1).
		 * Every suffix-link chain ends here (or at @ref atl::palindromic_tree::odd_root).
		 * @note constexpr: reads no tree state.
		 */
		[[nodiscard]] constexpr node_id even_root() const { return 1; }

		/**
		 * @brief Append one symbol to the word in progress.
		 * @return The node of the longest palindromic suffix of the current
		 *         word, also the seed @ref atl::palindromic_tree::occurrence_counts propagates.
		 * @complexity Expected amortized O(log(degree + 1)); a single suffix-link walk can be linear.
		 * @tparam Sym A standard integral type.
		 * @param sym An integral symbol in [0, Alphabet) (asserted).
		 * @pre Current word length is below INT32_MAX prior to appending (asserted).
		 * @pre Occurrence seeds and aggregate counts fit uint64; seed increments are checked.
		 * @note Not constexpr (uses the runtime-seeded tree).
		 */
		template<std::integral Sym>
		node_id extend(Sym sym) {
			atl_assert(atl::detail::symbol_below(sym, Alphabet)); // in [0, Alphabet)
			uint32 const c = uint32(sym);
			atl_assert(word_.size() < std::size_t((std::numeric_limits<int32>::max)()));
			word_.push_back(c);
			uint32 const x = extensible(last_, c);
			// The emplace result avoids a second probe for missing children.
			auto const placed = pool_.emplace(pool_[x].children, c);
			pool_[x].children = placed.root;
			uint32 const y = placed.node;
			if (placed.was_inserted) {
				// the new palindrome cXc; its suffix link is cZc for the next
				// extensible palindromic suffix Z, which — being shorter than
				// the one new palindrome per position — already exists, and
				// lives in a strictly shorter node's children than y
				uint32 const slink =
					pool_[x].length == -1
						? even_root()
						: pool_.find(pool_[extensible(pool_[x].link, c)].children, c);
				pool_[y].parent = x;
				pool_[y].link = slink;
				pool_[y].length = pool_[x].length + 2;
				pool_[y].suffix_depth = pool_[slink].suffix_depth + 1;
			}
			last_ = y;
			atl_assert(pool_[last_].ending_count < (std::numeric_limits<uint64>::max)());
			++pool_[last_].ending_count;
			return last_;
		}

		/**
		 * @brief End the word in progress: drop its text buffer and reset the
		 *        walker so the next @ref atl::palindromic_tree::extend / @ref atl::palindromic_tree::insert starts a fresh
		 *        word. Nodes and occurrence seeds persist.
		 * @return The ended word's last node — the longest palindromic suffix
		 *         of the whole word; its suffix-link chain enumerates all of
		 *         the word's palindromic suffixes.
		 * @note Not constexpr (uses the runtime-seeded tree).
		 */
		node_id finish_word() {
			uint32 const ended = last_;
			last_ = even_root();
			word_.clear();
			return ended;
		}

		/**
		 * @brief The node of the longest palindromic suffix of the word in progress
		 * (@ref atl::palindromic_tree::even_root if none).
		 * @note Not constexpr (uses the runtime-seeded tree).
		 */
		[[nodiscard]] node_id last() const { return last_; }

		/**
		 * @brief Add one whole word — @ref atl::palindromic_tree::extend each projected symbol, then
		 *        @ref atl::palindromic_tree::finish_word. Repeated calls build the generalized tree
		 *        of the collection.
		 * @tparam R An input range with integral projected symbols.
		 * @tparam Proj A projection invocable on range references.
		 * @param word Symbols appended to the word in progress.
		 * @param proj Projection into [0, Alphabet).
		 * @complexity Expected amortized O(word size * log(maximum degree + 1)).
		 * @return The word's last node (see @ref atl::palindromic_tree::finish_word).
		 * @note Not constexpr (uses the runtime-seeded tree).
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		node_id insert(R &&word, Proj proj = {}) {
			for (auto &&e : atl::detail::input_elements(word)) extend(std::invoke(proj, std::forward<decltype(e)>(e)));
			return finish_word();
		}

		/** @brief Palindrome length; -1 at @ref atl::palindromic_tree::odd_root, 0 at @ref atl::palindromic_tree::even_root.
		 * @note Not constexpr (uses the runtime-seeded tree).
		 * @param v Valid node id (asserted); root restrictions are stated above.
		 * @complexity O(1).
		 */
		[[nodiscard]] int32 length(node_id v) const {
			atl_assert(v < pool_.size());
			return pool_[v].length;
		}

		/**
		 * @brief The longest proper palindromic suffix (@ref atl::palindromic_tree::even_root when only the
		 * empty one exists). @pre `v` is not @ref atl::palindromic_tree::odd_root
		 * @note Not constexpr (uses the runtime-seeded tree).
		 * @param v Valid node id (asserted); root restrictions are stated above.
		 * @complexity O(1).
		 */
		[[nodiscard]] node_id suffix_link(node_id v) const {
			atl_assert(v != 0 && v < pool_.size());
			return pool_[v].link;
		}

		/**
		 * @brief The palindrome with the first and last symbols removed — the build
		 * tree's edge. @pre `v` is not a root
		 * @note Not constexpr (uses the runtime-seeded tree).
		 * @param v Valid node id (asserted); root restrictions are stated above.
		 * @complexity O(1).
		 */
		[[nodiscard]] node_id parent(node_id v) const {
			atl_assert(v >= 2 && v < pool_.size());
			return pool_[v].parent;
		}

		/** @brief The wrapping symbol on the @ref atl::palindromic_tree::parent edge. @pre `v` is not a root
		 * @note Not constexpr (uses the runtime-seeded tree).
		 * @param v Valid node id (asserted); root restrictions are stated above.
		 * @complexity O(1).
		 */
		[[nodiscard]] uint32 symbol(node_id v) const {
			atl_assert(v >= 2 && v < pool_.size());
			return pool_[v].symbol;
		}

		/**
		 * @brief Number of nonempty palindromic suffixes of @p v's palindrome — its
		 * depth in the suffix-link tree. 0 at the roots.
		 * @note Not constexpr (uses the runtime-seeded tree).
		 * @param v Valid node id (asserted); root restrictions are stated above.
		 * @complexity O(1).
		 */
		[[nodiscard]] uint32 palindromic_suffix_count(node_id v) const {
			atl_assert(v < pool_.size());
			return pool_[v].suffix_depth;
		}

		/**
		 * @brief The palindrome `sym·v·sym`, or `std::nullopt`. Descending from a
		 * root spells a palindrome center-outward: `walk(odd_root(), c)` is
		 * `c`, `walk(even_root(), c)` is `cc`.
		 * @note Not constexpr (uses the runtime-seeded tree).
		 * @tparam Sym A standard integral type.
		 * @param v Valid build-tree node (asserted).
		 * @param sym Wrapping symbol; out-of-alphabet values return nullopt.
		 * @complexity Expected O(log(degree + 1)), worst-case O(degree).
		 */
		template<std::integral Sym>
		[[nodiscard]] std::optional<node_id> walk(node_id v, Sym sym) const {
			atl_assert(v < pool_.size());
			if (!atl::detail::symbol_below(sym, Alphabet)) return std::nullopt;
			uint32 const child = pool_.find(pool_[v].children, uint32(sym));
			return child == nil ? std::nullopt : std::optional<node_id>(child);
		}

		/**
		 * @brief Visit `(uint32 symbol, node_id child)` for every build-tree
		 *        child `symbol·v·symbol` of @p v, in unspecified (hashed) order.
		 * @complexity O(children of @p v); allocates a traversal stack per call
		 * @note Not constexpr (uses the runtime-seeded tree).
		 * @tparam Visit Invocable with (uint32 symbol, node_id child).
		 * @param v Valid build-tree node (asserted).
		 * @param visit Callback, invoked through std::invoke.
		 * @pre The callback does not mutate the tree while traversing.
		 */
		template<child_visitor Visit>
		void for_each_child(node_id v, Visit visit) const {
			atl_assert(v < pool_.size());
			pool_.for_each(pool_[v].children, [&](uint32 c) { std::invoke(visit, pool_[c].symbol, node_id(c)); });
		}

		/**
		 * @brief Per-node series links, the tool of palindromic factorization
		 *        DPs: for a nonroot node `v` with `diff(v) = length(v) -
		 *        length(suffix_link(v))`, entry `v` is the longest palindromic
		 *        suffix `u` on `v`'s suffix-link chain with `diff(u) !=
		 *        diff(v)` (a root when the chain runs out first). The chain
		 *        from `v` down to its series link is one arithmetic series of
		 *        lengths, so hopping series links visits O(log length(v))
		 *        series. The root entries map to themselves.
		 * @complexity O(node_count()), one forward pass over creation order
		 *             (`suffix_link(v) < v`)
		 * @note Not constexpr (uses the runtime-seeded tree).
		 */
		[[nodiscard]] std::vector<node_id> series_links() const {
			std::vector<node_id> series(pool_.size());
			series[0] = 0;
			series[1] = 1;
			for (uint32 v = 2; v < pool_.size(); ++v) {
				uint32 const u = pool_[v].link;
				if (u < 2) {
					series[v] = u;
					continue;
				}
				int32 const diff_v = pool_[v].length - pool_[u].length;
				int32 const diff_u = pool_[u].length - pool_[pool_[u].link].length;
				series[v] = diff_v == diff_u ? series[u] : u;
			}
			return series;
		}

		/**
		 * @brief Per-node occurrence counts: how many times each palindrome
		 *        occurs across all inserted words (per word instance). Seeds
		 *        recorded by @ref atl::palindromic_tree::extend, propagated up the suffix links in
		 *        one reverse-id pass. The root entries are the propagation
		 *        remainders (every position reaches @ref atl::palindromic_tree::even_root) — ignore
		 *        them.
		 * @complexity O(node_count())
		 * @note Not constexpr (uses the runtime-seeded tree).
		 */
		[[nodiscard]] std::vector<uint64> occurrence_counts() const {
			std::vector<uint64> occ(pool_.size());
			for (uint32 v = 0; v < pool_.size(); ++v) occ[v] = pool_[v].ending_count;
			for (uint32 v = pool_.size(); v-- > 2;) {
				uint32 const target = pool_[v].link;
				atl_assert(occ[v] <= (std::numeric_limits<uint64>::max)() - occ[target]);
				occ[target] += occ[v];
			}
			return occ;
		}

		/**
		 * @brief Number of distinct nonempty palindromic substrings over all
		 * inserted words (= `node_count() - 2`).
		 * @note Not constexpr (uses the runtime-seeded tree).
		 */
		std::size_t palindrome_count() const { return pool_.size() - 2; }

		/** @brief Number of nodes, the two roots included.
		 * @note Not constexpr (uses the runtime-seeded tree).
		 */
		std::size_t node_count() const { return pool_.size(); }

		/**
		 * @brief Pre-allocate for @p symbols total input symbols (at most
		 * `symbols + 2` nodes).
		 * @param symbols Total input-symbol capacity.
		 * @pre symbols <= SIZE_MAX - 2 (asserted).
		 * @note Not constexpr (uses the runtime-seeded tree).
		 */
		void reserve(std::size_t symbols) {
			atl_assert(symbols <= (std::numeric_limits<std::size_t>::max)() - 2);
			pool_.reserve(symbols + 2);
		}

		/** @brief Back to the empty tree (memory kept).
		 * @note Not constexpr (uses the runtime-seeded tree).
		 */
		void clear() {
			pool_.clear();
			word_.clear();
			pool_.add();
			pool_.add();
			pool_[0].length = -1;
			pool_[0].link = 0;
			pool_[1].length = 0;
			pool_[1].link = 0;
			last_ = 1;
		}
	};
}
