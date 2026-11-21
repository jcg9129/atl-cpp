#pragma once
#include <atl/ds/detail/hashed_child_pool.hpp>
#include <atl/strings/detail/symbols.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	namespace detail {
		/** @brief The integral projection result is checked against the alphabet without narrowing. */
		template<typename R, typename Proj>
		using trie_symbol_t = std::remove_cvref_t<std::invoke_result_t<Proj &, std::ranges::range_reference_t<R>>>;
	}

	/**
	 * @brief A range is a `trie_word` under projection `Proj` when it is an input range
	 * whose projected symbol type is integral.
	 * @tparam R Candidate input range.
	 * @tparam Proj Lvalue-invocable symbol projection.
	 */
	template<typename R, typename Proj = std::identity>
	concept trie_word = std::ranges::input_range<R>
					 && std::invocable<Proj &, std::ranges::range_reference_t<R>>
					 && std::integral<detail::trie_symbol_t<R, Proj>>;

	/**
	 * @brief A range of words buildable into a @ref atl::static_trie or an
	 * @ref atl::aho_corasick automata: an input range whose elements satisfy @ref atl::trie_word under `Proj`.
	 * @tparam R Candidate range of words.
	 * @tparam Proj Lvalue-invocable symbol projection.
	 */
	template<typename R, typename Proj = std::identity>
	concept trie_word_range =
		std::ranges::input_range<R> && trie_word<std::ranges::range_reference_t<R>, Proj>;

	/**
	 * @brief A callback for a node's `for_each_child` (@ref atl::trie,
	 *        @ref atl::palindromic_tree): invoked as `visit(uint32 symbol,
	 *        uint32 child)` once per child edge.
	 * @tparam Visit Callback invocable as an lvalue on the symbol and child id.
	 */
	template<typename Visit>
	concept child_visitor = std::invocable<Visit &, uint32, uint32>;

	/**
	 * @brief A multiset of integral-symbol words with sparse, hashed sibling trees.
	 * @tparam Alphabet Exclusive symbol bound, at most 2^32 (and representable by size_t).
	 * @pre Inserted symbols lie in `[0, Alphabet)`; stored occurrences fit uint32;
	 *      fewer than 2^32 - 1 nodes exist when a new node is added (asserted).
	 * @note Queries treat out-of-alphabet symbols as absent. Signed characters are
	 *       numeric symbols: project to unsigned char to index arbitrary bytes.
	 *       Words may be non-contiguous input ranges, each read once, erase included
	 *       (it keeps the visited path in O(word length) scratch).
	 *       Operations use loops and heap scratch, with no recursion per node or symbol.
	 *       Hash order gives expected O(log(1 + degree)) child lookups, with linear
	 *       worst-case sibling height; it is not an adversarial worst-case guarantee.
	 * @note Not constexpr (captures the process hash seed in hashed_child_pool).
	 *       Construct from main or later. Allocation/projection exceptions propagate;
	 *       mutating operations do not promise transactional recovery.
	 */
	template<std::size_t Alphabet>
		requires(Alphabet <= (uint64(1) << 32))
	class trie {
		static constexpr uint32 nil = uint32(-1);

		struct node {
			uint32 symbol = 0;
			uint32 left = nil, right = nil;
			uint32 children = nil;
			uint32 word_count = 0;
			uint32 prefix_count = 0;
		};

		detail::hashed_child_pool<node> pool_;
		std::vector<uint32> path_; // erase's scratch: the nodes of the erased word

		uint32 link(uint32 parent, uint32 sym) {
			auto const placed = pool_.emplace(pool_[parent].children, sym);
			pool_[parent].children = placed.root;
			return placed.node;
		}

		[[nodiscard]] uint32 step(uint32 parent, uint32 sym) const {
			return pool_.find(pool_[parent].children, sym);
		}

		template<typename R, typename Proj>
			requires trie_word<R, Proj>
		[[nodiscard]] uint32 node_of(R &&w, Proj &proj) const {
			uint32 cur = 0;
			for (auto &&e : atl::detail::input_elements(w)) {
				auto const sym = std::invoke(proj, std::forward<decltype(e)>(e));
				// Compare at full width so 2^32 + s cannot alias s.
				if (!atl::detail::symbol_below(sym, Alphabet)) return nil;
				cur = step(cur, uint32(sym));
				if (cur == nil) return nil;
			}
			return cur;
		}

	public:
		/**
		 * @brief A handle to a node: @ref atl::trie::root, then @ref atl::trie::walk. Stable through insert, erase and reserve; clear and assignment invalidate handles.
		 */
		using node_id = uint32;

		/** @brief Construct an empty multiset with its root node.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		trie() { pool_.add(); }

		/** @brief Reserve for @p total_symbols edges plus the root.
		 * @pre The node capacity fits uint32 and size_t (asserted).
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		void reserve(std::size_t total_symbols) {
			atl_assert(total_symbols < nil && total_symbols < (std::numeric_limits<std::size_t>::max)());
			pool_.reserve(total_symbols + 1);
		}

		/** @brief Drop all nodes and words, invalidating handles; retain the hash seed and capacity.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		void clear() {
			pool_.clear();
			pool_.add();
		}

		/**
		 * @brief Insert one occurrence of @p word, each element seen through
		 *        @p proj.
		 * @return The node spelling @p word (stable, see @ref atl::trie::node_id).
		 * @complexity O(len · expected BST depth)
		 * @note Not constexpr (runtime node storage and hashing).
		 * @tparam R An input range of symbols.
		 * @tparam Proj Lvalue-invocable projection returning an integral symbol.
		 * @param word The projected symbol range.
		 * @param proj Symbol projection.
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		node_id insert(R &&word, Proj proj = {}) {
			atl_assert(pool_[0].prefix_count < (std::numeric_limits<uint32>::max)());
			++pool_[0].prefix_count;
			uint32 cur = 0;
			for (auto &&e : atl::detail::input_elements(word)) {
				auto const s = std::invoke(proj, std::forward<decltype(e)>(e));
				atl_assert(atl::detail::symbol_below(s, Alphabet));
				cur = link(cur, uint32(s));
				++pool_[cur].prefix_count;
			}
			++pool_[cur].word_count;
			return cur;
		}

		/**
		 * @brief Remove one occurrence of @p word; returns whether it was present. Nodes
		 * are not reclaimed — only counts drop — so handles stay valid. The word is read
		 * once, so an input-only range is accepted; the visited path is held in
		 * O(len) scratch.
		 * @complexity O(len · expected BST depth)
		 * @note Not constexpr (runtime node storage and hashing).
		 * @tparam R An input range of symbols.
		 * @tparam Proj Lvalue-invocable projection returning an integral symbol.
		 * @param word The projected symbol range.
		 * @param proj Symbol projection.
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		bool erase(R &&word, Proj proj = {}) {
			// Defer count changes until the terminal occurrence is known to exist.
			path_.clear();
			uint32 cur = 0;
			for (auto &&e : atl::detail::input_elements(word)) {
				auto const sym = std::invoke(proj, std::forward<decltype(e)>(e));
				if (!atl::detail::symbol_below(sym, Alphabet)) return false; // Preserve the full symbol width.
				cur = step(cur, uint32(sym));
				if (cur == nil) return false;
				path_.push_back(cur);
			}
			if (pool_[cur].word_count == 0) return false;
			--pool_[cur].word_count;
			--pool_[0].prefix_count;
			for (uint32 const v : path_) --pool_[v].prefix_count;
			return true;
		}

		/**
		 * @brief Multiplicity of the exact word @p word.
		 * @complexity O(len · expected BST depth)
		 * @note Not constexpr (runtime node storage and hashing).
		 * @tparam R An input range of symbols.
		 * @tparam Proj Lvalue-invocable projection returning an integral symbol.
		 * @param word The projected symbol range.
		 * @param proj Symbol projection.
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		[[nodiscard]] std::size_t count(R &&word, Proj proj = {}) const {
			uint32 const n = node_of(word, proj);
			return n == nil ? 0 : pool_[n].word_count;
		}

		/** @brief Whether @p word is stored at least once.
		 * @note Not constexpr (runtime node storage and hashing).
		 * @tparam R An input range of symbols.
		 * @tparam Proj Lvalue-invocable projection returning an integral symbol.
		 * @param word The projected symbol range.
		 * @param proj Symbol projection.
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		[[nodiscard]] bool contains(R &&word, Proj proj = {}) const {
			uint32 const n = node_of(word, proj);
			return n != nil && pool_[n].word_count != 0;
		}

		/**
		 * @brief The node spelling @p word, or `std::nullopt` if its path was never inserted since clear.
		 *        Allocated paths remain findable even when their counts reach zero;
		 *        the empty word always returns root().
		 * @complexity O(len · expected BST depth)
		 * @note Not constexpr (runtime node storage and hashing).
		 * @tparam R An input range of symbols.
		 * @tparam Proj Lvalue-invocable projection returning an integral symbol.
		 * @param word The projected symbol range.
		 * @param proj Symbol projection.
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		[[nodiscard]] std::optional<node_id> find(R &&word, Proj proj = {}) const {
			uint32 const n = node_of(word, proj);
			return n == nil ? std::nullopt : std::optional<node_id>(n);
		}

		/**
		 * @brief Number of stored occurrences whose word begins with @p prefix.
		 * @complexity O(len · expected BST depth)
		 * @note Not constexpr (runtime node storage and hashing).
		 * @tparam R An input range of symbols.
		 * @tparam Proj Lvalue-invocable projection returning an integral symbol.
		 * @param prefix The projected symbol range.
		 * @param proj Symbol projection.
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		[[nodiscard]] std::size_t count_with_prefix(R &&prefix, Proj proj = {}) const {
			uint32 const n = node_of(prefix, proj);
			return n == nil ? 0 : pool_[n].prefix_count;
		}

		/** @brief The root handle — the empty prefix.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] node_id root() const { return 0; }

		/** @brief The child of @p n along @p sym, or `std::nullopt` if there is none.
		 * @note Not constexpr (runtime node storage and hashing).
		 * @tparam Sym Integral symbol type.
		 * @param n The node to leave.
		 * @param sym Numeric edge label; out-of-alphabet values miss or reset.
		 * @pre The node handle belongs to this object (asserted).
		 */
		template<std::integral Sym>
		[[nodiscard]] std::optional<node_id> walk(node_id n, Sym sym) const {
			atl_assert(n < pool_.size());
			if (!atl::detail::symbol_below(sym, Alphabet)) return std::nullopt;
			uint32 const c = step(n, uint32(sym));
			return c == nil ? std::nullopt : std::optional<node_id>(c);
		}

		/**
		 * @brief Visit `(uint32 symbol, node_id child)` for every child of
		 *        @p n, in unspecified (hashed) order.
		 * @complexity O(children of @p n); allocates a traversal stack per call
		 * @note Not constexpr (runtime node storage and hashing).
		 * @pre The node handle belongs to this object (asserted).
		 * @tparam Visit Invocable on `(uint32 symbol, uint32 child)`.
		 * @param n The node whose children are visited.
		 * @param visit Callback; must not change this trie during traversal.
		 */
		template<child_visitor Visit>
		void for_each_child(node_id n, Visit visit) const {
			atl_assert(n < pool_.size());
			pool_.for_each(pool_[n].children, [&](uint32 c) { std::invoke(visit, pool_[c].symbol, node_id(c)); });
		}

		/** @brief Occurrences of the word ending exactly at @p n.
		 * @note Not constexpr (runtime node storage and hashing).
		 * @pre The node handle belongs to this object (asserted).
		 */
		[[nodiscard]] std::size_t word_count(node_id n) const {
			atl_assert(n < pool_.size());
			return pool_[n].word_count;
		}

		/** @brief Occurrences whose word passes through @p n (i.e. carries its prefix).
		 * @note Not constexpr (runtime node storage and hashing).
		 * @pre The node handle belongs to this object (asserted).
		 */
		[[nodiscard]] std::size_t prefix_count(node_id n) const {
			atl_assert(n < pool_.size());
			return pool_[n].prefix_count;
		}

		/** @brief Number of stored words counted with multiplicity.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		std::size_t size() const { return pool_[0].prefix_count; }

		/** @brief Whether the multiset contains no occurrences.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		bool empty() const { return size() == 0; }

		/** @brief Number of trie nodes (root included) — the memory footprint driver.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] std::size_t node_count() const { return pool_.size(); }
	};
}
