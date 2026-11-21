#pragma once
#include <atl/ds/flat_hash_map.hpp>
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

/**
 * @file
 * @brief Sparse trie storage alternatives: a global edge hash map and a raw-symbol BST.
 * Both use iterative descents and match a multiset oracle in tests/strings/trie.cpp.
 * Xeon Platinum 8358, GCC 15.2.0, RelWithDebInfo, assertions off, -O2/native,
 * 2026-10-15 (benchmarks/strings/trie.cpp): random 1e5x10 at U=1e6 build/query ms:
 * library 38.6/35.8 and 35.2/37.2; by global hash 148/77.8 and 60.1/80.8;
 * by symbol BST 31.6/24.1 and 31.0/24.2. Sorted global-hash build/query:
 * 158/89.7 and 65.5/84.8; library 46.9/55.3 and 43.1/52.4.
 * Raw-symbol BST is excluded from sorted timing because increasing symbols form
 * a linear sibling chain. The global map loses query throughput; raw-symbol
 * ordering lacks the expected logarithmic lookup bound of hashed ordering.
 */
namespace atl::experimental {
	/**
	 * @brief Trie whose edges live in one global `(parent<<32 | symbol) -> child`
	 *        hash map. Expected constant-time edge probes; random queries take 77.8/80.8 ms
	 *        versus atl::trie 35.8/37.2 ms (-O2/native; file record above).
	 * @tparam Alphabet Exclusive integral symbol bound, at most 2^32.
	 * @note Not constexpr (runtime hash map). Handles survive updates and reserve;
	 *       clear invalidates them. Count bounds and symbol semantics match atl::trie;
	 *       erase requires stable projections across its two passes.
	 */
	template<std::size_t Alphabet>
		requires(Alphabet <= (uint64(1) << 32))
	class trie_by_global_hash {
		static constexpr uint32 nil = uint32(-1);

		struct node {
			uint32 word = 0, sub = 0;
		};

		std::vector<node> nodes_;
		flat_hash_map<uint64, uint32> edge_;

		[[nodiscard]] static uint64 key(uint32 p, uint32 s) { return (uint64(p) << 32) | s; }

		[[nodiscard]] uint32 step(uint32 p, uint32 s) const {
			auto it = edge_.find(key(p, s));
			return it == edge_.end() ? nil : it->second;
		}

		/**
		 * @brief Child of @p p along @p s, creating the node if absent: one
		 *        find-or-insert probe, a fresh slot reading 0 (the root, never
		 *        a child).
		 */
		uint32 link(uint32 p, uint32 s) {
			uint32 &c = edge_[key(p, s)];
			if (c == 0) {
				atl_assert(nodes_.size() < nil);
				c = uint32(nodes_.size());
				nodes_.push_back(node{});
			}
			return c;
		}

		template<typename R, typename Proj>
			requires atl::trie_word<R, Proj>
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
		/** @brief A handle to a node: @ref root, then @ref walk. */
		using node_id = uint32;

		/** @brief An empty trie (the root node only).
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		trie_by_global_hash() { nodes_.push_back(node{}); }

		/** @brief Reserve capacity for up to @p total_symbols edges.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		void reserve(std::size_t total_symbols) {
			atl_assert(total_symbols < nil && total_symbols < (std::numeric_limits<std::size_t>::max)());
			nodes_.reserve(total_symbols + 1);
			edge_.reserve(total_symbols);
		}

		/** @brief Drop all words.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		void clear() {
			nodes_.clear();
			nodes_.push_back(node{});
			edge_.clear();
		}

		/**
		 * @brief Insert one occurrence of @p word, each element seen through
		 *        @p proj.
		 * @pre every projected symbol is in [0, Alphabet)
		 * @complexity O(len) child lookups
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		template<typename R, typename Proj = std::identity>
			requires atl::trie_word<R, Proj>
		void insert(R &&word, Proj proj = {}) {
			atl_assert(nodes_[0].sub < nil);
			++nodes_[0].sub;
			uint32 cur = 0;
			for (auto &&e : atl::detail::input_elements(word)) {
				auto const s = std::invoke(proj, std::forward<decltype(e)>(e));
				atl_assert(atl::detail::symbol_below(s, Alphabet));
				cur = link(cur, uint32(s));
				++nodes_[cur].sub;
			}
			++nodes_[cur].word;
		}

		/**
		 * @brief Remove one occurrence of @p word; returns whether it was
		 *        present. Nodes are not reclaimed, so handles stay valid.
		 * @complexity O(len) child lookups
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		template<typename R, typename Proj = std::identity>
			requires atl::trie_word<R, Proj> && std::ranges::forward_range<R>
		bool erase(R &&word, Proj proj = {}) {
			uint32 const term = node_of(word, proj);
			if (term == nil || nodes_[term].word == 0) return false;
			--nodes_[term].word;
			--nodes_[0].sub;
			uint32 cur = 0;
			for (auto &&e : atl::detail::input_elements(word)) {
				cur = step(cur, uint32(std::invoke(proj, std::forward<decltype(e)>(e))));
				--nodes_[cur].sub;
			}
			return true;
		}

		/**
		 * @brief Multiplicity of the exact word @p word.
		 * @complexity O(len) child lookups
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		template<typename R, typename Proj = std::identity>
			requires atl::trie_word<R, Proj>
		[[nodiscard]] std::size_t count(R &&word, Proj proj = {}) const {
			uint32 n = node_of(word, proj);
			return n == nil ? 0 : nodes_[n].word;
		}

		/**
		 * @brief Whether @p word is stored at least once.
		 * @complexity O(len) child lookups
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		template<typename R, typename Proj = std::identity>
			requires atl::trie_word<R, Proj>
		[[nodiscard]] bool contains(R &&word, Proj proj = {}) const {
			uint32 n = node_of(word, proj);
			return n != nil && nodes_[n].word != 0;
		}

		/**
		 * @brief Number of stored occurrences whose word begins with
		 *        @p prefix.
		 * @complexity O(len) child lookups
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		template<typename R, typename Proj = std::identity>
			requires atl::trie_word<R, Proj>
		[[nodiscard]] std::size_t count_with_prefix(R &&prefix, Proj proj = {}) const {
			uint32 n = node_of(prefix, proj);
			return n == nil ? 0 : nodes_[n].sub;
		}

		/** @brief The root handle — the empty prefix.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] node_id root() const { return 0; }

		/** @brief The child of @p n along @p sym, or `std::nullopt` if there
		 *         is none.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		template<std::integral Sym>
		[[nodiscard]] std::optional<node_id> walk(node_id n, Sym sym) const {
			atl_assert(n < nodes_.size());
			if (!atl::detail::symbol_below(sym, Alphabet)) return std::nullopt;
			uint32 c = step(n, uint32(sym));
			return c == nil ? std::nullopt : std::optional<node_id>(c);
		}

		/** @brief Occurrences of the word ending exactly at @p n.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] std::size_t word_count(node_id n) const {
			atl_assert(n < nodes_.size());
			return nodes_[n].word;
		}

		/** @brief Occurrences whose word passes through @p n.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] std::size_t prefix_count(node_id n) const {
			atl_assert(n < nodes_.size());
			return nodes_[n].sub;
		}

		/** @brief Number of stored word occurrences.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		std::size_t size() const { return nodes_[0].sub; }

		/** @brief Whether no word is stored.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		bool empty() const { return size() == 0; }

		/** @brief Number of allocated nodes, the root included.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] std::size_t node_count() const { return nodes_.size(); }
	};

	/**
	 * @brief Ternary trie whose siblings are ordered by the raw symbol. Build/query takes 31.6/24.1 ms (-O2, file record above)
	 *        on random input but degenerates to an O(alphabet)
	 *        chain under sorted symbols under sorted insertion.
	 * @tparam Alphabet Exclusive integral symbol bound, at most 2^32.
	 * @note Not constexpr (runtime vector operations in this contender). Handles survive
	 *       updates and reserve; clear invalidates them. Bounds and symbol semantics
	 *       match atl::trie. Erase requires stable projections across two passes.
	 */
	template<std::size_t Alphabet>
		requires(Alphabet <= (uint64(1) << 32))
	class trie_by_symbol_bst {
		static constexpr uint32 nil = uint32(-1);

		struct node {
			uint32 c = 0, lo = nil, hi = nil, eq = nil, word = 0, sub = 0;
		};

		std::vector<node> nodes_;

		uint32 make(uint32 sym) {
			atl_assert(nodes_.size() < nil);
			nodes_.push_back(node{});
			uint32 n = uint32(nodes_.size() - 1);
			nodes_[n].c = sym;
			return n;
		}

		uint32 link(uint32 parent, uint32 sym) {
			if (nodes_[parent].eq == nil) return nodes_[parent].eq = make(sym);
			uint32 cur = nodes_[parent].eq;
			for (;;) {
				uint32 const c = nodes_[cur].c;
				if (c == sym) return cur;
				if (sym < c) {
					if (nodes_[cur].lo == nil) return nodes_[cur].lo = make(sym);
					cur = nodes_[cur].lo;
				} else {
					if (nodes_[cur].hi == nil) return nodes_[cur].hi = make(sym);
					cur = nodes_[cur].hi;
				}
			}
		}

		[[nodiscard]] uint32 step(uint32 parent, uint32 sym) const {
			uint32 cur = nodes_[parent].eq;
			while (cur != nil) {
				uint32 const c = nodes_[cur].c;
				if (c == sym) return cur;
				cur = sym < c ? nodes_[cur].lo : nodes_[cur].hi;
			}
			return nil;
		}

		template<typename R, typename Proj>
			requires atl::trie_word<R, Proj>
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
		/** @brief A handle to a node: @ref root, then @ref walk. */
		using node_id = uint32;

		/** @brief An empty trie (the root node only).
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		trie_by_symbol_bst() { nodes_.push_back(node{}); }

		/** @brief Reserve capacity for up to @p total_symbols edges.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		void reserve(std::size_t total_symbols) {
			atl_assert(total_symbols < nil && total_symbols < (std::numeric_limits<std::size_t>::max)());
			nodes_.reserve(total_symbols + 1);
		}

		/** @brief Drop all words.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		void clear() {
			nodes_.clear();
			nodes_.push_back(node{});
		}

		/**
		 * @brief Insert one occurrence of @p word, each element seen through
		 *        @p proj.
		 * @pre every projected symbol is in [0, Alphabet)
		 * @complexity O(len) child lookups
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		template<typename R, typename Proj = std::identity>
			requires atl::trie_word<R, Proj>
		void insert(R &&word, Proj proj = {}) {
			atl_assert(nodes_[0].sub < nil);
			++nodes_[0].sub;
			uint32 cur = 0;
			for (auto &&e : atl::detail::input_elements(word)) {
				auto const s = std::invoke(proj, std::forward<decltype(e)>(e));
				atl_assert(atl::detail::symbol_below(s, Alphabet));
				cur = link(cur, uint32(s));
				++nodes_[cur].sub;
			}
			++nodes_[cur].word;
		}

		/**
		 * @brief Remove one occurrence of @p word; returns whether it was
		 *        present. Nodes are not reclaimed, so handles stay valid.
		 * @complexity O(len) child lookups
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		template<typename R, typename Proj = std::identity>
			requires atl::trie_word<R, Proj> && std::ranges::forward_range<R>
		bool erase(R &&word, Proj proj = {}) {
			uint32 const term = node_of(word, proj);
			if (term == nil || nodes_[term].word == 0) return false;
			--nodes_[term].word;
			--nodes_[0].sub;
			uint32 cur = 0;
			for (auto &&e : atl::detail::input_elements(word)) {
				cur = step(cur, uint32(std::invoke(proj, std::forward<decltype(e)>(e))));
				--nodes_[cur].sub;
			}
			return true;
		}

		/**
		 * @brief Multiplicity of the exact word @p word.
		 * @complexity O(len) child lookups
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		template<typename R, typename Proj = std::identity>
			requires atl::trie_word<R, Proj>
		[[nodiscard]] std::size_t count(R &&word, Proj proj = {}) const {
			uint32 n = node_of(word, proj);
			return n == nil ? 0 : nodes_[n].word;
		}

		/**
		 * @brief Whether @p word is stored at least once.
		 * @complexity O(len) child lookups
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		template<typename R, typename Proj = std::identity>
			requires atl::trie_word<R, Proj>
		[[nodiscard]] bool contains(R &&word, Proj proj = {}) const {
			uint32 n = node_of(word, proj);
			return n != nil && nodes_[n].word != 0;
		}

		/**
		 * @brief Number of stored occurrences whose word begins with
		 *        @p prefix.
		 * @complexity O(len) child lookups
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		template<typename R, typename Proj = std::identity>
			requires atl::trie_word<R, Proj>
		[[nodiscard]] std::size_t count_with_prefix(R &&prefix, Proj proj = {}) const {
			uint32 n = node_of(prefix, proj);
			return n == nil ? 0 : nodes_[n].sub;
		}

		/** @brief The root handle — the empty prefix.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] node_id root() const { return 0; }

		/** @brief The child of @p n along @p sym, or `std::nullopt` if there
		 *         is none.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		template<std::integral Sym>
		[[nodiscard]] std::optional<node_id> walk(node_id n, Sym sym) const {
			atl_assert(n < nodes_.size());
			if (!atl::detail::symbol_below(sym, Alphabet)) return std::nullopt;
			uint32 c = step(n, uint32(sym));
			return c == nil ? std::nullopt : std::optional<node_id>(c);
		}

		/** @brief Occurrences of the word ending exactly at @p n.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] std::size_t word_count(node_id n) const {
			atl_assert(n < nodes_.size());
			return nodes_[n].word;
		}

		/** @brief Occurrences whose word passes through @p n.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] std::size_t prefix_count(node_id n) const {
			atl_assert(n < nodes_.size());
			return nodes_[n].sub;
		}

		/** @brief Number of stored word occurrences.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		std::size_t size() const { return nodes_[0].sub; }

		/** @brief Whether no word is stored.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		bool empty() const { return size() == 0; }

		/** @brief Number of allocated nodes, the root included.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] std::size_t node_count() const { return nodes_.size(); }
	};
}
