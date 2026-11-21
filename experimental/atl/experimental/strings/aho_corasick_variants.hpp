#pragma once
#include <atl/strings/aho_corasick.hpp>
#include <atl/strings/dense_transitions.hpp>
#include <atl/strings/detail/symbols.hpp>
#include <atl/strings/trie.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/prefetch.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Dense automaton and row-fill alternatives to sparse Aho–Corasick.
 * The dense engine trades O(nodes * Alphabet) storage for one-load steps;
 * its build/scan comparison is not measured yet in the supplied records.
 * Row fills vary zero initialization, prefetch threshold and row-line hints,
 * omitting payload prefetch because public node access does not expose addresses.
 * Xeon Platinum 8358, GCC 15.2.0, RelWithDebInfo, assertions off,
 * -O2/native, 2026-11-02 (benchmarks/strings/aho_corasick.cpp): sigma=26
 * 1e5x10 table ms are library 72.3/71.9, zeroed first-line 76.4/75.2,
 * overwrite first-line 73.9/74.5. No row policy wins on every shape.
 */
/** @brief Alternative algorithms and data-structure implementations for comparison. */
namespace atl::experimental {
	/**
	 * @brief Aho–Corasick automaton storing the complete transition table
	 *        `go[v][c]` for every node and every symbol of `[0, Alphabet)`:
	 *        @ref step is one load. Mirrors @ref atl::aho_corasick, with the
	 *        same node ids (first-touch creation order), so the two compare
	 *        node for node.
	 *
	 * Memory is 4·Alphabet bytes per node on top of ~28 B of payload, built in
	 * O(L + nodes·Alphabet): only for small dense alphabets.
	 *
	 * @tparam Alphabet Exclusive positive symbol bound, below 2^32.
	 * @pre Node count, total pattern count and symbols obey atl::aho_corasick bounds;
	 *      the dense table fits size_t and vector limits (asserted).
	 * @note Not constexpr (runtime vector operations in this contender). No recursion
	 *       grows with node count or pattern length. Spans follow storage through
	 *       moves and expire on assignment or destruction. Allocation/projection
	 *       exceptions propagate. The dense engine comparison is not measured yet;
	 *       permanent O(nodes * Alphabet) storage limits its alphabet.
	 */
	template<std::size_t Alphabet>
		requires(Alphabet >= 1 && Alphabet < (uint64(1) << 32))
	class aho_corasick_by_dense_table {
		static constexpr uint32 nil = uint32(-1);
		static constexpr std::size_t sigma = Alphabet;

		struct node {
			uint32 parent = nil, symbol = 0, link = 0, output = nil;
			uint32 length = 0, words = 0, matches = 0;
		};

		std::vector<node> nodes_;
		std::vector<uint32> go_; // row-major nodes x sigma: trie children, then the full transition
		std::vector<uint32> word_node_;
		std::vector<uint32> by_depth_;
		std::vector<uint32> word_offset_;
		std::vector<uint32> word_ids_;

		uint32 add_node() {
			atl_assert(nodes_.size() < uint64(nil)); // ids are uint32, nil reserved
			constexpr std::size_t max_cells = (std::numeric_limits<std::size_t>::max)() / sizeof(uint32);
			atl_assert(nodes_.size() < max_cells / sigma);
			atl_assert(sigma <= go_.max_size() && go_.size() <= go_.max_size() - sigma);
			nodes_.emplace_back();
			go_.resize(go_.size() + sigma, nil);
			return uint32(nodes_.size() - 1);
		}

		void bucket_words() {
			atl_assert(word_node_.size() < (uint64(1) << 32)); // pattern ids are uint32
			atl_assert(nodes_.size() < word_offset_.max_size());
			word_offset_.assign(nodes_.size() + 1, 0);
			for (uint32 const v : word_node_) ++word_offset_[v + 1];
			for (std::size_t v = 1; v < word_offset_.size(); ++v) word_offset_[v] += word_offset_[v - 1];
			word_ids_.resize(word_node_.size());
			std::vector<uint32> fill(word_offset_.begin(), word_offset_.end() - 1);
			for (uint32 i = 0; i < uint32(word_node_.size()); ++i) word_ids_[fill[word_node_[i]]++] = i;
		}

	public:
		/** @brief A handle to a node, as in @ref atl::aho_corasick. */
		using node_id = uint32;

		/**
		 * @brief Build from @p words seen through @p proj: trie, suffix links,
		 *        match counts and the complete transition table.
		 * @complexity O(L + nodes · Alphabet)
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word_range<R, Proj>
		explicit aho_corasick_by_dense_table(R &&words, Proj proj = {}) {
			add_node();
			if constexpr (std::ranges::sized_range<R>) {
				auto const count = std::ranges::size(words);
				atl_assert(count <= nil);
				word_node_.reserve(std::size_t(count));
			}
			for (auto &&w : atl::detail::input_elements(words)) {
				atl_assert(word_node_.size() < nil);
				uint32 v = 0;
				for (auto &&e : atl::detail::input_elements(w)) {
					auto const s = std::invoke(proj, std::forward<decltype(e)>(e));
					atl_assert(atl::detail::symbol_below(s, Alphabet));
					std::size_t const at = std::size_t(v) * sigma + uint32(s);
					if (go_[at] == nil) {
						uint32 const u = add_node();
						go_[at] = u;
						nodes_[u].parent = v;
						nodes_[u].symbol = uint32(s);
						nodes_[u].length = nodes_[v].length + 1;
					}
					v = go_[at];
				}
				++nodes_[v].words;
				word_node_.push_back(v);
			}
			// BFS; a row still holds raw trie children (or nil) until its node
			// is dequeued, and suffix_link(v) is shallower, so its row is final
			by_depth_.reserve(nodes_.size());
			by_depth_.push_back(0);
			for (std::size_t i = 0; i < by_depth_.size(); ++i) {
				uint32 const v = by_depth_[i];
				node &nv = nodes_[v];
				if (v != 0) {
					node const &nl = nodes_[nv.link];
					nv.matches = nv.words + nl.matches;
					nv.output = nl.words != 0 ? nv.link : nl.output;
				} else {
					nv.matches = nv.words;
				}
				uint32 *const row = go_.data() + std::size_t(v) * sigma;
				uint32 const *const fallback = go_.data() + std::size_t(nv.link) * sigma;
				for (std::size_t c = 0; c < sigma; ++c) {
					if (uint32 const u = row[c]; u != nil) {
						nodes_[u].link = v == 0 ? 0 : fallback[c];
						by_depth_.push_back(u);
					} else {
						row[c] = v == 0 ? 0 : fallback[c];
					}
				}
			}
			bucket_words();
		}

		/** @brief The automaton of no patterns: just the root.
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		aho_corasick_by_dense_table() : nodes_(1), go_(sigma, 0), by_depth_{0}, word_offset_{0, 0} {}

		/** @brief The root, the empty prefix (id 0), where a text scan starts.
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		[[nodiscard]] node_id root() const { return 0; }

		/**
		 * @brief The matching transition, one table load; symbols outside the
		 *        alphabet reset to the root.
		 * @complexity O(1)
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		template<std::integral Sym>
		[[nodiscard]] node_id step(node_id v, Sym sym) const {
			atl_assert(v < nodes_.size());
			if (!atl::detail::symbol_below(sym, Alphabet)) return 0;
			return go_[std::size_t(v) * sigma + uint32(sym)];
		}

		/**
		 * @brief The pure trie edge, or `std::nullopt`. A table entry is a
		 *        trie child exactly when it is one level deeper: every
		 *        fallback target is a suffix, never deeper than @p v.
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		template<std::integral Sym>
		[[nodiscard]] std::optional<node_id> walk(node_id v, Sym sym) const {
			atl_assert(v < nodes_.size());
			if (!atl::detail::symbol_below(sym, Alphabet)) return std::nullopt;
			uint32 const t = go_[std::size_t(v) * sigma + uint32(sym)];
			return nodes_[t].length == nodes_[v].length + 1 ? std::optional<node_id>(t) : std::nullopt;
		}

		/** @brief The length of @p v's prefix (its trie depth); 0 at the root.
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		[[nodiscard]] uint32 length(node_id v) const {
			atl_assert(v < nodes_.size());
			return nodes_[v].length;
		}

		/** @brief The nearest proper suffix of @p v's prefix at which a pattern ends, or `std::nullopt`, as in @ref atl::aho_corasick::output_link.
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		[[nodiscard]] std::optional<node_id> output_link(node_id v) const {
			atl_assert(v < nodes_.size());
			return nodes_[v].output == nil ? std::nullopt : std::optional<node_id>(nodes_[v].output);
		}

		/** @brief The ids (insertion indices) of the patterns ending exactly at @p v's prefix, ascending.
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		[[nodiscard]] std::span<uint32 const> words_at(node_id v) const {
			atl_assert(v < nodes_.size());
			return std::span<uint32 const>(word_ids_).subspan(word_offset_[v], word_offset_[v + 1] - word_offset_[v]);
		}

		/** @brief Patterns (with multiplicity) ending exactly at @p v's prefix.
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		[[nodiscard]] std::size_t word_count(node_id v) const {
			atl_assert(v < nodes_.size());
			return nodes_[v].words;
		}

		/** @brief Patterns (with multiplicity) ending at @p v's prefix or at any of its suffixes.
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		[[nodiscard]] std::size_t match_count(node_id v) const {
			atl_assert(v < nodes_.size());
			return nodes_[v].matches;
		}

		/**
		 * @brief The node of the longest proper suffix of @p v's prefix that is also a pattern prefix.
		 * @pre `v != root()`
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		[[nodiscard]] node_id suffix_link(node_id v) const {
			atl_assert(v != 0 && v < nodes_.size());
			return nodes_[v].link;
		}

		/**
		 * @brief @p v's prefix minus its last symbol.
		 * @pre `v != root()`
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		[[nodiscard]] node_id parent(node_id v) const {
			atl_assert(v != 0 && v < nodes_.size());
			return nodes_[v].parent;
		}

		/**
		 * @brief The last symbol of @p v's prefix (its trie edge).
		 * @pre `v != root()`
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		[[nodiscard]] uint32 symbol(node_id v) const {
			atl_assert(v != 0 && v < nodes_.size());
			return nodes_[v].symbol;
		}

		/** @brief The end node of the @p i-th inserted pattern (insertion order).
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		[[nodiscard]] node_id word_node(std::size_t i) const {
			atl_assert(i < word_node_.size());
			return word_node_[i];
		}

		/** @brief All nodes in BFS order: parents preceding children in both the trie and the suffix-link tree.
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		[[nodiscard]] std::span<node_id const> nodes_by_depth() const { return by_depth_; }

		/** @brief A copy of the stored table, as @ref atl::aho_corasick::transition_table returns it. */
		[[nodiscard]] dense_transitions transition_table() const { return {go_, sigma}; }

		/**
		 * @brief The stored table restricted to the symbols `[0, sigma_limit)`,
		 *        as @ref atl::aho_corasick::transition_table(std::size_t)
		 *        returns it: `table(v, c) == step(v, c)` for every node and
		 *        `c < sigma_limit`.
		 * @pre `1 <= sigma_limit <= Alphabet` (asserted)
		 * @complexity O(node_count · sigma_limit)
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		[[nodiscard]] dense_transitions transition_table(std::size_t const sigma_limit) const {
			atl_assert(sigma_limit >= 1 && sigma_limit <= sigma);
			std::vector<uint32> next;
			next.reserve(nodes_.size() * sigma_limit);
			for (std::size_t v = 0; v < nodes_.size(); ++v) {
				auto const first = go_.begin() + std::ptrdiff_t(v * sigma);
				next.insert(next.end(), first, first + std::ptrdiff_t(sigma_limit));
			}
			return {std::move(next), sigma_limit};
		}

		/** @brief Number of inserted patterns (with multiplicity).
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		std::size_t size() const { return word_node_.size(); }

		/** @brief Whether no pattern was inserted.
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		bool empty() const { return word_node_.empty(); }

		/** @brief Number of nodes, root included.
		 * @note Not constexpr (mirrors the runtime-only sparse automaton interface).
		 */
		[[nodiscard]] std::size_t node_count() const { return nodes_.size(); }
	};

	/** @brief Which cache lines of a table row the fill of @ref transition_table_by_row_options prefetches. */
	enum class row_prefetch {
		none,			///< no prefetching at all
		first_line,		///< the first line of the destination and fallback rows (first-line policy)
		all_lines,		///< every line of both rows, read hints
		all_lines_write ///< every line of both rows, the destination with a write hint
	};

	namespace detail {
		/** @brief Cap on the lines prefetched per row, so a huge alphabet does not flood the fill buffers. */
		inline constexpr std::size_t max_prefetched_lines = 16;

		/**
		 * @brief Prefetch the cache lines of the `sigma`-cell row at @p row:
		 *        the line of its first cell, and for @p Rows past
		 *        `first_line` every further 16-cell step and the last cell
		 *        (a row is not line-aligned); rows longer than
		 *        @ref max_prefetched_lines lines get only their first that many.
		 * @tparam IsWrite Whether the row is about to be written.
		 */
		template<bool IsWrite, row_prefetch Rows>
		void prefetch_row(uint32 const *const row, std::size_t const sigma) {
			atl::prefetch<IsWrite>(row);
			if constexpr (Rows == row_prefetch::all_lines || Rows == row_prefetch::all_lines_write) {
				std::size_t const stop = (std::min)(sigma, max_prefetched_lines * 16);
				for (std::size_t c = 16; c < stop; c += 16) atl::prefetch<IsWrite>(row + c);
				if (sigma <= max_prefetched_lines * 16) atl::prefetch<IsWrite>(row + sigma - 1);
			}
		}
	}

	/**
	 * @brief Build transitions by row options: allocation and cache-line hints vary.
	 * @tparam IsOverwritten Use uninitialized allocation and zero only the root row.
	 * @tparam Rows Cache lines to prefetch; every policy uses the same BFS row copies.
	 * @tparam Alphabet Symbol bound of the source automaton.
	 * @param ac Source sparse automaton; suffix rows precede dependent rows in BFS order.
	 * @param sigma Restrict symbols to `[0, sigma)`.
	 * @param prefetch_min_cells Minimum table size to prefetch; zero always, SIZE_MAX never.
	 * @pre `1 <= sigma <= Alphabet`, table allocation addressable (asserted).
	 * @return A complete, independent dense table.
	 * @complexity O(node_count * sigma) time and storage.
	 * @note Not constexpr (source automaton uses runtime hashing).
	 *       The zeroed first-line instantiation is the certifying experimental oracle.
	 *       On sigma=2 1e4x20, zeroed/overwrite takes 0.808/0.788 ms at -O2
	 *       and 0.809/0.813 native; removing zeroing is not uniformly faster.
	 *       Every-line write hints take 0.922/0.910 ms there. See file record above.
	 */
	template<bool IsOverwritten, row_prefetch Rows, std::size_t Alphabet>
		requires(Alphabet <= (uint64(1) << 32))
	[[nodiscard]] auto transition_table_by_row_options(atl::aho_corasick<Alphabet> const &ac, std::size_t const sigma,
													   std::size_t const prefetch_min_cells = std::size_t(1) << 18) {
		atl_assert(sigma >= 1 && sigma <= Alphabet);
		std::size_t const states = ac.node_count();
		constexpr std::size_t max_cells = (std::numeric_limits<std::size_t>::max)() / sizeof(uint32);
		atl_assert(sigma <= max_cells / states);
		std::size_t const cells = states * sigma;
		std::unique_ptr<uint32[]> next;
		std::vector<uint32> zeroed;
		uint32 *table = nullptr;
		if constexpr (IsOverwritten) {
			next = std::make_unique_for_overwrite<uint32[]>(cells);
			table = next.get();
			std::ranges::fill_n(table, std::ptrdiff_t(sigma), uint32(0));
		} else {
			zeroed.resize(cells);
			table = zeroed.data();
		}
		constexpr std::size_t ahead = 8;
		bool const is_prefetched = Rows != row_prefetch::none && cells >= prefetch_min_cells;
		auto const order = ac.nodes_by_depth();
		std::size_t const count = order.size();
		std::size_t next_child = 1;
		for (std::size_t i = 0; i < count; ++i) {
			if (is_prefetched && i + ahead < count) {
				uint32 const w = order[i + ahead];
				detail::prefetch_row<Rows == row_prefetch::all_lines_write, Rows>(table + std::size_t(w) * sigma, sigma);
				detail::prefetch_row<false, Rows>(table + std::size_t(ac.suffix_link(w)) * sigma, sigma);
			}
			uint32 const v = order[i];
			uint32 *const row = table + std::size_t(v) * sigma;
			if (v != 0) {
				uint32 const *const fallback = table + std::size_t(ac.suffix_link(v)) * sigma;
				std::ranges::copy(fallback, fallback + sigma, row);
			}
			for (; next_child < count; ++next_child) {
				uint32 const u = order[next_child];
				if (ac.parent(u) != v) break;
				if (ac.symbol(u) < sigma) row[ac.symbol(u)] = u;
			}
		}
		if constexpr (IsOverwritten)
			return atl::detail::dense_transitions_builder::adopt(std::move(next), cells, sigma);
		else
			return atl::dense_transitions(std::move(zeroed), sigma);
	}
}
