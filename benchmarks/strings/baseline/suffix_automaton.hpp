#pragma once
#include <atl/ds/detail/hashed_child_pool.hpp>
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
#include <utility>
#include <vector>

/** @brief Benchmark-local construction contenders with the public query contracts. */
namespace atl::baseline {
	/**
	 * @brief Generalized suffix automaton that finds split transitions and reinserts cloned child edges.
	 * @note Not constexpr (the child pool uses a process hash seed). See the matching
	 * public component for bounds and operation contracts. Traversals do not
	 * recurse per character or node. Full results: docs/suffix-automaton.md.
	 * Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo,
	 * 2026-10-15, -O2 -g -DNDEBUG -std=c++23, optionally -march=native.
	 * Random26 1e6 build: library 434/449 ms, by transition reinsertion 565/533 ms.
	 */
	template<std::size_t Alphabet>
		requires(Alphabet <= (uint64(1) << 32))
	class suffix_automaton_by_transition_reinsertion {
		static constexpr uint32 nil = uint32(-1);

		/** @brief A transition entry: a `detail::sibling_node` (the pool manages
		 * `symbol`/`left`/`right`) carrying the explicit DAG `target` state.
		 */
		struct transition {
			uint32 symbol = 0;
			uint32 left = nil, right = nil;
			uint32 target = nil;
		};

		struct state {
			uint32 length = 0;
			uint32 link = nil;
			uint32 children = nil;
			uint32 ending_count = 0;
		};

		::atl::detail::hashed_child_pool<transition> pool_;
		std::vector<state> states_;
		std::vector<uint32> word_end_;
		std::vector<std::pair<uint32, uint32>> scratch_; // split()'s transition copy buffer
		uint32 last_ = 0;

		uint32 new_state(uint32 len) {
			states_.push_back({len, nil, nil, 0});
			atl_assert(states_.size() < (uint64(1) << 32)); // ids are uint32, nil reserved
			return uint32(states_.size() - 1);
		}

		[[nodiscard]] uint32 step(uint32 s, uint32 sym) const {
			uint32 const t = pool_.find(states_[s].children, sym);
			return t == nil ? nil : pool_[t].target;
		}

		/**
		 * @brief The classic split: @p q is reached from @p p along @p sym but
		 *        represents strings longer than `length(p) + 1`. Carve that
		 *        shorter class out as a clone: copy q's transitions and link,
		 *        redirect the p-chain's `sym` transitions from q to the clone,
		 *        and re-hang q below it.
		 * @return The clone's id.
		 */
		uint32 split(uint32 p, uint32 sym, uint32 q) {
			uint32 const clone = new_state(states_[p].length + 1);
			states_[clone].link = states_[q].link;
			// collect-then-emplace: emplacing into the pool while for_each
			// walks it would race the arena reallocation
			scratch_.clear();
			pool_.for_each(states_[q].children,
						   [&](uint32 t) { scratch_.emplace_back(pool_[t].symbol, pool_[t].target); });
			for (auto const &[s, target] : scratch_) {
				auto const placed = pool_.emplace(states_[clone].children, s);
				states_[clone].children = placed.root;
				pool_[placed.node].target = target;
			}
			for (uint32 v = p; v != nil; v = states_[v].link) {
				uint32 const t = pool_.find(states_[v].children, sym);
				if (t == nil || pool_[t].target != q) break;
				pool_[t].target = clone;
			}
			states_[q].link = clone;
			return clone;
		}

		template<typename R, typename Proj>
			requires trie_word<R, Proj>
		[[nodiscard]] uint32 node_of(R &&w, Proj &proj) const {
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
		/** @brief A handle to a state (@ref atl::baseline::suffix_automaton_by_transition_reinsertion::root, @ref atl::baseline::suffix_automaton_by_transition_reinsertion::walk, @ref atl::baseline::suffix_automaton_by_transition_reinsertion::suffix_link). Stable
		 * for the automaton's lifetime. Ids are creation order, NOT topological
		 * — DPs should iterate @ref atl::baseline::suffix_automaton_by_transition_reinsertion::states_by_length.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		using state_id = uint32;

		/** @brief The empty automaton: just the initial state.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		suffix_automaton_by_transition_reinsertion() { states_.push_back({}); }

		/**
		 * @brief Append one symbol to the word in progress.
		 * @return The state of the whole current word (the new @ref atl::baseline::suffix_automaton_by_transition_reinsertion::last).
		 * @complexity Expected amortized O(log(degree + 1)); @p sym must be in `[0, Alphabet)`.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		template<std::integral Sym>
		state_id extend(Sym sym) {
			atl_assert(atl::detail::symbol_below(sym, Alphabet)); // in [0, Alphabet)
			atl_assert(states_[last_].length < (std::numeric_limits<uint32>::max)() - 1);
			uint32 const c = uint32(sym);
			if (uint32 const q = step(last_, c); q != nil) {
				// generalized case: the word in progress rejoins known ground
				last_ = states_[q].length == states_[last_].length + 1 ? q : split(last_, c, q);
			} else {
				uint32 const cur = new_state(states_[last_].length + 1);
				uint32 p = last_;
				for (; p != nil; p = states_[p].link) {
					auto const placed = pool_.emplace(states_[p].children, c);
					states_[p].children = placed.root;
					if (!placed.was_inserted) break;
					pool_[placed.node].target = cur;
				}
				if (p == nil) {
					states_[cur].link = 0;
				} else {
					uint32 const q = step(p, c);
					states_[cur].link = states_[q].length == states_[p].length + 1 ? q : split(p, c, q);
				}
				last_ = cur;
			}
			atl_assert(states_[last_].ending_count < (std::numeric_limits<uint32>::max)());
			++states_[last_].ending_count;
			return last_;
		}

		/**
		 * @brief End the word in progress: record its state as a word end (the
		 *        seeds @ref atl::baseline::suffix_automaton_by_transition_reinsertion::terminals propagates from) and reset so the next
		 *        @ref atl::baseline::suffix_automaton_by_transition_reinsertion::extend / @ref atl::baseline::suffix_automaton_by_transition_reinsertion::insert starts a fresh word. Ending an empty
		 *        word records nothing.
		 * @return The ended word's state (@ref atl::baseline::suffix_automaton_by_transition_reinsertion::root if it was empty).
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		state_id finish_word() {
			uint32 const ended = last_;
			if (ended != 0) word_end_.push_back(ended);
			last_ = 0;
			return ended;
		}

		/** @brief The state of the word currently in progress (@ref atl::baseline::suffix_automaton_by_transition_reinsertion::root if none).
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] state_id last() const { return last_; }

		/**
		 * @brief Add one whole word — @ref atl::baseline::suffix_automaton_by_transition_reinsertion::extend each projected symbol, then
		 *        @ref atl::baseline::suffix_automaton_by_transition_reinsertion::finish_word. Repeated calls build the generalized
		 *        automaton of the collection; duplicates count with multiplicity
		 *        in @ref atl::baseline::suffix_automaton_by_transition_reinsertion::occurrence_counts.
		 * @return The word's state.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		state_id insert(R &&word, Proj proj = {}) {
			for (auto &&e : atl::detail::input_elements(word)) extend(std::invoke(proj, std::forward<decltype(e)>(e)));
			return finish_word();
		}

		/** @brief Is @p word a substring of some inserted word? The empty word always
		 * is. @complexity Expected O(len · log(degree + 1))
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		[[nodiscard]] bool contains(R &&word, Proj proj = {}) const {
			return node_of(word, proj) != nil;
		}

		/** @brief The initial state — the class of the empty word (id 0).
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] state_id root() const { return 0; }

		/** @brief The target of @p s along @p sym, or `std::nullopt`.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		template<std::integral Sym>
		[[nodiscard]] std::optional<state_id> walk(state_id s, Sym sym) const {
			atl_assert(s < states_.size());
			if (!atl::detail::symbol_below(sym, Alphabet)) return std::nullopt;
			uint32 const t = step(s, uint32(sym));
			return t == nil ? std::nullopt : std::optional<state_id>(t);
		}

		/** @brief The class of @p s's next-shorter suffixes. @pre `s != root()`
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] state_id suffix_link(state_id s) const {
			atl_assert(s != 0 && s < states_.size());
			return states_[s].link;
		}

		/** @brief Length of the longest substring in @p s's class.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] uint32 length(state_id s) const {
			atl_assert(s < states_.size());
			return states_[s].length;
		}

		/** @brief Length of the shortest substring in @p s's class
		 * (= `length(suffix_link(s)) + 1`). @pre `s != root()`
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] uint32 shortest_length(state_id s) const {
			atl_assert(s != 0 && s < states_.size());
			return states_[states_[s].link].length + 1;
		}

		/** @brief Visit `(uint32 symbol, state_id target)` for each transition out of
		 * @p s, in unspecified (hashed) order.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		template<typename Visit>
			requires std::invocable<Visit &, uint32, state_id>
		void for_each_transition(state_id s, Visit visit) const {
			atl_assert(s < states_.size());
			pool_.for_each(states_[s].children,
						   [&](uint32 t) { std::invoke(visit, pool_[t].symbol, pool_[t].target); });
		}

		/**
		 * @brief All states in increasing @ref atl::baseline::suffix_automaton_by_transition_reinsertion::length order — a topological
		 *        order of both the suffix-link tree (parents first) and the
		 *        transition DAG, i.e. the iteration every automaton DP wants.
		 * @complexity O(state_count()) counting sort
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] std::vector<state_id> states_by_length() const {
			uint32 max_len = 0;
			for (auto const &s : states_) max_len = (std::max)(max_len, s.length);
			atl_assert(uint64(max_len) + 2 <= (std::numeric_limits<std::size_t>::max)());
			std::vector<uint32> bucket(std::size_t(max_len) + 2, 0);
			for (auto const &s : states_) ++bucket[s.length + 1];
			for (std::size_t l = 1; l < bucket.size(); ++l) bucket[l] += bucket[l - 1];
			std::vector<state_id> order(states_.size());
			for (uint32 v = 0; v < uint32(states_.size()); ++v) order[bucket[states_[v].length]++] = v;
			return order;
		}

		/**
		 * @brief Per-state occurrence counts: how many times the substrings of
		 *        each class occur across all inserted words, counted per word
		 *        instance (a duplicate word doubles its counts). Ending-position
		 *        seeds are propagated up the suffix-link tree in one pass. The
		 *        root's entry is the total number of positions (every position
		 *        ends the empty word).
		 * @complexity O(state_count())
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] std::vector<uint64> occurrence_counts() const {
			std::vector<uint64> occ(states_.size());
			for (uint32 v = 0; v < uint32(states_.size()); ++v) occ[v] = states_[v].ending_count;
			auto const order = states_by_length();
			for (uint32 i = uint32(order.size()); i-- > 1;) occ[states_[order[i]].link] += occ[order[i]];
			return occ;
		}

		/**
		 * @brief For each state, whether its class contains a suffix of some
		 *        finished word: the recorded word ends, propagated up the
		 *        suffix links. The root is always terminal (the empty suffix).
		 * @complexity O(state_count() + finished words)
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] std::vector<bool> terminals() const {
			std::vector<bool> mark(states_.size(), false);
			mark[0] = true;
			for (uint32 e : word_end_)
				for (uint32 v = e; v != nil && !mark[v]; v = states_[v].link) mark[v] = true;
			return mark;
		}

		/** @brief Number of distinct nonempty substrings over all inserted words:
		 * each state contributes its class size `length - shortest_length + 1`.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] uint64 distinct_substring_count() const {
			uint64 total = 0;
			for (uint32 v = 1; v < uint32(states_.size()); ++v)
				total += states_[v].length - states_[states_[v].link].length;
			return total;
		}

		/** @brief Number of states, the initial state included.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		std::size_t state_count() const { return states_.size(); }

		/** @brief Number of transitions over all states.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		std::size_t transition_count() const { return pool_.size(); }

		/** @brief Pre-allocate for @p symbols total input symbols (the classic bounds:
		 * at most `2·symbols` states and `3·symbols` transitions).
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		void reserve(std::size_t symbols) {
			atl_assert(symbols <= ((std::numeric_limits<std::size_t>::max)() - 1) / 3);
			atl_assert(uint64(symbols) <= ((uint64(1) << 32) - 2) / 3);
			states_.reserve(2 * symbols + 1);
			pool_.reserve(3 * symbols);
		}

		/** @brief Back to the empty automaton (memory kept).
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		void clear() {
			pool_.clear();
			states_.clear();
			states_.push_back({});
			word_end_.clear();
			last_ = 0;
		}
	};
}
