#pragma once

/**
 * @file
 * @brief Search contenders for Eppstein-style k shortest walks.
 * Class and forest forms either walk heap children or pop versions; popping
 * allocates O(log m) nodes per state. Class-versus-forest and walk-versus-pop comparisons are not measured yet.
 * The heap-construction and forest-walk tuning records are in docs/persistent-heap.md.
 */

#include <atl/experimental/graph/k_shortest_walks.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <limits>
#include <queue>
#include <ranges>
#include <utility>
#include <vector>

namespace atl::experimental {
	/**
	 * @brief The stages a search consumes: a shortest-path tree (`distance`,
	 *        `next`, `order` per vertex, as @ref atl::experimental::shortest_path_tree)
	 *        and sidetrack lists (`of(v)`: a range of `{extra, head}`, as @ref
	 *        atl::experimental::sidetrack_lists).
	 * @tparam Tree Indexed target-tree distances, parent vertices and settlement order.
	 * @tparam Lists Per-vertex input ranges of sidetrack-like references.
	 */
	template<typename Tree, typename Lists>
	concept eppstein_stages = requires(Tree const &tree, Lists const &lists, uint32 v) {
		{ tree.distance[v] } -> std::convertible_to<int64>;
		{ tree.next[v] } -> std::convertible_to<uint32>;
		{ tree.order } -> std::ranges::input_range;
		requires std::convertible_to<std::ranges::range_reference_t<decltype((tree.order))>, uint32>;
		{ lists.of(v) } -> std::ranges::input_range;
		requires requires(std::ranges::range_reference_t<decltype(lists.of(v))> item) {
			{ item.extra } -> std::convertible_to<int64>;
			{ item.head } -> std::convertible_to<uint32>;
		};
	};

	/** @brief Class heap supporting both version pops and borrowed subtree walks. @tparam Heap Copyable class versions with sidetrack values. */
	template<typename Heap>
	concept eppstein_class_heap = std::copyable<Heap> && std::default_initializable<Heap> && requires(Heap const &h, typename Heap::node_view view) {
		requires std::copyable<typename Heap::node_view>;
		{ h.top().extra } -> std::convertible_to<int64>;
		{ h.top().head } -> std::convertible_to<uint32>;
		{ h.empty() } -> std::convertible_to<bool>;
		{ h.pop() } -> std::same_as<Heap>;
		{ h.root() } -> std::same_as<typename Heap::node_view>;
		{ view.empty() } -> std::convertible_to<bool>;
		{ view.left() } -> std::same_as<typename Heap::node_view>;
		{ view.right() } -> std::same_as<typename Heap::node_view>;
		{ view.value().extra } -> std::convertible_to<int64>;
		{ view.value().head } -> std::convertible_to<uint32>;
	};

	/** @brief Forest heap supporting uint32 versions and subtree ids. @tparam Forest Default-constructible heap pool with sidetrack values. */
	template<typename Forest>
	concept eppstein_forest_heap = std::default_initializable<Forest> && requires(Forest &f, Forest const &cf, uint32 h) {
		{ cf.top(h).extra } -> std::convertible_to<int64>;
		{ cf.top(h).head } -> std::convertible_to<uint32>;
		{ cf.empty(h) } -> std::convertible_to<bool>;
		{ cf.left(h) } -> std::same_as<uint32>;
		{ cf.right(h) } -> std::same_as<uint32>;
		{ f.pop(h) } -> std::same_as<uint32>;
		{ Forest::empty_heap } -> std::convertible_to<uint32>;
	};

	namespace detail {
		template<typename Heap, typename Tree, typename Lists>
		concept class_stages = eppstein_class_heap<Heap> && eppstein_stages<Tree, Lists> && requires(Heap const &h, Lists const &lists, uint32 v) {
			{ h.push_range(lists.of(v)) } -> std::same_as<Heap>;
		};
		template<typename Forest, typename Tree, typename Lists>
		concept forest_stages = eppstein_forest_heap<Forest> && eppstein_stages<Tree, Lists> && requires(Forest &f, Lists const &lists, uint32 v) {
			{ f.push_range(v, lists.of(v)) } -> std::same_as<uint32>;
		};
	}

	namespace detail {
		template<std::copyable Where>
		struct eppstein_state {
			int64 length;
			Where at;
		};

		struct longer_state {
			template<std::copyable Where>
			[[nodiscard]] bool operator()(eppstein_state<Where> const &a, eppstein_state<Where> const &b) const {
				return a.length > b.length;
			}
		};

		/** @brief `H[v]` for every vertex as class-form versions: the parent's heap plus `v`'s own sidetracks. */
		template<eppstein_class_heap Heap, typename Tree, typename Lists>
			requires class_stages<Heap, Tree, Lists>
		[[nodiscard]] std::vector<Heap> class_sidetrack_heaps(std::size_t n, Tree const &tree, Lists const &lists, uint32 target) {
			Heap const none;
			std::vector<Heap> heaps(n, none);
			for (auto const &vertex : tree.order) {
				uint32 const v = uint32(vertex);
				heaps[v] = (v == target ? none : heaps[uint32(tree.next[v])]).push_range(lists.of(v));
			}
			return heaps;
		}

		template<eppstein_forest_heap Forest, typename Tree, typename Lists>
			requires forest_stages<Forest, Tree, Lists>
		std::vector<uint32> forest_sidetrack_heaps(Forest &forest, std::size_t n, Tree const &tree, Lists const &lists,
												   uint32 target) {
			std::vector<uint32> heaps(n, Forest::empty_heap);
			for (auto const &vertex : tree.order) {
				uint32 const v = uint32(vertex);
				heaps[v] = forest.push_range(v == target ? Forest::empty_heap : heaps[uint32(tree.next[v])], lists.of(v));
			}
			return heaps;
		}
	}

	/**
	 * @brief Enumerate walk lengths using heap-node walks on class-form Heap.
	 * @tparam Heap Persistent sidetrack heap satisfying the matching heap concept and push_range for Lists.
	 * @tparam Tree Target tree satisfying eppstein_stages with Lists.
	 * @tparam Lists Stable borrowed sidetrack ranges consumed during heap construction.
	 * @param n Number of vertices.
	 * @param tree Valid shortest-path tree for the graph and target.
	 * @param lists Sidetracks of that graph and tree.
	 * @param source Starting vertex, below n (asserted).
	 * @param target Destination vertex, below n (asserted).
	 * @param k Maximum number of answers.
	 * @return Nondecreasing owning lengths, retaining multiplicities; fewer if exhausted.
	 * @pre n < no_edge (asserted), tree arrays have n entries, all numeric
	 * conversions to int64/uint32 are exact, parent heaps precede children in order,
	 * and every list extra is nonnegative. Every candidate walk sum formed by the search must stay below
	 * INT64_MAX (asserted); pool counts fit the chosen heap's contract.
	 * @note Not constexpr (C++23 std::priority_queue). An isolated search-method
	 * comparison for this contender is not measured yet.
	 * @complexity Heap construction plus O(k log(k+2)) queue time;
	 * storage is constructed heaps plus O(k) states.
	 */
	template<eppstein_class_heap Heap, typename Tree, typename Lists>
		requires detail::class_stages<Heap, Tree, Lists>
	[[nodiscard]] std::vector<int64> k_shortest_walks_by_class_walk(std::size_t n, Tree const &tree, Lists const &lists, uint32 source,
																	uint32 target, std::size_t k) {
		atl_assert(n < no_edge && source < n && target < n);
		std::vector<int64> lengths;
		if (k == 0 || int64(tree.distance[source]) == (std::numeric_limits<int64>::max)()) return lengths;
		std::vector<Heap> const heaps = detail::class_sidetrack_heaps<Heap>(n, tree, lists, target);
		using view = typename Heap::node_view;
		using state = detail::eppstein_state<view>;
		std::priority_queue<state, std::vector<state>, detail::longer_state> queue;
		lengths.push_back(int64(tree.distance[source]));
		if (!heaps[source].empty()) queue.push({detail::walk_sum(int64(tree.distance[source]), int64(heaps[source].top().extra)), heaps[source].root()});
		while (lengths.size() < k && !queue.empty()) {
			state const s = queue.top();
			queue.pop();
			lengths.push_back(s.length);
			int64 const prefix = s.length - int64(s.at.value().extra);
			for (view child : {s.at.left(), s.at.right()})
				if (!child.empty()) queue.push({detail::walk_sum(prefix, int64(child.value().extra)), child});
			if (Heap const &next = heaps[uint32(s.at.value().head)]; !next.empty())
				queue.push({detail::walk_sum(s.length, int64(next.top().extra)), next.root()});
		}
		return lengths;
	}

	/**
	 * @brief Enumerate walk lengths using version pops on class-form Heap.
	 * @tparam Heap Persistent sidetrack heap satisfying the matching heap concept and push_range for Lists.
	 * @tparam Tree Target tree satisfying eppstein_stages with Lists.
	 * @tparam Lists Stable borrowed sidetrack ranges consumed during heap construction.
	 * @param n Number of vertices.
	 * @param tree Valid shortest-path tree for the graph and target.
	 * @param lists Sidetracks of that graph and tree.
	 * @param source Starting vertex, below n (asserted).
	 * @param target Destination vertex, below n (asserted).
	 * @param k Maximum number of answers.
	 * @return Nondecreasing owning lengths, retaining multiplicities; fewer if exhausted.
	 * @pre n < no_edge (asserted), tree arrays have n entries, all numeric
	 * conversions to int64/uint32 are exact, parent heaps precede children in order,
	 * and every list extra is nonnegative. Every candidate walk sum formed by the search must stay below
	 * INT64_MAX (asserted); pool counts fit the chosen heap's contract.
	 * @note Not constexpr (C++23 std::priority_queue). An isolated search-method
	 * comparison for this contender is not measured yet.
	 * @complexity Heap construction plus O(k log(k+2)) queue time and O(k log(m+2)) heap work;
	 * storage is constructed heaps plus O(k) states and newly copied heap nodes.
	 */
	template<eppstein_class_heap Heap, typename Tree, typename Lists>
		requires detail::class_stages<Heap, Tree, Lists>
	[[nodiscard]] std::vector<int64> k_shortest_walks_by_class_pop(std::size_t n, Tree const &tree, Lists const &lists, uint32 source,
																   uint32 target, std::size_t k) {
		atl_assert(n < no_edge && source < n && target < n);
		std::vector<int64> lengths;
		if (k == 0 || int64(tree.distance[source]) == (std::numeric_limits<int64>::max)()) return lengths;
		std::vector<Heap> const heaps = detail::class_sidetrack_heaps<Heap>(n, tree, lists, target);
		using state = detail::eppstein_state<Heap>;
		std::priority_queue<state, std::vector<state>, detail::longer_state> queue;
		lengths.push_back(int64(tree.distance[source]));
		if (!heaps[source].empty()) queue.push({detail::walk_sum(int64(tree.distance[source]), int64(heaps[source].top().extra)), heaps[source]});
		while (lengths.size() < k && !queue.empty()) {
			state s = queue.top();
			queue.pop();
			lengths.push_back(s.length);
			int64 const extra = int64(s.at.top().extra);
			uint32 const head = uint32(s.at.top().head);
			if (Heap const &next = heaps[head]; !next.empty()) queue.push({detail::walk_sum(s.length, int64(next.top().extra)), next});
			Heap rest = std::move(s.at).pop();
			if (!rest.empty()) {
				int64 const length = detail::walk_sum(s.length - extra, int64(rest.top().extra));
				queue.push({length, std::move(rest)});
			}
		}
		return lengths;
	}

	/**
	 * @brief Enumerate walk lengths using heap-node walks on forest-form Forest.
	 * @tparam Forest Persistent sidetrack heap satisfying the matching heap concept and push_range for Lists.
	 * @tparam Tree Target tree satisfying eppstein_stages with Lists.
	 * @tparam Lists Stable borrowed sidetrack ranges consumed during heap construction.
	 * @param n Number of vertices.
	 * @param tree Valid shortest-path tree for the graph and target.
	 * @param lists Sidetracks of that graph and tree.
	 * @param source Starting vertex, below n (asserted).
	 * @param target Destination vertex, below n (asserted).
	 * @param k Maximum number of answers.
	 * @return Nondecreasing owning lengths, retaining multiplicities; fewer if exhausted.
	 * @pre n < no_edge (asserted), tree arrays have n entries, all numeric
	 * conversions to int64/uint32 are exact, parent heaps precede children in order,
	 * and every list extra is nonnegative. Every candidate walk sum formed by the search must stay below
	 * INT64_MAX (asserted); pool counts fit the chosen heap's contract.
	 * @note Not constexpr (C++23 std::priority_queue). An isolated search-method
	 * comparison for this contender is not measured yet.
	 * @complexity Heap construction plus O(k log(k+2)) queue time;
	 * storage is constructed heaps plus O(k) states.
	 */
	template<eppstein_forest_heap Forest, typename Tree, typename Lists>
		requires detail::forest_stages<Forest, Tree, Lists>
	[[nodiscard]] std::vector<int64> k_shortest_walks_by_forest_walk(std::size_t n, Tree const &tree, Lists const &lists, uint32 source,
																	 uint32 target, std::size_t k) {
		atl_assert(n < no_edge && source < n && target < n);
		std::vector<int64> lengths;
		if (k == 0 || int64(tree.distance[source]) == (std::numeric_limits<int64>::max)()) return lengths;
		Forest forest;
		std::vector<uint32> const heaps = detail::forest_sidetrack_heaps(forest, n, tree, lists, target);
		using state = detail::eppstein_state<uint32>;
		std::priority_queue<state, std::vector<state>, detail::longer_state> queue;
		lengths.push_back(int64(tree.distance[source]));
		if (!forest.empty(heaps[source]))
			queue.push({detail::walk_sum(int64(tree.distance[source]), int64(forest.top(heaps[source]).extra)), heaps[source]});
		while (lengths.size() < k && !queue.empty()) {
			state const s = queue.top();
			queue.pop();
			lengths.push_back(s.length);
			auto const &top = forest.top(s.at);
			int64 const prefix = s.length - int64(top.extra);
			for (uint32 child : {forest.left(s.at), forest.right(s.at)})
				if (!forest.empty(child)) queue.push({detail::walk_sum(prefix, int64(forest.top(child).extra)), child});
			if (uint32 const next = heaps[uint32(top.head)]; !forest.empty(next))
				queue.push({detail::walk_sum(s.length, int64(forest.top(next).extra)), next});
		}
		return lengths;
	}

	/**
	 * @brief Enumerate walk lengths using version pops on forest-form Forest.
	 * @tparam Forest Persistent sidetrack heap satisfying the matching heap concept and push_range for Lists.
	 * @tparam Tree Target tree satisfying eppstein_stages with Lists.
	 * @tparam Lists Stable borrowed sidetrack ranges consumed during heap construction.
	 * @param n Number of vertices.
	 * @param tree Valid shortest-path tree for the graph and target.
	 * @param lists Sidetracks of that graph and tree.
	 * @param source Starting vertex, below n (asserted).
	 * @param target Destination vertex, below n (asserted).
	 * @param k Maximum number of answers.
	 * @return Nondecreasing owning lengths, retaining multiplicities; fewer if exhausted.
	 * @pre n < no_edge (asserted), tree arrays have n entries, all numeric
	 * conversions to int64/uint32 are exact, parent heaps precede children in order,
	 * and every list extra is nonnegative. Every candidate walk sum formed by the search must stay below
	 * INT64_MAX (asserted); pool counts fit the chosen heap's contract.
	 * @note Not constexpr (C++23 std::priority_queue). An isolated search-method
	 * comparison for this contender is not measured yet.
	 * @complexity Heap construction plus O(k log(k+2)) queue time and O(k log(m+2)) heap work;
	 * storage is constructed heaps plus O(k) states and newly copied heap nodes.
	 */
	template<eppstein_forest_heap Forest, typename Tree, typename Lists>
		requires detail::forest_stages<Forest, Tree, Lists>
	[[nodiscard]] std::vector<int64> k_shortest_walks_by_forest_pop(std::size_t n, Tree const &tree, Lists const &lists, uint32 source,
																	uint32 target, std::size_t k) {
		atl_assert(n < no_edge && source < n && target < n);
		std::vector<int64> lengths;
		if (k == 0 || int64(tree.distance[source]) == (std::numeric_limits<int64>::max)()) return lengths;
		Forest forest;
		std::vector<uint32> const heaps = detail::forest_sidetrack_heaps(forest, n, tree, lists, target);
		using state = detail::eppstein_state<uint32>;
		std::priority_queue<state, std::vector<state>, detail::longer_state> queue;
		lengths.push_back(int64(tree.distance[source]));
		if (!forest.empty(heaps[source]))
			queue.push({detail::walk_sum(int64(tree.distance[source]), int64(forest.top(heaps[source]).extra)), heaps[source]});
		while (lengths.size() < k && !queue.empty()) {
			state const s = queue.top();
			queue.pop();
			lengths.push_back(s.length);
			int64 const extra = int64(forest.top(s.at).extra);
			uint32 const head = uint32(forest.top(s.at).head);
			if (uint32 const next = heaps[head]; !forest.empty(next))
				queue.push({detail::walk_sum(s.length, int64(forest.top(next).extra)), next});
			if (uint32 const rest = forest.pop(s.at); !forest.empty(rest))
				queue.push({detail::walk_sum(s.length - extra, int64(forest.top(rest).extra)), rest});
		}
		return lengths;
	}
}
