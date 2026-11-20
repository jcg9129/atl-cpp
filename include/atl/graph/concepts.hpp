#pragma once
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <limits>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

namespace atl::graph {
	/**
	 * @brief Canonical adjacency element: a target vertex plus an optional
	 *        weight. `edge<>` is the unweighted form (just the target).
	 *
	 * Any user type with the same member names models the same concepts; these
	 * exist so there is rarely a reason to define one.
	 * @tparam Weight Stored weight type, or `void` for an unweighted element.
	 */
	template<typename Weight = void>
	struct edge {
		vertex_id to;  ///< Target vertex.
		Weight weight; ///< Edge weight.
	};

	/** @brief Unweighted adjacency element. */
	template<>
	struct edge<void> {
		vertex_id to; ///< Target vertex.
	};

	/**
	 * @brief Canonical edge-list element: both endpoints plus an optional
	 *        weight. The flat, input-friendly representation.
	 * @tparam Weight Stored weight type, or `void` to omit it.
	 */
	template<typename Weight = void>
	struct arc {
		vertex_id from; ///< Source vertex.
		vertex_id to;	///< Target vertex.
		Weight weight;	///< Edge weight.
	};

	/** @brief Unweighted edge-list element. */
	template<>
	struct arc<void> {
		vertex_id from; ///< Source vertex.
		vertex_id to;	///< Target vertex.
	};

	/**
	 * @brief Canonical flow-network edge: both endpoints plus a capacity,
	 *        and optionally a per-unit cost (`flow_arc<Capacity, Cost>`) for the
	 *        min-cost algorithms. Cost is its own field and accessor —
	 *        deliberately not `weight`, so path weights and flow costs stay
	 *        distinct vocabularies.
	 * @tparam Capacity Upper-bound value type.
	 * @tparam Cost Per-unit cost type, or `void` to omit costs.
	 */
	template<typename Capacity, typename Cost = void>
	struct flow_arc {
		vertex_id from;	   ///< Source vertex.
		vertex_id to;	   ///< Target vertex.
		Capacity capacity; ///< Upper capacity.
		Cost cost;		   ///< Per-unit cost.
	};

	/** @brief Capacity-only edge-list element. */
	template<typename Capacity>
	struct flow_arc<Capacity, void> {
		vertex_id from;	   ///< Source vertex.
		vertex_id to;	   ///< Target vertex.
		Capacity capacity; ///< Upper capacity.
	};

	/**
	 * @brief Canonical edge of the general b-flow problem: flow bounded to
	 *        `[lower, capacity]`, optionally costed.
	 * @tparam Capacity Lower/upper-bound value type.
	 * @tparam Cost Per-unit cost type, or `void` to omit costs.
	 */
	template<typename Capacity, typename Cost = void>
	struct bounded_flow_arc {
		vertex_id from;	   ///< Source vertex.
		vertex_id to;	   ///< Target vertex.
		Capacity lower;	   ///< Lower capacity.
		Capacity capacity; ///< Upper capacity.
		Cost cost;		   ///< Per-unit cost.
	};

	/** @brief Edge-list element with lower and upper capacities. */
	template<typename Capacity>
	struct bounded_flow_arc<Capacity, void> {
		vertex_id from;	   ///< Source vertex.
		vertex_id to;	   ///< Target vertex.
		Capacity lower;	   ///< Lower capacity.
		Capacity capacity; ///< Upper capacity.
	};

	/**
	 * @brief Customization point for edge types that cannot follow the member
	 *        conventions (`.to` / `.from` / `.weight`): specialize and provide
	 *        the matching static accessors (`target(e)`, and `source(e)` /
	 *        `weight(e)` where applicable); capacity, lower and cost follow the same rule. A specialization overrides member
	 *        detection — explicit beats convention.
	 * @tparam Edge Adapted element type.
	 */
	template<typename Edge>
	struct graph_traits {};

	namespace detail {
		template<typename T>
		concept character_type =
			std::same_as<std::remove_cv_t<T>, char> || std::same_as<std::remove_cv_t<T>, signed char>
			|| std::same_as<std::remove_cv_t<T>, unsigned char> || std::same_as<std::remove_cv_t<T>, wchar_t>
			|| std::same_as<std::remove_cv_t<T>, char8_t> || std::same_as<std::remove_cv_t<T>, char16_t>
			|| std::same_as<std::remove_cv_t<T>, char32_t>;

		/**
		 * @brief An integral type that can name a vertex: any integral type
		 *        except `bool` and the character types, so a `vector<string>`
		 *        grid or a `vector<vector<bool>>` matrix is not mistaken for an
		 *        adjacency structure.
		 */
		template<typename T>
		concept vertex_integral =
			std::integral<T> && !std::same_as<std::remove_cv_t<T>, bool> && !character_type<T>;

		/**
		 * @brief A member endpoint type that can name a vertex: a
		 *        `vertex_integral`, or a non-arithmetic type converting to
		 *        `vertex_id` (a strong id wrapper). Floating, `bool` and
		 *        character members are rejected.
		 */
		template<typename T>
		concept vertex_member =
			(vertex_integral<std::remove_cvref_t<T>> || !std::is_arithmetic_v<std::remove_cvref_t<T>>)
			&& std::convertible_to<T, vertex_id>;

		template<typename Edge>
		concept traits_target = requires(Edge const &e) {
			{ graph_traits<Edge>::target(e) } -> std::convertible_to<vertex_id>;
		};

		template<typename Edge>
		concept member_target = requires(Edge const &e) {
			requires vertex_member<decltype((e.to))>;
		};

		template<typename Edge>
		concept traits_source = requires(Edge const &e) {
			{ graph_traits<Edge>::source(e) } -> std::convertible_to<vertex_id>;
		};

		template<typename Edge>
		concept member_source = requires(Edge const &e) {
			requires vertex_member<decltype((e.from))>;
		};

		template<typename Edge>
		concept traits_weight = requires(Edge const &e) { graph_traits<Edge>::weight(e); };

		template<typename Edge>
		concept member_weight = requires(Edge const &e) { e.weight; };

		template<typename Edge>
		concept traits_capacity = requires(Edge const &e) { graph_traits<Edge>::capacity(e); };

		template<typename Edge>
		concept member_capacity = requires(Edge const &e) { e.capacity; };

		template<typename Edge>
		concept traits_lower = requires(Edge const &e) { graph_traits<Edge>::lower(e); };

		template<typename Edge>
		concept member_lower = requires(Edge const &e) { e.lower; };

		template<typename Edge>
		concept traits_cost = requires(Edge const &e) { graph_traits<Edge>::cost(e); };

		template<typename Edge>
		concept member_cost = requires(Edge const &e) { e.cost; };

		/**
		 * @brief Whether @p id fits the `vertex_id` representation, including `none`:
		 *        nonnegative and at most the largest `vertex_id`. Compared in
		 *        64 bits to avoid narrowing during the check.
		 */
		template<std::integral Id>
		[[nodiscard]] constexpr bool fits_vertex_id(Id id) noexcept {
			if constexpr (std::is_signed_v<Id>)
				if (id < 0) return false;
			return uint64(id) <= (std::numeric_limits<vertex_id>::max)();
		}
	}

	/**
	 * @brief Target vertex of an edge: the value itself for integral edges, `.to` by
	 * convention, or `graph_traits<Edge>::target`. An integral id outside
	 * `[0, 2^32)` violates an asserted precondition. `none` is representable
	 * but cannot name a vertex. Traits returning integral ids receive the same check.
	 * @pre Integral endpoints fit `vertex_id`; other endpoint conversions preserve their value.
	 * @return The endpoint, converted to `vertex_id`. @complexity O(1) plus trait/conversion costs.
	 */
	template<typename Edge>
		requires(detail::vertex_integral<Edge> || detail::traits_target<Edge> || detail::member_target<Edge>)
	[[nodiscard]] constexpr vertex_id target(Edge const &e) {
		if constexpr (detail::vertex_integral<Edge>) {
			atl_assert(detail::fits_vertex_id(e));
			return vertex_id(e);
		} else if constexpr (detail::traits_target<Edge>) {
			decltype(auto) id = graph_traits<Edge>::target(e);
			if constexpr (std::integral<std::remove_cvref_t<decltype(id)>>) atl_assert(detail::fits_vertex_id(id));
			return vertex_id(std::forward<decltype(id)>(id));
		} else {
			if constexpr (std::integral<std::remove_cvref_t<decltype(e.to)>>) atl_assert(detail::fits_vertex_id(e.to));
			return vertex_id(e.to);
		}
	}

	/**
	 * @brief Source vertex of an edge-list element: `.from` or `graph_traits`. An
	 * integral endpoint outside `[0, 2^32)` violates an asserted precondition.
	 * @pre Integral endpoints fit `vertex_id`; other endpoint conversions preserve their value.
	 * @return The source, converted to `vertex_id`. @complexity O(1) plus trait/conversion costs.
	 */
	template<typename Edge>
		requires(detail::traits_source<Edge> || detail::member_source<Edge>)
	[[nodiscard]] constexpr vertex_id source(Edge const &e) {
		if constexpr (detail::traits_source<Edge>) {
			decltype(auto) id = graph_traits<Edge>::source(e);
			if constexpr (std::integral<std::remove_cvref_t<decltype(id)>>) atl_assert(detail::fits_vertex_id(id));
			return vertex_id(std::forward<decltype(id)>(id));
		} else {
			if constexpr (std::integral<std::remove_cvref_t<decltype(e.from)>>)
				atl_assert(detail::fits_vertex_id(e.from));
			return vertex_id(e.from);
		}
	}

	/**
	 * @brief Weight of an edge: `.weight` or `graph_traits`. Integral edges are
	 * unweighted — algorithms that want unit weights say so explicitly.
	 */
	template<typename Edge>
		requires(detail::traits_weight<Edge> || detail::member_weight<Edge>)
	[[nodiscard]] constexpr decltype(auto) weight(Edge const &e) {
		if constexpr (detail::traits_weight<Edge>)
			return graph_traits<Edge>::weight(e);
		else
			return (e.weight);
	}

	/** @brief Capacity of a flow-network edge: `.capacity` or `graph_traits`. */
	template<typename Edge>
		requires(detail::traits_capacity<Edge> || detail::member_capacity<Edge>)
	[[nodiscard]] constexpr decltype(auto) capacity(Edge const &e) {
		if constexpr (detail::traits_capacity<Edge>)
			return graph_traits<Edge>::capacity(e);
		else
			return (e.capacity);
	}

	/**
	 * @brief Lower bound of a flow-network edge: `.lower` or `graph_traits`. An
	 * edge type without one has lower bound 0 *structurally* — algorithms
	 * that cannot honor lower bounds reject bounded edge types at compile
	 * time instead of ignoring the field.
	 */
	template<typename Edge>
		requires(detail::traits_lower<Edge> || detail::member_lower<Edge>)
	[[nodiscard]] constexpr decltype(auto) lower(Edge const &e) {
		if constexpr (detail::traits_lower<Edge>)
			return graph_traits<Edge>::lower(e);
		else
			return (e.lower);
	}

	/**
	 * @brief Per-unit cost of a flow-network edge: `.cost` or `graph_traits` —
	 * deliberately distinct from `weight` (path weights and flow costs are
	 * different vocabularies).
	 */
	template<typename Edge>
		requires(detail::traits_cost<Edge> || detail::member_cost<Edge>)
	[[nodiscard]] constexpr decltype(auto) cost(Edge const &e) {
		if constexpr (detail::traits_cost<Edge>)
			return graph_traits<Edge>::cost(e);
		else
			return (e.cost);
	}

	/** @brief An element usable inside an adjacency range: `target(e)` is valid. */
	template<typename Edge>
	concept adjacency_edge = requires(Edge const &e) { graph::target(e); };

	/** @brief An adjacency element that also carries a weight. */
	template<typename Edge>
	concept weighted_adjacency_edge = adjacency_edge<Edge> && requires(Edge const &e) { graph::weight(e); };

	/** @brief An element usable in an edge list: both endpoints are extractable. */
	template<typename Edge>
	concept list_edge = requires(Edge const &e) {
		graph::source(e);
		graph::target(e);
	};

	/** @brief An edge-list element that also carries a weight. */
	template<typename Edge>
	concept weighted_list_edge = list_edge<Edge> && requires(Edge const &e) { graph::weight(e); };

	namespace detail {
		template<typename Graph>
		concept member_graph = requires(Graph const &g, vertex_id u) {
			{ g.vertex_count() } -> std::convertible_to<std::size_t>;
			{ g.out_edges(u) } -> std::ranges::input_range;
		};

		/**
		 * @brief Default tier: any sized random-access range of edge ranges —
		 * `vector<vector<Edge>>` and friends, unwrapped.
		 */
		template<typename Graph>
		concept range_graph = std::ranges::random_access_range<Graph const> && std::ranges::sized_range<Graph const>
						   && std::ranges::input_range<std::ranges::range_reference_t<Graph const>>;
	}

	/** @brief Number of vertices: `g.vertex_count()` if provided, else the range size. */
	template<typename Graph>
		requires(detail::member_graph<Graph> || detail::range_graph<Graph>)
	[[nodiscard]] constexpr std::size_t vertex_count(Graph const &g) {
		if constexpr (detail::member_graph<Graph>)
			return g.vertex_count();
		else
			return std::ranges::size(g);
	}

	/** @brief Out-edges of vertex @p u: `g.out_edges(u)` if provided, else `g[u]`.
	 * @pre `u < vertex_count(g)`. Repeated calls give the same adjacency order;
	 * reference-returning ranges remain valid while algorithms scan the graph.
	 * @return The member result or a reference to the indexed row.
	 */
	template<typename Graph>
		requires(detail::member_graph<Graph> || detail::range_graph<Graph>)
	[[nodiscard]] constexpr decltype(auto) out_edges(Graph const &g, vertex_id u) {
		if constexpr (detail::member_graph<Graph>)
			return g.out_edges(u);
		else
			return std::ranges::begin(g)[u];
	}

	/** @brief The out-edge range type of graph @p Graph (cv-ref stripped). */
	template<typename Graph>
	using out_edges_t = std::remove_cvref_t<decltype(graph::out_edges(std::declval<Graph const &>(), vertex_id{}))>;

	/** @brief The adjacency element type of graph @p Graph. */
	template<typename Graph>
	using edge_t = std::ranges::range_value_t<out_edges_t<Graph>>;

	/** @brief The weight type of edge element @p Edge (adjacency or list). */
	template<typename Edge>
	using edge_weight_t = std::remove_cvref_t<decltype(graph::weight(std::declval<Edge const &>()))>;

	/** @brief The capacity type of flow-network edge @p Edge. */
	template<typename Edge>
	using edge_capacity_t = std::remove_cvref_t<decltype(graph::capacity(std::declval<Edge const &>()))>;

	/** @brief The per-unit cost type of costed flow-network edge @p Edge. */
	template<typename Edge>
	using edge_cost_t = std::remove_cvref_t<decltype(graph::cost(std::declval<Edge const &>()))>;

	/** @brief The weight type of graph @p Graph's adjacency elements. */
	template<typename Graph>
	using weight_t = edge_weight_t<edge_t<Graph>>;

	/**
	 * @brief The concept every vertex-centric algorithm requires: integral
	 *        vertices `[0, n)`, an out-edge range per vertex, targets
	 *        extractable from its elements. Modeled without wrapping by
	 *        `vector<vector<int>>`, `vector<vector<edge<W>>>`, @ref atl::graph::csr, and
	 *        @ref atl::graph::implicit_graph.
	 */
	template<typename Graph>
	concept adjacency_graph = requires(Graph const &g, vertex_id u) {
		{ graph::vertex_count(g) } -> std::convertible_to<std::size_t>;
		{ graph::out_edges(g, u) } -> std::ranges::input_range;
		requires adjacency_edge<std::ranges::range_value_t<std::remove_cvref_t<decltype(graph::out_edges(g, u))>>>;
		requires adjacency_edge<std::remove_cvref_t<std::ranges::range_reference_t<decltype(graph::out_edges(g, u))>>>;
	};

	/** @brief An @ref atl::graph::adjacency_graph whose adjacency elements carry weights. */
	template<typename Graph>
	concept weighted_adjacency_graph =
		adjacency_graph<Graph>
		&& weighted_adjacency_edge<std::ranges::range_value_t<out_edges_t<Graph>>>
		&& weighted_adjacency_edge<std::remove_cvref_t<std::ranges::range_reference_t<
			decltype(graph::out_edges(std::declval<Graph const &>(), vertex_id{}))>>>;

	/**
	 * @brief A flat range of edges with both endpoints — the natural input
	 *        format, and enough on its own for Kruskal / Bellman-Ford / flow.
	 *
	 * Multipass and const-iterable (`forward_range<EdgeList const>`): a
	 * consumer may walk the list twice (count, then fill), so a
	 * single-pass range is rejected here rather than silently read once.
	 */
	template<typename EdgeList>
	concept edge_list =
		std::ranges::forward_range<EdgeList const> && list_edge<std::ranges::range_value_t<EdgeList>>
		&& list_edge<std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>>;

	/**
	 * @brief A batch of vertex pairs (LCA queries and friends): a sized forward
	 * range of tuple-like two-vertex elements — `std::pair`, two-element
	 * tuples, `std::array<vertex_id, 2>`. Tuple-likeness is what the
	 * structured bindings consuming these require.
	 */
	template<typename R>
	concept vertex_pair_range =
		std::ranges::forward_range<R> && std::ranges::sized_range<R>
		&& requires(std::ranges::range_value_t<R> const &q) {
			   // the typename requirement guards the arity check: tuple_size
			   // of a non-tuple is incomplete, and instantiating tuple_size_v
			   // on it is a hard error, not a substitution failure (the same
			   // shape the standard's pair-like exposition concept uses)
			   typename std::tuple_size<std::ranges::range_value_t<R>>::type;
			   requires std::tuple_size_v<std::ranges::range_value_t<R>> == 2;
			   { std::get<0>(q) } -> std::convertible_to<vertex_id>;
			   { std::get<1>(q) } -> std::convertible_to<vertex_id>;
		   };

	/** @brief An @ref atl::graph::edge_list whose elements carry weights. */
	template<typename EdgeList>
	concept weighted_edge_list = edge_list<EdgeList> && weighted_list_edge<std::ranges::range_value_t<EdgeList>>
							  && weighted_list_edge<std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>>;

	/** @brief A flow-network edge: both endpoints plus a capacity. */
	template<typename Edge>
	concept capacitated_edge = list_edge<Edge> && requires(Edge const &e) { graph::capacity(e); };

	/** @brief A flow-network edge that additionally carries a lower bound. */
	template<typename Edge>
	concept bounded_capacitated_edge = capacitated_edge<Edge> && requires(Edge const &e) { graph::lower(e); };

	/** @brief A flow-network edge that additionally carries a per-unit cost. */
	template<typename Edge>
	concept cost_capacitated_edge = capacitated_edge<Edge> && requires(Edge const &e) { graph::cost(e); };

	/** @brief An @ref atl::graph::edge_list of @ref atl::graph::capacitated_edge elements: a flow network. */
	template<typename EdgeList>
	concept capacitated_edge_list = edge_list<EdgeList> && capacitated_edge<std::ranges::range_value_t<EdgeList>>
								 && capacitated_edge<std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>>;

	/** @brief An @ref atl::graph::edge_list of @ref atl::graph::cost_capacitated_edge elements: a costed flow network. */
	template<typename EdgeList>
	concept cost_capacitated_edge_list =
		edge_list<EdgeList> && cost_capacitated_edge<std::ranges::range_value_t<EdgeList>>
		&& cost_capacitated_edge<std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>>;

	/** @brief The capacity type of a capacitated edge list. */
	template<typename EdgeList>
	using capacity_t = edge_capacity_t<std::ranges::range_value_t<EdgeList>>;

	/** @brief The cost type of a cost-capacitated edge list. */
	template<typename EdgeList>
	using cost_t = edge_cost_t<std::ranges::range_value_t<EdgeList>>;

	/**
	 * @brief A flow or capacity value type as the flow algorithms use it:
	 *        totally ordered, additive, and constructible from `0`. Floating
	 *        capacities are admitted; algebraic laws and overflow bounds are
	 *        documented by the consuming algorithm.
	 */
	template<typename Flow>
	concept flow_value = std::totally_ordered<Flow> && std::constructible_from<Flow, int>
					  && requires(Flow &a, Flow const &b) {
							 { a + b } -> std::convertible_to<Flow>;
							 { a - b } -> std::convertible_to<Flow>;
							 a += b;
							 a -= b;
						 };

	/**
	 * @brief A @ref atl::graph::flow_value that is also signed: what algorithms negate
	 *        need (the min-cost simplex's `Flow` and `Cost`, supplies and
	 *        lower-bound reductions), so an unsigned type cannot wrap silently.
	 *        Read from `numeric_limits`, so `int128` qualifies even in strict
	 *        ISO mode, where `std::is_signed` rejects it.
	 */
	template<typename Flow>
	concept signed_flow_value = flow_value<Flow> && std::numeric_limits<Flow>::is_signed
							 && requires(Flow const &value) { { -value } -> std::convertible_to<Flow>; };

	/**
	 * @brief Adjacency graph whose edges are computed on demand: `neighbors(u)`
	 *        returns any range of adjacency edges (a `static_vector` of grid
	 *        moves, a generated span, ...). The graph is never materialized.
	 * @tparam Neighbors Const-callable row generator accepting `vertex_id`.
	 */
	template<typename Neighbors>
		requires requires(Neighbors const &f, vertex_id u) {
			{ f(u) } -> std::ranges::input_range;
			requires adjacency_edge<std::ranges::range_value_t<std::invoke_result_t<Neighbors const &, vertex_id>>>;
		}
	class implicit_graph {
		std::size_t n_;
		[[no_unique_address]] Neighbors neighbors_;

	public:
		/**
		 * @brief The graph on vertices `[0, n)` whose out-edges of `u` are
		 *        `neighbors(u)`.
		 * @param n Number of vertices.
		 * @param neighbors Row generator, moved into the graph.
		 * @pre `n <= 2^32 - 1`: every vertex must be a `vertex_id` (asserted).
		 */
		constexpr implicit_graph(std::size_t n, Neighbors neighbors)
			requires std::constructible_from<Neighbors, Neighbors &&>
			: n_(n), neighbors_(std::move(neighbors)) {
			atl_assert(n <= (std::numeric_limits<vertex_id>::max)());
		}

		/** @brief Number of vertices in the generated domain. @complexity O(1) */
		[[nodiscard]] constexpr std::size_t vertex_count() const noexcept { return n_; }

		/** @brief Generate the row of @p u. @pre `u < vertex_count()` (asserted).
		 * @return `neighbors(u)`; cost and lifetime follow the neighbor callable.
		 */
		[[nodiscard]] constexpr decltype(auto) out_edges(vertex_id u) const {
			atl_assert(u < n_);
			return neighbors_(u);
		}
	};
}
