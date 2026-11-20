#include <atl/experimental/graph/biconnected_components_variants.hpp>
#include <atl/experimental/graph/centroid_bisect_variants.hpp>
#include <atl/experimental/graph/centroid_variants.hpp>
#include <atl/experimental/graph/csr_variants.hpp>
#include <atl/experimental/graph/deletion_time_connectivity.hpp>
#include <atl/experimental/graph/lca_variants.hpp>
#include <atl/experimental/graph/scc_variants.hpp>
#include <atl/experimental/graph/two_edge_connected_components_variants.hpp>
#include <atl/experimental/graph/virtual_tree_variants.hpp>
#include <atl/graph/biconnected_components.hpp>
#include <atl/graph/centroid_bisect.hpp>
#include <atl/graph/centroid_decomposition.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/lca.hpp>
#include <atl/graph/scc.hpp>
#include <atl/graph/ternarization.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/graph/two_edge_connected_components.hpp>
#include <atl/graph/virtual_tree.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <compare>
#include <concepts>
#include <cstddef>
#include <forward_list>
#include <iterator>
#include <limits>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace graph_boundary {
	struct no_negation {
		int value;

		no_negation(int initial) : value(initial) {}

		auto operator<=>(no_negation const &) const = default;

		no_negation operator+(no_negation other) const { return value + other.value; }

		no_negation operator-(no_negation other) const { return value - other.value; }

		no_negation &operator+=(no_negation other) {
			value += other.value;
			return *this;
		}

		no_negation &operator-=(no_negation other) {
			value -= other.value;
			return *this;
		}
	};

	template<std::default_initializable Value>
	struct endpoint_proxy {
		atl::graph::vertex_id from{}, to{};

		operator Value() const { return {}; }
	};

	template<std::default_initializable Value>
	struct endpoint_proxy_range {
		struct iterator {
			using value_type = Value;
			using difference_type = std::ptrdiff_t;
			using iterator_concept = std::forward_iterator_tag;
			int index{};

			endpoint_proxy<Value> operator*() const { return {}; }

			iterator &operator++() {
				++index;
				return *this;
			}

			iterator operator++(int) {
				auto saved = *this;
				++*this;
				return saved;
			}

			bool operator==(iterator const &) const = default;
		};

		iterator begin() const { return {}; }

		iterator end() const { return {}; }
	};

}

template<>
struct std::numeric_limits<graph_boundary::no_negation> : std::numeric_limits<int> {};

namespace {
	namespace g = atl::graph;
	using graph_rows = std::vector<std::vector<int>>;
	using graph_arcs = std::vector<g::arc<>>;

	struct mutable_outer {
		graph_rows rows;

		auto begin() { return rows.begin(); }

		auto end() { return rows.end(); }

		auto size() const { return rows.size(); }
	};

	struct mutable_row_graph {
		std::size_t vertex_count() const;

		auto out_edges(g::vertex_id) const {
			return std::views::iota(0, 3) | std::views::filter([](int) { return true; });
		}
	};

	struct no_default_edge {
		g::vertex_id to;
		no_default_edge() = delete;

		explicit no_default_edge(g::vertex_id v) : to(v) {}
	};

	template<typename Graph>
	concept has_shape_accessors = requires(Graph const &rows) { g::vertex_count(rows); g::out_edges(rows, 0); };
	template<typename Graph>
	concept has_traversal_views = requires(Graph const &rows) { g::bfs(rows, 0); g::dfs(rows, 0); };
	template<typename Step>
	concept accepts_bfs_step = requires(graph_rows const &rows, Step step) { g::bfs_visit(rows, 0, step); };
	template<typename Event>
	concept accepts_events = requires(Event event) { g::events(event); };
	template<typename Init, typename Fold, typename Finalize>
	concept accepts_fold = requires(graph_rows const &rows, Init init, Fold fold, Finalize finalize) {
		g::dfs_fold(rows, 0, init, fold);
		g::dfs_fold(rows, 0, init, fold, finalize);
	};
	template<typename R>
	concept accepts_bucket_csr = requires(R const &arcs) { atl::experimental::csr_by_bucket_gather<g::vertex_id, false>(3, arcs); };
	template<typename R>
	concept accepts_undirected_csr = requires(R const &arcs) { g::undirected_csr(3, arcs); g::csr<g::vertex_id>::undirected(3, arcs); };

	template<typename Sources>
	concept accepts_source_range = requires(graph_rows const &rows, Sources const &sources, g::epoch_marker &marker) {
		g::bfs(rows, sources);
		g::bfs(rows, sources, marker);
		g::bfs_visit(rows, sources, g::skip);
		g::bfs_visit(rows, sources, marker, g::skip);
	};
	template<typename Neighbors>
	concept accepts_generated_graph = requires { typename g::implicit_graph<Neighbors>; };
	template<typename Edge>
	concept accepts_default_conversion = requires(graph_arcs const &arcs) {
		g::csr<Edge>::undirected(3, arcs);
		atl::experimental::csr_by_cursor_copy<Edge, false>(3, arcs);
		atl::experimental::csr_by_bucket_gather<Edge, false>(3, arcs);
	};

	struct non_range_neighbors {
		int operator()(g::vertex_id) const;
	};

	struct immovable_exit {
		immovable_exit() = default;
		immovable_exit(immovable_exit &&) = delete;
		void operator()(g::vertex_id, g::vertex_id) const;
	};

	struct immovable_neighbors {
		immovable_neighbors() = default;
		immovable_neighbors(immovable_neighbors &&) = delete;

		std::array<g::vertex_id, 1> operator()(g::vertex_id) const { return {0}; }
	};

	template<typename Neighbors>
	concept accepts_generator_storage = requires { g::implicit_graph(3, Neighbors{}); };

	struct immovable_enter {
		immovable_enter() = default;
		immovable_enter(immovable_enter &&) = delete;
		void operator()(g::vertex_id, g::vertex_id, atl::uint32, atl::uint32) const;
	};

	template<typename Enter>
	concept accepts_forwarded_enter = requires(graph_rows const &rows) {
		g::dfs_visit(rows, 0, Enter{}, g::skip, g::skip);
		g::dfs_visit(rows, Enter{}, g::skip, g::skip);
	};

	template<typename Enter>
	constexpr bool rejects_bundle_overloads = [] {
		using Skip = std::remove_cvref_t<decltype(g::skip)>;
		using Bundle = g::event_set<Enter, Skip, Skip>;
		return !requires(graph_rows const &rows) { g::dfs_visit(rows, 0, Bundle{}); }
			&& !requires(graph_rows const &rows) { g::dfs_visit(rows, Bundle{}); }
			&& !requires(graph_rows const &rows, g::epoch_marker &marker) { g::dfs_visit(rows, 0, marker, Bundle{}); }
			&& !requires(graph_rows const &rows, g::epoch_marker &marker) { g::dfs_visit(rows, marker, Bundle{}); };
	}();

	template<typename Event>
	concept accepts_event_lvalue = requires(Event &event) { g::events(event); };

	struct wrong_callback {
		void operator()(int *) const;
	};

	struct void_init {
		void operator()(g::visit) const;
	};

	struct reference_init {
		int &operator()(g::visit) const;
	};

	struct int_init {
		int operator()(g::visit) const { return 0; }
	};

	struct valid_fold {
		void operator()(int &, int, g::visit, g::visit, int) const {}
	};

	struct valid_finalize {
		void operator()(int &, g::visit) const {}
	};

	struct ambiguous_event {
		template<typename... Args>
		void operator()(Args &&...) const;
	};

	template<typename R>
	concept accepts_deduced_undirected = requires(R const &arcs) { g::undirected_csr(3, arcs); };

	using weighted_proxy_list = graph_boundary::endpoint_proxy_range<g::arc<int>>;
	using flow_proxy_list = graph_boundary::endpoint_proxy_range<g::flow_arc<int, int>>;
	static_assert(g::edge_list<weighted_proxy_list>);
	static_assert(!g::weighted_edge_list<weighted_proxy_list>);
	static_assert(!accepts_deduced_undirected<weighted_proxy_list>);
	static_assert(g::adjacency_graph<std::vector<weighted_proxy_list>>);
	static_assert(!g::weighted_adjacency_graph<std::vector<weighted_proxy_list>>);
	static_assert(!g::capacitated_edge_list<flow_proxy_list>);
	static_assert(!g::cost_capacitated_edge_list<flow_proxy_list>);
	static_assert(!std::constructible_from<g::csr<g::edge<int>>, std::size_t, weighted_proxy_list>);
	static_assert(!accepts_undirected_csr<graph_boundary::endpoint_proxy_range<g::edge<int>>>);

	static_assert(g::flow_value<graph_boundary::no_negation>);
	static_assert(!g::signed_flow_value<graph_boundary::no_negation>);
	static_assert(!accepts_source_range<std::vector<std::string>>);
	static_assert(!accepts_generated_graph<non_range_neighbors>);
	static_assert(!accepts_generator_storage<immovable_neighbors>);
	static_assert(!accepts_default_conversion<no_default_edge>);
	static_assert(!accepts_default_conversion<g::edge<int>>);
	static_assert(!accepts_event_lvalue<immovable_exit>);
	static_assert(!accepts_forwarded_enter<immovable_enter>);
	static_assert(rejects_bundle_overloads<immovable_enter>);
	static_assert(!has_shape_accessors<mutable_outer>);
	static_assert(!g::adjacency_graph<mutable_outer>);
	static_assert(!has_traversal_views<mutable_outer>);
	static_assert(!has_traversal_views<mutable_row_graph>);
	static_assert(!g::adjacency_graph<std::vector<std::string>>);
	static_assert(!g::adjacency_graph<std::vector<std::vector<bool>>>);
	static_assert(!g::adjacency_edge<bool> && !g::adjacency_edge<char>);
	static_assert(!g::list_edge<int>);
	static_assert(!std::constructible_from<g::csr<no_default_edge>, std::size_t, graph_arcs>);
	static_assert(!std::constructible_from<g::csr<g::edge<int>>, std::size_t, graph_arcs>);
	static_assert(!std::constructible_from<g::csr<g::vertex_id>, std::size_t, graph_arcs, wrong_callback>);
	static_assert(!accepts_undirected_csr<std::vector<int>>);
	static_assert(!accepts_bucket_csr<std::forward_list<g::arc<>>>);
	static_assert(!accepts_bfs_step<wrong_callback>);
	static_assert(!accepts_events<wrong_callback> && !accepts_events<ambiguous_event>);
	static_assert(!accepts_fold<void_init, valid_fold, valid_finalize>);
	static_assert(!accepts_fold<reference_init, valid_fold, valid_finalize>);
	static_assert(!accepts_fold<int_init, wrong_callback, valid_finalize>);
	static_assert(!accepts_fold<int_init, valid_fold, wrong_callback>);
	static_assert(!g::dfs_fold_callbacks<int, int_init, valid_fold, valid_finalize>);
}

TEST_CASE("graph constraints reject unsupported shapes and callbacks", "[graph][constraints]") {
	SUCCEED();
}

namespace graph_components_boundary {
	template<typename Graph>
	concept accepts_scc = requires(Graph const &rows) { atl::graph::scc(rows); };
	template<typename Graph>
	concept accepts_bridges = requires(Graph const &rows) { atl::graph::two_edge_connected_components(rows); };
	template<typename Graph>
	concept accepts_blocks = requires(Graph const &rows) { atl::graph::biconnected_components(rows); };
	template<typename Graph>
	concept accepts_event_scc = requires(Graph const &rows) { atl::experimental::scc_by_tarjan_events(rows); };
	template<typename Graph>
	concept accepts_paired_blocks = requires(Graph const &rows) { atl::experimental::biconnected_components_by_paired_boundaries(rows); };

	static_assert(!accepts_scc<int> && !accepts_scc<std::vector<std::vector<bool>>>);
	static_assert(!accepts_bridges<int> && !accepts_bridges<std::vector<std::string>>);
	static_assert(!accepts_blocks<int> && !accepts_blocks<std::vector<std::vector<double>>>);
	static_assert(!accepts_event_scc<int> && !accepts_event_scc<std::vector<std::vector<bool>>>);
	static_assert(!accepts_paired_blocks<int> && !accepts_paired_blocks<std::vector<std::vector<double>>>);
}

TEST_CASE("graph component constraints reject non-adjacency inputs", "[graph][constraints]") {
	SUCCEED();
}

namespace graph_tree_boundary {
	using namespace atl::graph;
	using rows = std::vector<std::vector<vertex_id>>;

	template<typename Graph, typename Queries>
	concept offline_queries = requires(Graph const &graph, Queries const &queries) {
		offline_lca(graph, queries);
		offline_lca(graph, 0, queries);
	};
	template<typename Keys>
	concept virtual_keys = requires(lca const &tree, Keys &keys, virtual_tree &out) {
		virtual_tree_of(tree, keys);
		virtual_tree_of(tree, keys, out);
	};
	template<typename Event>
	concept path_callback = requires(lca const &tree, Event event) { tree.path(0, 0, event); };
	template<typename Event>
	concept separator_callback = requires(rows const &graph, Event event) { centroid_visit(graph, event); };
	template<typename Event>
	concept bisect_callback = requires(rows const &graph, Event event) { centroid_bisect(graph, event); };
	template<typename Event>
	concept split_callback = requires(rows const &graph, Event event) { atl::experimental::centroid_bisect_by_single_span(graph, event); };
	template<typename Weight>
	concept ternarize = requires(rows const &graph) { ternarization_of<Weight>(graph, 0); };
	template<typename Graph>
	concept separators = requires(Graph const &graph) { centroid_decomposition(graph); };

	struct wrong_event {
		void operator()(int *) const;
	};

	struct no_default_weight {
		no_default_weight() = delete;
		explicit no_default_weight(int);
	};

	static_assert(!std::constructible_from<lca, int>);
	static_assert(!std::constructible_from<lca, std::vector<std::string>>);
	static_assert(!offline_queries<rows, std::vector<int>>);
	static_assert(!offline_queries<int, std::vector<std::pair<vertex_id, vertex_id>>>);
	static_assert(!virtual_keys<std::vector<std::string>>);
	static_assert(!path_callback<wrong_event> && !separator_callback<wrong_event>);
	static_assert(!bisect_callback<wrong_event> && !split_callback<wrong_event>);
	static_assert(!ternarize<no_default_weight> && !separators<int>);
}

namespace graph_tree_boundary {
	static_assert(!std::constructible_from<atl::experimental::lca_by_binary_lifting, int, vertex_id>);
	static_assert(!std::constructible_from<atl::experimental::lca_by_head_jumping, int, vertex_id>);
	static_assert(!std::constructible_from<atl::experimental::lca_by_euler_sparse_table, int, vertex_id>);
	static_assert(!std::constructible_from<atl::experimental::lca_by_preorder_sparse_table, int, vertex_id>);
	static_assert(!std::constructible_from<atl::experimental::lca_by_euler_direct_rmq, int, vertex_id>);
	static_assert(!std::constructible_from<atl::experimental::lca_by_preorder_direct_rmq, int, vertex_id>);
	static_assert(!std::constructible_from<atl::experimental::lca_by_adjacency_preorder, int, vertex_id>);
	static_assert(!std::constructible_from<atl::experimental::lca_by_farach_colton_bender, int, vertex_id>);
}

namespace graph_connectivity_boundary {
	template<typename Callback>
	concept replay_callback = requires(atl::experimental::deletion_log events, Callback &&callback) {
		atl::experimental::replay_by_deletion_time(0, events, std::forward<Callback>(callback));
	};

	struct wrong_query {
		void operator()(int *) const;
	};

	static_assert(!replay_callback<wrong_query>);
	static_assert(!replay_callback<int>);
}

namespace graph_review_boundary {
	struct rvalue_id {
		operator atl::graph::vertex_id() &&;
	};

	struct rvalue_members {
		rvalue_id to, from;
	};

	template<typename Edge>
	concept has_target = requires(Edge const &value) { atl::graph::target(value); };
	template<typename Edge>
	concept has_source = requires(Edge const &value) { atl::graph::source(value); };
	static_assert(!has_target<rvalue_members>);
	static_assert(!has_source<rvalue_members>);
	static_assert(!atl::graph::adjacency_edge<rvalue_members>);
	static_assert(!atl::graph::dfs_fold_callbacks<std::vector<std::vector<atl::graph::vertex_id>>,
												  decltype([](atl::graph::visit) { return 0; }),
												  decltype([](int &, int, atl::graph::visit, atl::graph::visit, atl::graph::vertex_id) {}),
												  decltype([](int &, atl::graph::visit &&) {})>);
}
