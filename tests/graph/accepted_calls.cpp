#include <atl/experimental/graph/biconnected_components_variants.hpp>
#include <atl/experimental/graph/centroid_bisect_variants.hpp>
#include <atl/experimental/graph/centroid_variants.hpp>
#include <atl/experimental/graph/csr_variants.hpp>
#include <atl/experimental/graph/deletion_time_connectivity.hpp>
#include <atl/experimental/graph/lca_variants.hpp>
#include <atl/experimental/graph/offline_dynamic_connectivity.hpp>
#include <atl/experimental/graph/scc_variants.hpp>
#include <atl/experimental/graph/two_edge_connected_components_variants.hpp>
#include <atl/experimental/graph/virtual_tree_variants.hpp>
#include <atl/graph/biconnected_components.hpp>
#include <atl/graph/centroid_bisect.hpp>
#include <atl/graph/centroid_decomposition.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/lca.hpp>
#include <atl/graph/partition.hpp>
#include <atl/graph/scc.hpp>
#include <atl/graph/ternarization.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/graph/two_edge_connected_components.hpp>
#include <atl/graph/virtual_tree.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <ranges>
#include <span>
#include <vector>

namespace {
	struct proxy_sources {
		std::vector<bool> *values;

		auto begin() const { return values->begin(); }

		auto end() const { return values->end(); }
	};

	struct arc_proxy {
		atl::graph::vertex_id &from, &to;
		int &weight;

		operator atl::graph::arc<int>() const { return {from, to, weight}; }
	};

	struct proxy_arcs {
		atl::graph::arc<int> *first = nullptr, *last = nullptr;

		struct iterator {
			using value_type = atl::graph::arc<int>;
			using difference_type = std::ptrdiff_t;
			using iterator_concept = std::forward_iterator_tag;
			atl::graph::arc<int> *at = nullptr;

			arc_proxy operator*() const { return {at->from, at->to, at->weight}; }

			iterator &operator++() {
				++at;
				return *this;
			}

			iterator operator++(int) {
				auto saved = *this;
				++*this;
				return saved;
			}

			bool operator==(iterator const &) const = default;
		};

		iterator begin() const { return {first}; }

		iterator end() const { return {last}; }
	};

	struct immovable_projection {
		immovable_projection() = default;
		immovable_projection(immovable_projection const &) = delete;
		immovable_projection(immovable_projection &&) = delete;

		atl::graph::vertex_id operator()(atl::graph::arc<int> const &arc) const { return arc.to; }
	};

	struct fold_accumulator {
		int size;

		void merge(fold_accumulator child, atl::graph::visit, atl::graph::visit, atl::graph::vertex_id) { size += child.size; }

		void finish(atl::graph::visit) { ++size; }
	};
}

namespace graph_accepted {
	struct endpoint {
		atl::graph::vertex_id value;

		explicit endpoint(atl::graph::vertex_id v) : value(v) {}

		endpoint(endpoint const &) = delete;

		endpoint(endpoint &&) = default;

		operator atl::graph::vertex_id() const { return value; }
	};

	struct rvalue_endpoint {
		atl::graph::vertex_id value;

		operator atl::graph::vertex_id() && { return value; }
	};

	struct rvalue_trait_edge {
		atl::graph::vertex_id from, to;
	};

	struct borrowed_endpoint {
		endpoint const &id;
	};
}

template<>
struct atl::graph::graph_traits<graph_accepted::borrowed_endpoint> {
	static graph_accepted::endpoint const &target(graph_accepted::borrowed_endpoint const &edge) { return edge.id; }

	static graph_accepted::endpoint const &source(graph_accepted::borrowed_endpoint const &edge) { return edge.id; }
};

template<>
struct atl::graph::graph_traits<graph_accepted::rvalue_trait_edge> {
	static graph_accepted::rvalue_endpoint target(graph_accepted::rvalue_trait_edge const &edge) { return {edge.to}; }

	static graph_accepted::rvalue_endpoint source(graph_accepted::rvalue_trait_edge const &edge) { return {edge.from}; }
};

TEST_CASE("graph accepted calls: CSR projections and multipass arc ranges", "[graph][accepted]") {
	using namespace atl::graph;
	static_assert(signed_flow_value<atl::int128> && flow_value<atl::uint128>);
	std::vector<arc<int>> arcs{{0, 1, 4}, {1, 2, 5}};
	graph_accepted::endpoint const endpoint_value(2);
	graph_accepted::borrowed_endpoint const adapted{endpoint_value};
	REQUIRE(target(adapted) == 2);
	REQUIRE(source(adapted) == 2);
	REQUIRE(target(graph_accepted::rvalue_trait_edge{0, 2}) == 2);
	REQUIRE(source(graph_accepted::rvalue_trait_edge{1, 2}) == 1);
	int offset = 0;
	auto projection = [&offset](arc<int> const &a) { return vertex_id(a.to + offset); };
	auto mutable_projection = [offset](arc<int> const &a) mutable { return vertex_id(a.to + offset); };
	csr<vertex_id> const captured(3, arcs, projection);
	csr<vertex_id> const wrapped(3, arcs, std::ref(mutable_projection));
	csr<vertex_id> const member_projection(3, arcs, &arc<int>::to);
	csr<vertex_id> const immovable(3, arcs, immovable_projection{});
	REQUIRE(immovable.out_degree(0) == 1);
	REQUIRE(captured.out_edges(0)[0] == wrapped.out_edges(0)[0]);
	REQUIRE(member_projection.out_edges(1)[0] == 2);
	std::list<arc<int>> const linked(arcs.begin(), arcs.end());
	csr<edge<int>> const weighted(3, linked);
	REQUIRE(weighted.out_edges(1)[0].weight == 5);
	REQUIRE(csr<edge<>>::undirected(3, linked).edge_count() == 4);
	REQUIRE(undirected_csr(3, linked).edge_count() == 4);
	REQUIRE(atl::experimental::csr_by_cursor_copy<vertex_id, false>(3, linked).edges.size() == 2);
	REQUIRE(atl::experimental::csr_by_bucket_gather<vertex_id, true>(3, arcs).edges.size() == 4);
	auto generated = arcs | std::views::transform([](arc<int> const &a) { return arc<>{a.from, a.to}; });
	csr const transformed(3, generated);
	REQUIRE(transformed.out_edges(1)[0] == 2);
}

TEST_CASE("graph accepted calls: callable categories, source proxies and marker overloads", "[graph][accepted]") {
	using namespace atl::graph;
	std::vector<std::vector<vertex_id>> const rows{{1, 2}, {3}, {}, {}};
	std::vector<bool> bits{false, true};
	proxy_sources const sources{&bits};
	static_assert(vertex_range<proxy_sources const>);
	int visits = 0;
	auto step = [&](vertex_id, vertex_id, atl::uint32) mutable { ++visits; };
	auto const const_step = [&](vertex_id, vertex_id, atl::uint32) { ++visits; };
	epoch_marker marker(4);
	bfs_visit(rows, 0, std::ref(step));
	bfs_visit(rows, sources, const_step);
	bfs_visit(rows, 0, marker, step);
	marker.next_run();
	bfs_visit(rows, sources, marker, std::ref(step));
	REQUIRE(visits == 16);
	int lazy_visits = 0;
	for ([[maybe_unused]] auto const &node : bfs(rows, sources)) ++lazy_visits;
	marker.next_run();
	for ([[maybe_unused]] auto const &node : bfs(rows, sources, marker)) ++lazy_visits;
	marker.next_run();
	for ([[maybe_unused]] auto const &node : bfs(rows, 0, marker)) ++lazy_visits;
	for ([[maybe_unused]] auto const &node : dfs(rows, 0)) ++lazy_visits;
	marker.next_run();
	for ([[maybe_unused]] auto const &node : dfs(rows, 0, marker)) ++lazy_visits;
	REQUIRE(lazy_visits == 20);
	auto enter = [&](vertex_id, vertex_id, atl::uint32, atl::uint32) mutable { ++visits; };
	auto const exit = [&](vertex_id, vertex_id) { ++visits; };
	dfs_visit(rows, 0, std::ref(enter), skip, exit);
	dfs_visit(rows, std::ref(enter), skip, exit);
	marker.next_run();
	dfs_visit(rows, 0, marker, std::ref(enter), skip, exit);
	marker.next_run();
	dfs_visit(rows, marker, std::ref(enter), skip, exit);
	dfs_visit(rows, 0, events(exit, std::ref(enter)));
	dfs_visit(rows, events(exit));
	marker.next_run();
	dfs_visit(rows, 0, marker, events(std::ref(enter)));
	marker.next_run();
	dfs_visit(rows, marker, events(exit, std::ref(enter)));
	REQUIRE(visits == 72);
}

TEST_CASE("graph accepted calls: forward adjacency, generated rows and move-only folds", "[graph][accepted]") {
	using namespace atl::graph;
	std::vector<std::list<vertex_id>> const linked{{1, 2}, {}, {}};
	int visited = 0;
	for ([[maybe_unused]] auto const &node : dfs(linked, 0)) ++visited;
	REQUIRE(visited == 3);
	auto neighbors = [limit = vertex_id(3)](vertex_id v) { return std::array{(v + 1) % limit}; };
	implicit_graph const generated(3, std::cref(neighbors));
	implicit_graph const move_only(3, [limit = std::make_unique<int>(3)](vertex_id v) { return std::array{(v + 1) % vertex_id(*limit)}; });
	REQUIRE(move_only.out_edges(2)[0] == 0);
	for ([[maybe_unused]] auto const &node : bfs(generated, 0)) ++visited;
	REQUIRE(visited == 6);
	auto init = [](visit) { return std::make_unique<int>(1); };
	auto fold = [](std::unique_ptr<int> &acc, std::unique_ptr<int> child, visit, visit, vertex_id) { *acc += *child; };
	auto const finalize = [](std::unique_ptr<int> &, visit) {};
	REQUIRE(*dfs_fold(linked, 0, std::ref(init), std::ref(fold)) == 3);
	REQUIRE(*dfs_fold(linked, 0, std::ref(init), std::ref(fold), finalize) == 3);
	auto member_init = [](visit) { return fold_accumulator{1}; };
	REQUIRE(dfs_fold(linked, 0, member_init, &fold_accumulator::merge, &fold_accumulator::finish).size == 6);
}

TEST_CASE("graph accepted calls: immovable callbacks used directly with a caller marker", "[graph][accepted]") {
	struct immovable_enter {
		int *count;

		explicit immovable_enter(int &value) : count(&value) {}

		immovable_enter(immovable_enter const &) = delete;
		immovable_enter(immovable_enter &&) = delete;

		void operator()(atl::graph::vertex_id, atl::graph::vertex_id, atl::uint32, atl::uint32) { ++*count; }
	};

	std::vector<std::vector<atl::graph::vertex_id>> const rows{{1}, {}};
	atl::graph::epoch_marker marks(2);
	int count = 0;
	atl::graph::dfs_visit(rows, 0, marks, immovable_enter(count), atl::graph::skip, atl::graph::skip);
	marks.next_run();
	atl::graph::dfs_visit(rows, marks, immovable_enter(count), atl::graph::skip, atl::graph::skip);
	REQUIRE(count == 4);
}

TEST_CASE("graph accepted calls: proxy arc and adjacency references", "[graph][accepted]") {
	using namespace atl::graph;
	std::array<arc<int>, 2> arcs{{{0, 1, 4}, {0, 2, 5}}};
	proxy_arcs const proxies{arcs.data(), arcs.data() + arcs.size()};
	static_assert(weighted_edge_list<proxy_arcs>);
	csr const rows(3, proxies);
	REQUIRE(rows.out_edges(0)[1].weight == 5);
	REQUIRE(undirected_csr(3, proxies).out_degree(0) == 2);
	std::vector<proxy_arcs> const graph{proxies, {}, {}};
	static_assert(weighted_adjacency_graph<decltype(graph)>);
	int visited = 0;
	for ([[maybe_unused]] auto const &node : dfs(graph, 0)) ++visited;
	REQUIRE(visited == 3);
	int const total = dfs_fold(graph, 0, [](visit) { return 1; }, [](int &sum, int child, visit, visit, arc<int> const &) { sum += child; });
	REQUIRE(total == 3);
}

namespace graph_components_accepted {
	template<atl::graph::detail::scannable_graph Graph>
	void triangle(Graph const &rows) {
		REQUIRE(atl::graph::scc(rows).count() == 1);
		REQUIRE(atl::graph::two_edge_connected_components(rows).count() == 1);
		REQUIRE(atl::graph::biconnected_components(rows).count() == 1);
		REQUIRE(atl::experimental::scc_by_tarjan_events(rows).count() == 1);
		REQUIRE(atl::experimental::biconnected_components_by_paired_boundaries(rows).count() == 1);
	}
}

TEST_CASE("graph components accept forward, generated and proxy adjacency", "[graph][accepted]") {
	using namespace atl::graph;
	std::vector<std::list<vertex_id>> const forward{{1, 2}, {0, 2}, {0, 1}};
	graph_components_accepted::triangle(forward);
	auto const neighbors = [vertices = vertex_id(3)](vertex_id v) {
		return std::array{(v + 1) % vertices, (v + 2) % vertices};
	};
	implicit_graph generated(3, std::cref(neighbors));
	graph_components_accepted::triangle(generated);

	std::array<arc<int>, 6> arcs{{{0, 1, 0}, {0, 2, 0}, {1, 0, 0}, {1, 2, 0}, {2, 0, 0}, {2, 1, 0}}};
	std::array<proxy_arcs, 3> const proxies{{{arcs.data(), arcs.data() + 2}, {arcs.data() + 2, arcs.data() + 4}, {arcs.data() + 4, arcs.data() + 6}}};
	graph_components_accepted::triangle(proxies);
	csr<vertex_id> const rows(3, arcs);
	graph_components_accepted::triangle(rows);
	REQUIRE(atl::experimental::scc_by_tarjan(rows).count() == 1);
	REQUIRE(atl::experimental::scc_by_gabow(rows).count() == 1);
	REQUIRE(atl::experimental::scc_by_kosaraju(rows).count() == 1);
	REQUIRE(atl::experimental::two_edge_connected_components_by_low_link(rows).count() == 1);
	REQUIRE(atl::experimental::two_edge_connected_components_by_gabow(rows).count() == 1);
	REQUIRE(atl::experimental::two_edge_connected_components_by_chain_decomposition(rows).count() == 1);
	REQUIRE(atl::experimental::biconnected_components_by_single_array(rows).count() == 1);
	REQUIRE(atl::experimental::biconnected_components_by_low_link(rows).count() == 1);
	REQUIRE(atl::experimental::cut_vertex_mask_by_chain_decomposition(rows) == std::vector<atl::uint8>{0, 0, 0});
}

TEST_CASE("tree algorithms accept captures, wrappers, forward rows and proxy keys", "[graph][accepted]") {
	using namespace atl::graph;
	std::vector<std::list<vertex_id>> const rows{{1, 2}, {0}, {0}};
	lca const rooted(rows, 0), forest(rows);
	std::vector<bool> parent_bits{false, false};
	lca const proxy_parents(parent_bits);
	lca const wide_parents(std::array<atl::int64, 3>{-1, 0, 0});
	REQUIRE(wide_parents(1, 2) == 0);
	REQUIRE(virtual_tree_of(wide_parents, std::array<atl::uint64, 2>{1, 2}).vertices.size() == 3);
	REQUIRE(proxy_parents.parent(1) == 0);
	std::vector<vertex_id> parent_values{none, 0, 0};
	auto parents = parent_values | std::views::take_while([](vertex_id) { return true; });
	lca const filtered_parents(parents);
	REQUIRE(filtered_parents(1, 2) == 0);
	std::array<std::array<vertex_id, 2>, 1> const queries{{{1, 2}}};
	REQUIRE(offline_lca(rows, 0, queries) == std::vector<vertex_id>{0});
	REQUIRE(offline_lca(rows, queries) == std::vector<vertex_id>{0});
	int segments = 0;
	auto path_event = [&](atl::uint32, atl::uint32, bool) mutable { ++segments; };
	rooted.path(1, 2, std::ref(path_event));
	REQUIRE(segments > 0);
	std::vector<bool> key_bits{true, false, true};
	virtual_tree result;
	virtual_tree_of(forest, key_bits, result);
	REQUIRE(result.vertices == virtual_tree_of(rooted, key_bits).vertices);
	int visits = 0;
	auto event = [&](vertex_id, vertex_id, std::span<centroid_vertex const>) mutable { ++visits; };
	centroid_visit(rows, std::ref(event));
	auto const edges = [&](vertex_id, vertex_id, std::span<centroid_vertex const> comp, std::span<vertex_id const> ec) {
		REQUIRE(comp.size() == ec.size());
		++visits;
	};
	centroid_visit(rows, std::cref(edges));
	REQUIRE(visits == 6);
	REQUIRE(centroid_decomposition(rows).size() == 3);
	std::size_t pairs = 0;
	auto bisect = [&](vertex_id, std::span<visit const> a, std::span<visit const> b) mutable { pairs += a.size() * b.size(); };
	centroid_bisect(rows, std::ref(bisect));
	auto const weighted = [&](vertex_id, std::span<visit const> a, std::span<visit const> b,
							  std::span<vertex_id const> ea, std::span<vertex_id const> eb) {
		REQUIRE(a.size() == ea.size());
		REQUIRE(b.size() == eb.size());
		pairs += a.size() * b.size();
	};
	centroid_bisect(rows, std::cref(weighted));
	REQUIRE(pairs == 2);
	auto const split = [&](vertex_id, std::span<visit const>, std::size_t) { ++visits; };
	atl::experimental::centroid_bisect_by_single_span(rows, std::cref(split));
	atl::experimental::centroid_visit_by_subtree_sizes(rows, std::ref(event));
	atl::experimental::centroid_visit_by_reused_sizes(rows, std::ref(event));
	REQUIRE(atl::experimental::centroid_decomposition_by_component_sweep(rows).size() == 3);
	REQUIRE(atl::experimental::lca_by_binary_lifting(rows, 0)(1, 2) == 0);
	REQUIRE(atl::experimental::lca_by_head_jumping(rows, 0)(1, 2) == 0);
	REQUIRE(atl::experimental::lca_by_euler_sparse_table(rows, 0)(1, 2) == 0);
	REQUIRE(atl::experimental::lca_by_preorder_sparse_table(rows, 0)(1, 2) == 0);
	REQUIRE(atl::experimental::lca_by_euler_direct_rmq(rows, 0)(1, 2) == 0);
	REQUIRE(atl::experimental::lca_by_preorder_direct_rmq(rows, 0)(1, 2) == 0);
	REQUIRE(atl::experimental::lca_by_adjacency_preorder(rows, 0)(1, 2) == 0);
	REQUIRE(atl::experimental::lca_by_farach_colton_bender(rows, 0)(1, 2) == 0);
	REQUIRE(ternarization_of(rows, 0).real_count == 3);
	REQUIRE(ternarization_of<atl::int128>(rows, 0).real_count == 3);
	virtual_tree_clear();
}

TEST_CASE("deletion-time replay accepts callable references and captures", "[graph][constraints]") {
	using namespace atl::experimental;
	std::vector<graph_event> const log{{atl::event_kind::query, {0, 0}}};
	int calls = 0;
	auto mutable_query = [&](deletion_time_connectivity &state, atl::uint32 time) mutable {
		CHECK(state.component_count() == 1);
		CHECK(time == 0);
		++calls;
	};
	auto const const_query = [&](deletion_time_connectivity &, atl::uint32) { ++calls; };
	replay_by_deletion_time(1, log, mutable_query);
	replay_by_deletion_time(1, log, std::ref(mutable_query));
	replay_by_deletion_time(1, log, const_query);
	replay_by_deletion_time(1, log, std::cref(const_query));
	replay_by_deletion_time(1, log, [&](deletion_time_connectivity &, atl::uint32) { ++calls; });
	CHECK(calls == 5);
}

TEST_CASE("graph accepted calls: const projection and move-only source ids", "[graph][accepted]") {
	using namespace atl::graph;

	struct const_projection {
		void operator()(arc<> const &) {}

		vertex_id operator()(arc<> const &a) const { return a.to; }
	};

	std::array<arc<>, 1> const input{{{0, 1}}};
	csr<vertex_id> const projected(2, input, const_projection{});
	REQUIRE(projected.out_edges(0)[0] == 1);
	std::array<graph_accepted::endpoint, 3> const seeds{
		graph_accepted::endpoint(0), graph_accepted::endpoint(0), graph_accepted::endpoint(1)};
	std::vector<std::vector<vertex_id>> const rows{{1}, {2}, {}};
	epoch_marker marks(3);
	int total = 0;
	for ([[maybe_unused]] auto const &node : bfs(rows, seeds)) ++total;
	for ([[maybe_unused]] auto const &node : bfs(rows, seeds, marks)) ++total;
	bfs_visit(rows, seeds, [&](vertex_id, vertex_id, atl::uint32) { ++total; });
	marks.next_run();
	bfs_visit(rows, seeds, marks, [&](vertex_id, vertex_id, atl::uint32) { ++total; });
	REQUIRE(total == 12);
}

TEST_CASE("graph accepted calls: event scalar value categories", "[graph][accepted]") {
	using namespace atl::graph;
	std::vector<std::vector<vertex_id>> const rows{{1, 2}, {0}, {0}};
	int bfs_calls = 0, enter_calls = 0, seen_calls = 0, exit_calls = 0;
	bfs_visit(rows, 0, [&](vertex_id, vertex_id, atl::uint32) { ++bfs_calls; });
	bfs_visit(rows, 0, [&](vertex_id const &, vertex_id const &, atl::uint32 const &) { ++bfs_calls; });
	bfs_visit(rows, 0, [&](vertex_id &&, vertex_id &&, atl::uint32 &&) { ++bfs_calls; });
	bfs_visit(rows, 0, [&](auto &&, auto &&, auto &&) { ++bfs_calls; });
	dfs_visit(rows, 0, [&](vertex_id, vertex_id, atl::uint32, atl::uint32) { ++enter_calls; }, [&](vertex_id, atl::uint32, vertex_id) { ++seen_calls; }, [&](vertex_id, vertex_id) { ++exit_calls; });
	dfs_visit(rows, 0, [&](vertex_id const &, vertex_id const &, atl::uint32 const &, atl::uint32 const &) { ++enter_calls; }, [&](vertex_id const &, atl::uint32 const &, vertex_id const &) { ++seen_calls; }, [&](vertex_id const &, vertex_id const &) { ++exit_calls; });
	dfs_visit(rows, 0, events([&](vertex_id &&, vertex_id &&, atl::uint32 &&, atl::uint32 &&) { ++enter_calls; }, [&](vertex_id &&, atl::uint32 &&, vertex_id &&) { ++seen_calls; }, [&](vertex_id &&, vertex_id &&) { ++exit_calls; }));
	dfs_visit(rows, 0, [&](auto &&, auto &&, auto &&, auto &&) { ++enter_calls; }, [&](auto &&, auto &&, auto &&) { ++seen_calls; }, [&](auto &&, auto &&) { ++exit_calls; });
	REQUIRE(bfs_calls == 12);
	REQUIRE(enter_calls == 12);
	REQUIRE(seen_calls == 8);
	REQUIRE(exit_calls == 12);
	auto fold_init = [](visit &&) { return 1; };
	auto fold_child = [](int &sum, int child, visit &&, visit &&, vertex_id const &) { sum += child; };
	REQUIRE(dfs_fold(rows, 0, fold_init, fold_child, [](int &, visit const &) {}) == 3);
	REQUIRE(dfs_fold(rows, 0, fold_init, fold_child, [](int &, visit) {}) == 3);
	lca const tree(rows, 0);
	std::vector<std::array<atl::uint32, 3>> by_value, by_reference;
	for (vertex_id u = 0; u < 3; ++u)
		for (vertex_id v = 0; v < 3; ++v)
			for (bool includes_top : {false, true}) {
				tree.path(u, v, [&](std::uint32_t l, std::uint32_t r, bool reversed) { by_value.push_back({l, r, atl::uint32(reversed)}); }, includes_top);
				tree.path(u, v, [&](std::uint32_t &&l, std::uint32_t &&r, bool reversed) { by_reference.push_back({l, r, atl::uint32(reversed)}); }, includes_top);
			}
	REQUIRE_FALSE(by_value.empty());
	REQUIRE(by_value == by_reference);
}

namespace graph_review_accepted {
	struct stored_edge;

	struct edge_reference {
		atl::graph::vertex_id to;
		operator stored_edge() &&;
	};

	struct stored_edge {
		atl::graph::vertex_id to{};
		stored_edge() = default;

		explicit stored_edge(edge_reference const &element) : to(element.to) {}

		stored_edge &operator=(edge_reference const &element) {
			to = element.to;
			return *this;
		}
	};

	inline edge_reference::operator stored_edge() && { return stored_edge(*this); }

	struct edge_row {
		atl::graph::vertex_id const *first{}, *last{};

		struct iterator {
			using value_type = stored_edge;
			using difference_type = std::ptrdiff_t;
			using iterator_concept = std::forward_iterator_tag;
			atl::graph::vertex_id const *position{};

			edge_reference operator*() const { return {*position}; }

			iterator &operator++() {
				++position;
				return *this;
			}

			iterator operator++(int) {
				auto saved = *this;
				++*this;
				return saved;
			}

			bool operator==(iterator const &) const = default;
		};

		iterator begin() const { return {first}; }

		iterator end() const { return {last}; }
	};
}

TEST_CASE("graph accepted calls: explicitly constructible stored parent edges", "[graph][accepted]") {
	using namespace graph_review_accepted;
	using atl::graph::vertex_id;
	static_assert(std::constructible_from<stored_edge, edge_reference const &>);
	static_assert(!std::convertible_to<edge_reference const &, stored_edge>);
	std::array<vertex_id, 4> const targets{1, 2, 0, 0};
	std::array<edge_row, 3> const rows{{{targets.data(), targets.data() + 2},
										{targets.data() + 2, targets.data() + 3},
										{targets.data() + 3, targets.data() + 4}}};
	std::size_t pair_total = 0;
	atl::graph::centroid_bisect(rows, [&](vertex_id, std::span<atl::graph::visit const> a,
										  std::span<atl::graph::visit const> b, std::span<stored_edge const> ea, std::span<stored_edge const> eb) {
		REQUIRE(ea.size() == a.size());
		REQUIRE(eb.size() == b.size());
		pair_total += a.size() * b.size();
	});
	REQUIRE(pair_total == 1);
}
