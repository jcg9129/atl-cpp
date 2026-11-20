#include <atl/graph/concepts.hpp>
#include <atl/graph/csr.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;

static_assert(g::adjacency_graph<std::vector<std::vector<int>>>);
static_assert(g::adjacency_graph<std::vector<std::vector<g::vertex_id>>>);
static_assert(g::adjacency_graph<std::vector<std::vector<g::edge<>>>>);
static_assert(g::adjacency_graph<std::vector<std::vector<g::edge<long long>>>>);
static_assert(g::weighted_adjacency_graph<std::vector<std::vector<g::edge<int>>>>);
static_assert(!g::weighted_adjacency_graph<std::vector<std::vector<int>>>);
static_assert(g::adjacency_graph<g::csr<g::vertex_id>>);
static_assert(g::weighted_adjacency_graph<g::csr<g::edge<int>>>);

// positional pairs deliberately do NOT model anything — no named access
static_assert(!g::adjacency_graph<std::vector<std::vector<std::pair<int, int>>>>);

static_assert(g::edge_list<std::vector<g::arc<>>>);
static_assert(g::weighted_edge_list<std::vector<g::arc<int>>>);
static_assert(!g::edge_list<std::vector<g::edge<int>>>); // no source

struct convention_edge {
	unsigned to;
	double weight;
};

static_assert(g::weighted_adjacency_graph<std::vector<std::vector<convention_edge>>>);

struct adapted_edge {
	int head;
	double cost;
};

template<>
struct atl::graph::graph_traits<adapted_edge> {
	static vertex_id target(adapted_edge const &e) { return vertex_id(e.head); }

	static double weight(adapted_edge const &e) { return e.cost; }
};

static_assert(g::weighted_adjacency_graph<std::vector<std::vector<adapted_edge>>>);

TEST_CASE("graph accessors: integral, member, trait tiers", "[graph]") {
	REQUIRE(g::target(7) == 7);
	REQUIRE(g::target(g::edge<int>{3, 5}) == 3);
	REQUIRE(g::weight(g::edge<int>{3, 5}) == 5);
	REQUIRE(g::source(g::arc<int>{1, 2, 9}) == 1);
	REQUIRE(g::target(adapted_edge{4, 2.5}) == 4);
	REQUIRE(g::weight(adapted_edge{4, 2.5}) == 2.5);
}

TEST_CASE("vertex_count / out_edges dispatch over plain containers", "[graph]") {
	std::vector<std::vector<g::edge<int>>> adj(4);
	adj[1].push_back({2, 10});
	adj[1].push_back({3, 20});
	REQUIRE(g::vertex_count(adj) == 4);
	REQUIRE(g::out_edges(adj, 1).size() == 2);
	REQUIRE(g::target(g::out_edges(adj, 1)[1]) == 3);
	REQUIRE(g::out_edges(adj, 0).empty());
}

TEST_CASE("csr: directed build preserves per-vertex input order", "[graph]") {
	std::vector<g::arc<int>> arcs{{0, 2, 5}, {1, 0, 7}, {0, 1, 3}, {2, 2, 1}};
	g::csr h(4, arcs);
	REQUIRE(h.vertex_count() == 4);
	REQUIRE(h.edge_count() == 4);
	REQUIRE(h.out_degree(0) == 2);
	auto const e0 = h.out_edges(0);
	REQUIRE(g::target(e0[0]) == 2);
	REQUIRE(g::weight(e0[0]) == 5);
	REQUIRE(g::target(e0[1]) == 1);
	REQUIRE(h.out_degree(3) == 0);
	auto const e2 = h.out_edges(2);
	REQUIRE(e2.size() == 1);
	REQUIRE(g::target(e2[0]) == 2);
}

TEST_CASE("csr: unweighted arcs compress to bare vertex ids", "[graph]") {
	std::vector<g::arc<>> arcs{{0, 1}, {1, 2}, {0, 2}};
	g::csr h(3, arcs);
	static_assert(std::is_same_v<decltype(h)::value_type, g::vertex_id>);
	REQUIRE(h.out_edges(0)[0] == 1);
	REQUIRE(h.out_edges(0)[1] == 2);
}

// a weighted stored element cannot be built from weightless arcs: rejected at
// the constraint, not inside the build
static_assert(!std::constructible_from<g::csr<g::edge<int>>, std::size_t, std::vector<g::arc<>>>);
static_assert(std::constructible_from<g::csr<g::edge<int>>, std::size_t, std::vector<g::arc<int>>>);

TEST_CASE("csr: out-of-range accessor vertex is asserted, including the wrap-prone ~0u", "[graph]") {
	std::vector<g::arc<>> arcs{{0, 1}};
	g::csr h(2, arcs);
	REQUIRE(atl_test::aborts([&] { (void)h.out_edges(2); }));
	REQUIRE(atl_test::aborts([&] { (void)h.out_edges(~0u); }));
	REQUIRE(atl_test::aborts([&] { (void)h.out_degree(~0u); }));
}

TEST_CASE("graph accessors: an integral id outside vertex_id is asserted, not wrapped", "[graph]") {
	REQUIRE(atl_test::aborts([] { (void)g::target(-1); }));
	REQUIRE(atl_test::aborts([] { (void)g::target(std::int64_t(1) << 32); }));
	REQUIRE(!atl_test::aborts([] { (void)g::source(g::arc<>{0, (std::numeric_limits<g::vertex_id>::max)()}); }));

	struct wide_arc {
		std::int64_t from, to;
	};

	REQUIRE(atl_test::aborts([] { (void)g::source(wide_arc{-1, 0}); }));
	REQUIRE(atl_test::aborts([] { (void)g::target(wide_arc{0, std::int64_t(1) << 32}); }));
}

TEST_CASE("implicit_graph: a vertex count beyond vertex_id is asserted", "[graph]") {
	auto const neighbors = [](g::vertex_id) { return std::vector<g::vertex_id>{}; };
	if constexpr (std::numeric_limits<std::size_t>::digits > 32) {
		REQUIRE(atl_test::aborts([&] { g::implicit_graph(static_cast<std::size_t>(atl::uint64(1) << 32), neighbors); }));
	}
	REQUIRE(!atl_test::aborts([&] { g::implicit_graph((std::numeric_limits<g::vertex_id>::max)(), neighbors); }));
}

TEST_CASE("csr: undirected build inserts both directions", "[graph]") {
	std::vector<g::arc<int>> arcs{{0, 1, 4}, {1, 2, 6}};
	auto h = g::undirected_csr(3, arcs);
	REQUIRE(h.edge_count() == 4);
	REQUIRE(h.out_degree(1) == 2);
	REQUIRE(g::target(h.out_edges(1)[0]) == 0);
	REQUIRE(g::weight(h.out_edges(1)[0]) == 4);
	REQUIRE(g::target(h.out_edges(2)[0]) == 1);
}

TEST_CASE("implicit_graph models adjacency_graph", "[graph]") {
	auto grid = g::implicit_graph(6, [](g::vertex_id u) {
		std::vector<g::vertex_id> out;
		if (u + 1 < 6) out.push_back(u + 1);
		return out;
	});
	static_assert(g::adjacency_graph<decltype(grid)>);
	REQUIRE(g::vertex_count(grid) == 6);
	REQUIRE(g::out_edges(grid, 5).empty());
	REQUIRE(g::target(g::out_edges(grid, 2)[0]) == 3);
}

static_assert(!g::adjacency_graph<std::vector<std::string>>);		// a char grid
static_assert(!g::adjacency_graph<std::vector<std::vector<bool>>>); // an adjacency matrix
static_assert(!g::adjacency_graph<std::vector<std::vector<char32_t>>>);
static_assert(!g::adjacency_edge<unsigned char>);

struct bool_edge {
	bool to; // must be `to`: the member tier of `target` reads `e.to`, and a bool there is what gets rejected
};

struct floating_arc {
	double from, to;
};

static_assert(!g::adjacency_edge<bool_edge>);
static_assert(!g::list_edge<floating_arc>);

static_assert(std::same_as<g::edge_weight_t<g::edge<long long>>, long long>);
static_assert(std::same_as<g::edge_weight_t<g::arc<short>>, short>);
static_assert(std::same_as<g::edge_capacity_t<g::flow_arc<int, double>>, int>);
static_assert(std::same_as<g::edge_cost_t<g::flow_arc<int, double>>, double>);
static_assert(std::same_as<g::weight_t<std::vector<std::vector<g::edge<double>>>>, double>);
static_assert(std::same_as<g::capacity_t<std::vector<g::bounded_flow_arc<long long>>>, long long>);
static_assert(std::same_as<g::cost_t<std::vector<g::flow_arc<int, short>>>, short>);

/** @brief A single-pass range of arcs (declarations only: never run). */
struct input_arcs {
	struct iterator {
		using value_type = g::arc<>;
		using difference_type = std::ptrdiff_t;
		g::arc<> operator*() const;
		iterator &operator++();
		void operator++(int);
		friend bool operator==(iterator const &, std::default_sentinel_t);
	};

	iterator begin() const;
	std::default_sentinel_t end() const;
};

static_assert(std::ranges::input_range<input_arcs> && !std::ranges::forward_range<input_arcs>);
static_assert(!g::edge_list<input_arcs>);
static_assert(!std::constructible_from<g::csr<g::vertex_id>, std::size_t, input_arcs>);
// a filter view is not const-iterable, so it cannot be walked through `R const &`
using filtered_arcs = decltype(std::declval<std::vector<g::arc<>> &>()
							   | std::views::filter([](g::arc<> const &) { return true; }));
static_assert(!g::edge_list<filtered_arcs>);

static_assert(std::constructible_from<g::csr<g::edge<>>, std::size_t, std::vector<g::arc<>>>);
static_assert(std::constructible_from<g::csr<g::edge<>>, std::size_t, std::vector<g::arc<int>>>);

TEST_CASE("csr: edge<> elements build from weightless and weighted arcs", "[graph]") {
	g::csr<g::edge<>> h(3, std::vector<g::arc<>>{{0, 1}, {0, 2}, {2, 1}});
	REQUIRE(h.out_degree(0) == 2);
	REQUIRE(h.out_edges(0)[1].to == 2);
	auto const u = g::csr<g::edge<>>::undirected(3, std::vector<g::arc<int>>{{0, 1, 9}, {1, 2, 4}});
	REQUIRE(u.edge_count() == 4);
	REQUIRE(u.out_edges(1)[0].to == 0);
	REQUIRE(u.out_edges(1)[1].to == 2);
}

struct packed_arc {
	unsigned endpoints[2];
	int length;
};

template<>
struct atl::graph::graph_traits<packed_arc> {
	static vertex_id source(packed_arc const &e) { return e.endpoints[0]; }

	static vertex_id target(packed_arc const &e) { return e.endpoints[1]; }

	static int weight(packed_arc const &e) { return e.length; }
};

TEST_CASE("csr: undirected build over a graph_traits-adapted edge type", "[graph]") {
	std::vector<packed_arc> const arcs{{{0, 1}, 5}, {{1, 2}, 7}};
	auto const h = g::undirected_csr(3, arcs);
	static_assert(std::same_as<std::remove_cvref_t<decltype(h)>, g::csr<g::edge<int>>>);
	REQUIRE(h.edge_count() == 4);
	REQUIRE(h.out_degree(1) == 2);
	REQUIRE(h.out_edges(1)[0].to == 0);
	REQUIRE(h.out_edges(1)[0].weight == 5);
	REQUIRE(h.out_edges(1)[1].to == 2);
	REQUIRE(h.out_edges(1)[1].weight == 7);
	REQUIRE(h.out_edges(2)[0].to == 1);
	REQUIRE(h.out_edges(2)[0].weight == 7);
}

struct wide_trait_arc {
	atl::uint64 from, to;
};

template<>
struct atl::graph::graph_traits<wide_trait_arc> {
	static atl::uint64 source(wide_trait_arc const &a) { return a.from; }

	static atl::uint64 target(wide_trait_arc const &a) { return a.to; }
};

TEST_CASE("graph traits: integral endpoints are checked prior to narrowing", "[graph]") {
	atl::uint64 const unrepresentable = atl::uint64(1) << 32;
	REQUIRE(atl_test::aborts([&] { (void)g::target(wide_trait_arc{0, unrepresentable}); }));
	REQUIRE(atl_test::aborts([&] { (void)g::source(wide_trait_arc{unrepresentable, 0}); }));
}
