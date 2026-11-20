#include <atl/experimental/graph/biconnected_components_variants.hpp>
#include <atl/experimental/graph/centroid_bisect_variants.hpp>
#include <atl/experimental/graph/centroid_variants.hpp>
#include <atl/experimental/graph/lca_variants.hpp>
#include <atl/experimental/graph/offline_bipartiteness.hpp>
#include <atl/experimental/graph/offline_dynamic_connectivity.hpp>
#include <atl/experimental/graph/virtual_tree_variants.hpp>
#include <atl/graph/biconnected_components.hpp>
#include <atl/graph/centroid_bisect.hpp>
#include <atl/graph/centroid_decomposition.hpp>
#include <atl/graph/concepts.hpp>
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
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace graph_checks {
	constexpr bool vocabulary() {
		using namespace atl::graph;
		std::array<std::array<int, 1>, 3> const rows{{{1}, {2}, {0}}};
		implicit_graph generated(4, [](vertex_id u) { return std::array{(u + 1) % 4}; });
		return target(7) == 7 && target(edge<int>{3, -2}) == 3 && weight(edge<int>{3, -2}) == -2
			&& source(arc<>{4, 1}) == 4 && target(arc<>{4, 1}) == 1
			&& capacity(flow_arc<int, long>{0, 1, 5, 9}) == 5 && cost(flow_arc<int, long>{0, 1, 5, 9}) == 9
			&& lower(bounded_flow_arc<int>{0, 1, 2, 5}) == 2 && vertex_count(rows) == 3
			&& target(out_edges(rows, 1)[0]) == 2 && vertex_count(generated) == 4
			&& target(out_edges(generated, 3)[0]) == 0;
	}

	constexpr bool event_roles() {
		int entered = 0, exited = 0;
		auto bundle = atl::graph::events(
			[&](atl::graph::vertex_id, atl::graph::vertex_id) { ++exited; },
			[&](atl::graph::vertex_id, atl::graph::vertex_id, atl::uint32, atl::uint32) { ++entered; });
		bundle.enter(0, 0, atl::graph::none, 0);
		bundle.exit(0, 0);
		bundle.seen_edge(0, 0, 0);
		return entered == 1 && exited == 1;
	}

	constexpr bool markers() {
		atl::graph::dense_marker dense(65);
		atl::graph::epoch_marker epoch(3);
		bool const is_fresh = dense.try_visit(64) && !dense.try_visit(64) && epoch.try_visit(2);
		epoch.next_run();
		bool const is_reset = is_fresh && dense.visited(64) && !epoch.visited(2) && epoch.try_visit(2);
		dense.reset();
		return is_reset && !dense.visited(64) && !dense.visited(0);
	}

	constexpr bool eager_traversals() {
		std::vector<std::vector<atl::graph::vertex_id>> const rows{{1, 2}, {3}, {}, {}};
		std::vector<atl::graph::vertex_id> order;
		int exits = 0;
		atl::graph::dfs_visit(
			rows, 0, [&](atl::graph::vertex_id v, atl::graph::vertex_id, atl::uint32, atl::uint32) { order.push_back(v); },
			atl::graph::skip, [&](atl::graph::vertex_id, atl::graph::vertex_id) { ++exits; });
		std::array<atl::uint32, 4> depth{};
		atl::graph::bfs_visit(rows, 0, [&](atl::graph::vertex_id v, atl::graph::vertex_id, atl::uint32 d) { depth[v] = d; });
		int const size = atl::graph::dfs_fold(
			rows, 0, [](atl::graph::visit) { return 1; },
			[](int &acc, int child, atl::graph::visit, atl::graph::visit, atl::graph::vertex_id) { acc += child; });
		return order == std::vector<atl::graph::vertex_id>{0, 1, 3, 2} && exits == 4
			&& depth == std::array<atl::uint32, 4>{0, 1, 1, 2} && size == 4;
	}

	constexpr bool compressed_rows() {
		std::array<atl::graph::arc<>, 3> const arcs{{{0, 2}, {1, 0}, {0, 1}}};
		atl::graph::csr rows(3, arcs);
		auto const out = rows.out_edges(0);
		auto const both = atl::graph::undirected_csr(3, arcs);
		std::array<atl::graph::arc<int>, 2> const weighted{{{0, 1, 5}, {1, 2, 7}}};
		atl::graph::csr weights(3, weighted);
		auto const adopted = atl::graph::csr<atl::graph::vertex_id>::adopt({0, 1, 1}, {1});
		return rows.vertex_count() == 3 && rows.edge_count() == 3 && rows.out_degree(0) == 2
			&& out[0] == 2 && out[1] == 1 && both.edge_count() == 6 && both.out_degree(0) == 3
			&& both.out_edges(0)[1] == 1 && weights.out_edges(1)[0].to == 2
			&& weights.out_edges(1)[0].weight == 7 && adopted.out_edges(1).empty();
	}

	constexpr bool partitions() {
		atl::graph::vertex_partition const partition{{0, 1, 0}, 2};
		auto const flat = partition.grouped();
		return partition.count() == 2 && partition.same(0, 2) && !partition.same(0, 1)
			&& partition.groups()[0].size() == 2 && flat.count() == 2
			&& flat[0].size() == 2 && flat[1][0] == 1
			&& atl::graph::vertex_partition{{}, 2}.grouped()[0].empty();
	}

	constexpr bool lazy_traversals() {
		std::vector<std::vector<atl::graph::vertex_id>> const rows{{1, 2}, {3}, {}, {}};
		std::vector<atl::graph::vertex_id> breadth, depth;
		for (auto const &node : atl::graph::bfs(rows, 0)) breadth.push_back(node.vertex);
		for (auto const &node : atl::graph::dfs(rows, 0)) depth.push_back(node.vertex);
		return breadth == std::vector<atl::graph::vertex_id>{0, 1, 2, 3}
			&& depth == std::vector<atl::graph::vertex_id>{0, 1, 3, 2};
	}

	constexpr bool overloads() {
		using namespace atl::graph;
		std::vector<std::vector<vertex_id>> const rows{{1, 2}, {3}, {}, {}};
		std::array<vertex_id, 2> const sources{0, 1};
		epoch_marker marks(4);
		int count = 0;
		for ([[maybe_unused]] auto const &node : bfs(rows, sources)) ++count;
		for ([[maybe_unused]] auto const &node : bfs(rows, sources, marks)) ++count;
		marks.next_run();
		for ([[maybe_unused]] auto const &node : bfs(rows, 0, marks)) ++count;
		marks.next_run();
		for ([[maybe_unused]] auto const &node : dfs(rows, 0, marks)) ++count;
		auto step = [&](vertex_id, vertex_id, atl::uint32) { ++count; };
		bfs_visit(rows, sources, step);
		marks.next_run();
		bfs_visit(rows, sources, marks, step);
		marks.next_run();
		bfs_visit(rows, 0, marks, step);
		auto exit = [&](vertex_id, vertex_id) { ++count; };
		dfs_visit(rows, skip, skip, exit);
		marks.next_run();
		dfs_visit(rows, 0, marks, skip, skip, exit);
		marks.next_run();
		dfs_visit(rows, marks, skip, skip, exit);
		dfs_visit(rows, 0, events(exit));
		dfs_visit(rows, events(exit));
		marks.next_run();
		dfs_visit(rows, 0, marks, events(exit));
		marks.next_run();
		dfs_visit(rows, marks, events(exit));
		auto const folded = dfs_fold(rows, 0, [](visit) { return 1; }, [](int &acc, int child, visit, visit, vertex_id) { acc += child; }, [](int &, visit) {});
		std::array<arc<>, 1> const arcs{{{0, 1}}};
		csr<vertex_id> const projected(2, arcs, [](arc<> const &a) { return a.to; });
		csr<short> const narrow(2, arcs);
		return count == 56 && folded == 4 && projected.out_edges(0)[0] == 1
			&& narrow.out_edges(0)[0] == 1 && vertex_groups{}.count() == 0;
	}

	static_assert(atl::graph::flow_value<atl::int128>);
	static_assert(atl::graph::signed_flow_value<atl::int128>);
	static_assert(atl::graph::flow_value<atl::uint128>);
	static_assert(!atl::graph::signed_flow_value<atl::uint128>);
	static_assert(std::size_t(atl::graph::none) / 64 + 1 == (atl::uint64(atl::graph::none) + 63) / 64);
	static_assert(overloads());
	static_assert(vocabulary());
	static_assert(event_roles());
	static_assert(markers());
	static_assert(eager_traversals());
	static_assert(compressed_rows());
	static_assert(partitions());
	static_assert(lazy_traversals());
}

TEST_CASE("graph vocabulary, rows, partitions and traversals constant-evaluate", "[graph][constexpr]") {
	REQUIRE(graph_checks::overloads());
	REQUIRE(graph_checks::vocabulary());
	REQUIRE(graph_checks::event_roles());
	REQUIRE(graph_checks::markers());
	REQUIRE(graph_checks::eager_traversals());
	REQUIRE(graph_checks::compressed_rows());
	REQUIRE(graph_checks::partitions());
	REQUIRE(graph_checks::lazy_traversals());
}

namespace graph_components_checks {
	constexpr bool hand_bowtie() {
		atl::graph::block_decomposition const bowtie{{atl::graph::none, 0, 0, 1, 1}, {0, 2}};
		auto const cut = bowtie.cut_vertex_mask();
		auto const groups = bowtie.groups();
		return bowtie.count() == 2 && cut == std::vector<atl::uint8>{0, 0, 1, 0, 0}
			&& groups[1] == std::vector<atl::graph::vertex_id>{2, 3, 4} && bowtie.grouped().count() == 2
			&& bowtie.grouped()[1].size() == 3 && bowtie.grouped()[1][0] == 2;
	}

	constexpr bool computed_bowtie() {
		std::vector<std::vector<atl::graph::vertex_id>> const g{{1, 2}, {0, 2}, {0, 1, 3, 4}, {2, 4}, {2, 3}};
		auto const r = atl::graph::biconnected_components(g);
		return r.count() == 2 && r.cut_vertex_mask() == std::vector<atl::uint8>{0, 0, 1, 0, 0}
			&& r.edge_block(0, 1) == r.edge_block(1, 2) && r.edge_block(2, 3) == r.edge_block(3, 4)
			&& r.edge_block(0, 2) != r.edge_block(2, 4);
	}

	constexpr bool directed_components() {
		std::vector<std::vector<atl::graph::vertex_id>> const rows{{1}, {0, 2}, {3}, {2}, {}};
		auto const result = atl::graph::scc(rows);
		return result.count() == 3 && result.same(0, 1) && result.same(2, 3)
			&& !result.same(1, 2) && result.component[1] < result.component[2];
	}

	constexpr bool bridge_components() {
		std::vector<std::vector<atl::graph::vertex_id>> const rows{{1, 1}, {0, 0, 2}, {1}, {3}};
		auto const result = atl::graph::two_edge_connected_components(rows);
		return result.count() == 3 && result.same(0, 1) && !result.same(1, 2)
			&& result.component[1] < result.component[2];
	}

	constexpr bool empty_components() {
		std::array<std::array<int, 0>, 0> const rows{};
		auto const blocks = atl::graph::biconnected_components(rows);
		return atl::graph::scc(rows).count() == 0 && atl::graph::two_edge_connected_components(rows).count() == 0
			&& blocks.count() == 0 && blocks.groups().empty() && blocks.grouped().count() == 0
			&& blocks.cut_vertex_mask().empty();
	}

	constexpr bool paired_boundaries() {
		std::vector<std::vector<atl::graph::vertex_id>> const rows{{1, 2}, {0, 2}, {0, 1, 3, 4}, {2, 4}, {2, 3}};
		auto const result = atl::experimental::biconnected_components_by_paired_boundaries(rows);
		return result.count() == 2 && result.cut_vertex_mask() == std::vector<atl::uint8>{0, 0, 1, 0, 0}
			&& result.edge_block(0, 1) != result.edge_block(2, 3);
	}

	static_assert(paired_boundaries());
	static_assert(hand_bowtie());
	static_assert(computed_bowtie());
	static_assert(directed_components());
	static_assert(bridge_components());
	static_assert(empty_components());
}

TEST_CASE("graph components and block materialization constant-evaluate", "[graph][constexpr]") {
	REQUIRE(graph_components_checks::paired_boundaries());
	REQUIRE(graph_components_checks::hand_bowtie());
	REQUIRE(graph_components_checks::computed_bowtie());
	REQUIRE(graph_components_checks::directed_components());
	REQUIRE(graph_components_checks::bridge_components());
	REQUIRE(graph_components_checks::empty_components());
}

namespace graph_tree_checks {
	using namespace atl::graph;

	constexpr bool parent_queries() {
		lca anc(std::array<atl::uint32, 6>{none, 0, 0, 1, 1, 2});
		int segments = 0;
		anc.path(3, 5, [&](atl::uint32, atl::uint32, bool) { ++segments; });
		return anc(3, 4) == 1 && anc(3, 5) == 0 && anc.distance(3, 5) == 4 && anc.is_ancestor(0, 5)
			&& !anc.is_ancestor(1, 5) && anc.ancestor(3, 2) == 0 && anc.ancestor(3, 3) == none
			&& anc.path_vertex(3, 5, 2) == 0 && anc.parent(4) == 1 && anc.root(5) == 0 && anc.depth(3) == 2
			&& anc.exit_time(1) - anc.entry_time(1) == 3 && anc.order().size() == anc.size() && segments >= 2;
	}

	constexpr bool virtual_queries() {
		lca const anc(std::array<atl::uint32, 6>{none, 0, 0, 1, 1, 2});
		auto const vt = virtual_tree_of(anc, std::array<atl::uint32, 3>{4, 5, 3});
		std::vector<atl::uint32> many(40, 4);
		many[7] = 3;
		virtual_tree refilled;
		virtual_tree_of(anc, many, refilled);
		return vt.vertices == std::vector<atl::uint32>{0, 1, 3, 4, 5}
			&& vt.parent == std::vector<atl::uint32>{none, 0, 1, 1, 0}
			&& refilled.vertices == std::vector<atl::uint32>{1, 3, 4}
			&& refilled.parent == std::vector<atl::uint32>{none, 0, 0};
	}

	constexpr bool virtual_policies() {
		namespace ex = atl::experimental;
		lca const anc(std::array<atl::uint32, 6>{none, 0, 0, 1, 1, 2});
		ex::lca_by_entry_sparse_table const sparse(anc);
		std::array<atl::uint32, 3> const keys{4, 5, 3};
		ex::virtual_tree_workspace<atl::uint64> wide;
		ex::virtual_tree_workspace<atl::uint32> narrow;
		auto const want = virtual_tree_of(anc, keys);
		virtual_tree out;
		bool ok = true;
		auto const same = [&] { return out.vertices == want.vertices && out.parent == want.parent; };
		ex::virtual_tree_by_policy<ex::virtual_tree_algorithm::stack_sweep>(ex::packed_entry_keys{anc}, std::array<atl::uint32, 3>{4, 5, 3}, wide, out);
		ok = ok && same();
		ex::virtual_tree_by_policy<ex::virtual_tree_algorithm::two_sort>(ex::packed_entry_keys{anc}, keys, wide, out);
		ok = ok && same();
		ex::virtual_tree_by_policy<ex::virtual_tree_algorithm::stack_sweep>(ex::projected_keys{anc}, keys, narrow, out);
		ok = ok && same();
		ex::virtual_tree_by_policy<ex::virtual_tree_algorithm::two_sort>(ex::entry_keys{anc}, keys, narrow, out, true);
		ok = ok && same();
		ex::virtual_tree_by_policy<ex::virtual_tree_algorithm::stack_sweep>(ex::sparse_entry_keys{sparse}, keys, narrow, out);
		ok = ok && same();
		ex::virtual_tree_by_policy<ex::virtual_tree_algorithm::two_sort>(ex::sparse_entry_keys{sparse}, keys, narrow, out);
		ok = ok && same();
		ex::virtual_tree_by_depth(anc, keys, wide, out);
		ok = ok && same();
		ex::virtual_tree_by_raw_array(anc, keys, wide, out);
		return ok && same();
	}

	constexpr bool virtual_sorts() {
		namespace ex = atl::experimental;
		lca const anc(std::array<atl::uint32, 6>{none, 0, 0, 1, 1, 2});
		std::vector<atl::uint32> many(40, 4);
		many[7] = 3;
		many[9] = 5;
		auto const want = virtual_tree_of(anc, many);
		atl::graph::detail::virtual_tree_scratch scratch;
		virtual_tree dispatch, counting;
		ex::virtual_tree_by_sort(anc, many, ex::virtual_tree_sort::counting_dispatch, scratch, dispatch);
		ex::virtual_tree_by_sort(anc, many, ex::virtual_tree_sort::counting, scratch, counting);
		return dispatch.vertices == want.vertices && dispatch.parent == want.parent
			&& counting.vertices == want.vertices && counting.parent == want.parent;
	}

	constexpr bool separators() {
		std::vector<std::vector<vertex_id>> const rows{{1}, {0, 2}, {1, 3}, {2, 4}, {3}};
		auto const parent = centroid_decomposition(rows);
		std::size_t total = 0;
		bool is_parent_first = true;
		centroid_visit(rows, [&](vertex_id, vertex_id, std::span<centroid_vertex const> comp) {
			total += comp.size();
			for (std::size_t i = 1; i < comp.size(); ++i) {
				bool is_seen = false;
				for (std::size_t k = 0; k < i; ++k) is_seen = is_seen || comp[k].vertex == comp[i].parent;
				is_parent_first = is_parent_first && is_seen;
			}
		});
		centroid_visit(rows, [&](vertex_id, vertex_id, std::span<centroid_vertex const> comp, std::span<vertex_id const> edges) {
			is_parent_first = is_parent_first && comp.size() == edges.size();
		});
		int tops = 0;
		for (auto const p : parent) tops += p == none;
		return tops == 1 && total <= 5 * 3 && is_parent_first
			&& parent == atl::experimental::centroid_decomposition_by_component_sweep(rows);
	}

	constexpr bool graph_queries() {
		std::vector<std::vector<vertex_id>> const rows{{1, 2}, {0, 3, 4}, {0, 5}, {1}, {1}, {2}};
		lca const rooted(rows, 0), rootless(rows);
		std::array<std::pair<vertex_id, vertex_id>, 3> const queries{{{3, 4}, {3, 5}, {5, 5}}};
		auto const offline = offline_lca(rows, 0, queries);
		auto const forest = offline_lca(rows, queries);
		return rooted(3, 4) == 1 && rootless(3, 5) == 0 && offline == std::vector<vertex_id>{1, 0, 5}
			&& forest == offline;
	}

	constexpr bool bounded_degree() {
		std::vector<std::vector<vertex_id>> star(6);
		for (vertex_id v = 1; v < 6; ++v) {
			star[0].push_back(v);
			star[v].push_back(0);
		}
		auto const tree = ternarization_of(star, 0);
		return tree.root == 0 && tree.tree.vertex_count() < 12 && tree.tree.out_degree(0) == 2 && tree.real_count == 6
			&& tree.is_real(3) && !tree.is_real(vertex_id(tree.tree.vertex_count() - 1));
	}

	constexpr bool balanced_pairs() {
		std::vector<std::vector<vertex_id>> const rows{{1}, {0, 2}, {1, 3}, {2, 4}, {3}};
		std::size_t pairs = 0, weighted_pairs = 0, split_pairs = 0;
		centroid_bisect(rows, [&](vertex_id, std::span<visit const> a, std::span<visit const> b) {
			pairs += a.size() * b.size();
		});
		centroid_bisect(rows, [&](vertex_id, std::span<visit const> a, std::span<visit const> b,
								  std::span<vertex_id const> ea, std::span<vertex_id const> eb) {
			weighted_pairs += a.size() * b.size() * (ea.size() == a.size() && eb.size() == b.size());
		});
		atl::experimental::centroid_bisect_by_single_span(rows, [&](vertex_id, std::span<visit const> comp, std::size_t n0) {
			split_pairs += n0 * (comp.size() - 1 - n0);
		});
		return pairs == 6 && weighted_pairs == 6 && split_pairs == 6;
	}

	static_assert(parent_queries());
	static_assert(virtual_queries());
	static_assert(virtual_policies());
	static_assert(virtual_sorts());
	static_assert(separators());
	static_assert(graph_queries());
	static_assert(bounded_degree());
	static_assert(balanced_pairs());
}

TEST_CASE("tree algorithms evaluate the same contracts at compile time and runtime", "[graph][constexpr]") {
	REQUIRE(graph_tree_checks::parent_queries());
	REQUIRE(graph_tree_checks::virtual_queries());
	REQUIRE(graph_tree_checks::virtual_policies());
	REQUIRE(graph_tree_checks::virtual_sorts());
	REQUIRE(graph_tree_checks::separators());
	REQUIRE(graph_tree_checks::graph_queries());
	REQUIRE(graph_tree_checks::bounded_degree());
	REQUIRE(graph_tree_checks::balanced_pairs());
}

namespace graph_tree_checks {
	constexpr bool experimental_queries() {
		namespace ex = atl::experimental;
		std::vector<std::vector<vertex_id>> const rows{{1, 2}, {0, 3, 4}, {0, 5}, {1}, {1}, {2}};
		ex::lca_by_binary_lifting const lifting(rows, 0);
		ex::lca_by_head_jumping const heads(rows, 0);
		ex::lca_by_euler_sparse_table const euler(rows, 0);
		ex::lca_by_euler_direct_rmq const euler_rmq(rows, 0);
		ex::lca_by_preorder_sparse_table const preorder(rows, 0);
		ex::lca_by_preorder_direct_rmq const preorder_rmq(rows, 0);
		ex::lca_by_adjacency_preorder const labels(rows, 0);
		ex::lca_by_farach_colton_bender const blocks(rows, 0);
		std::size_t visits = 0;
		auto const event = [&](vertex_id, vertex_id, std::span<centroid_vertex const>) { ++visits; };
		ex::centroid_visit_by_subtree_sizes(rows, event);
		ex::centroid_visit_by_reused_sizes(rows, event);
		return lifting(3, 4) == 1 && heads(3, 4) == 1 && euler(3, 4) == 1 && euler_rmq(3, 4) == 1
			&& preorder(3, 4) == 1 && preorder_rmq(3, 4) == 1 && labels(3, 4) == 1 && blocks(3, 4) == 1
			&& lifting.depth(5) == 2 && heads.depth(5) == 2 && euler.depth(5) == 2 && euler_rmq.depth(5) == 2
			&& preorder.depth(5) == 2 && preorder_rmq.depth(5) == 2 && labels.depth(5) == 2 && blocks.depth(5) == 2
			&& visits == 12;
	}

	static_assert(experimental_queries());
}

TEST_CASE("experimental tree engines support constant evaluation", "[graph][constexpr]") {
	REQUIRE(graph_tree_checks::experimental_queries());
}

namespace graph_connectivity_checks {
	constexpr bool parity_rollback() {
		atl::experimental::parity_state state{decltype(atl::experimental::parity_state::sides)(3)};
		auto const token = state.sides.time();
		state.sides.unite(0, 1, 1);
		state.sides.unite(1, 2, 1);
		if (state.sides.unite(0, 2, 1) == atl::unite_outcome::contradicts) ++state.odd_cycles;
		if (state.odd_cycles != 1 || state.sides.difference(0, 2) != 0) return false;
		state.sides.rollback(token);
		state.odd_cycles = 0;
		return state.sides.count() == 3 && state.odd_cycles == 0;
	}

	constexpr atl::experimental::graph_event addition{atl::event_kind::add, {1, 2}};
	static_assert(addition.key.first == 1 && addition.key.second == 2);
	static_assert(parity_rollback());
}

TEST_CASE("offline connectivity constexpr event and parity state", "[graph][constexpr]") {
	CHECK(graph_connectivity_checks::parity_rollback());
}
