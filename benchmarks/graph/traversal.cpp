/** @file
 * @brief Full-traversal throughput: raw-array loops, library lazy BFS/DFS views,
 * eager BFS/DFS events, BFS by allocated seeds, BFS by record queue, byte versus
 * bit markers, and implicit versus materialized grids. Timed loops consume the
 * traversal; graph construction is excluded. Shapes include random digraphs,
 * a million-vertex path, a 2000x2000 grid, and a million tiny-component runs.
 * Intel Xeon Platinum 8358, GCC 15.2.0, RelWithDebInfo, -O2 -g -DNDEBUG
 * -std=c++23, also -march=native, 2026-10-15: n=1e6,m=4e6 BFS library view
 * 61.0/60.3 ms, eager 54.4/50.8 ms, raw arrays 61.0/60.5 ms.
 * Full tables: docs/graph.md.
 */
#include <atl/ds/static_vector.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

/** @brief Traversal contenders with allocated seeds and complete queue records. */
namespace atl::graph::bench_methods {
	/** @brief Lazy BFS retaining heap-allocated source and pending lists, including a single source. */
	template<detail::scannable_graph Graph, typename Marker = void>
		requires(std::is_void_v<Marker> || visit_marker<Marker>)
	class bfs_by_allocated_seeds_view : public std::ranges::view_interface<bfs_by_allocated_seeds_view<Graph, Marker>> {
		static constexpr bool owns_marks = std::is_void_v<Marker>;
		using marker_param = std::conditional_t<owns_marks, detail::nothing, Marker>;

		Graph const *g_;
		std::vector<vertex_id> sources_;
		[[no_unique_address]] std::conditional_t<owns_marks, detail::nothing, Marker *> marker_{};

	public:
		class iterator {
			Graph const *g_ = nullptr;
			std::vector<visit> order_;
			std::vector<vertex_id> pending_;
			std::size_t scan_ = 0;
			detail::edge_scan<Graph> edges_;
			[[no_unique_address]] detail::marks_of<Marker> marks_;
			bool is_done_ = false;

			constexpr void seed(std::vector<vertex_id> const &sources) {
				atl_assert(graph::vertex_count(*g_) <= none);
				for ([[maybe_unused]] vertex_id const s : sources) atl_assert(s < graph::vertex_count(*g_));
				pending_.assign(sources.rbegin(), sources.rend());
				while (!pending_.empty()) {
					vertex_id const s = pending_.back();
					if (marks_.get().try_visit(s)) break;
					pending_.pop_back();
				}
				if (pending_.empty()) {
					is_done_ = true;
					return;
				}
				order_.push_back({pending_.back(), pending_.back(), 0});
				pending_.pop_back();
				edges_ = detail::edge_scan<Graph>(*g_, order_.front().vertex);
			}

		public:
			using value_type = visit;
			using difference_type = std::ptrdiff_t;
			using iterator_concept = std::input_iterator_tag;

			iterator() = default;

			constexpr iterator(Graph const &g, std::vector<vertex_id> const &sources)
				requires(owns_marks)
				: g_(&g), marks_{.owned_marker = dense_marker(graph::vertex_count(g)), .external_marker = {}} {
				seed(sources);
			}

			constexpr iterator(Graph const &g, std::vector<vertex_id> const &sources,
							   bfs_by_allocated_seeds_view::marker_param &marker)
				requires(!owns_marks)
				: g_(&g), marks_{.owned_marker = {}, .external_marker = &marker} {
				seed(sources);
			}

			constexpr visit const &operator*() const { return order_.back(); }

			constexpr iterator &operator++() {
				while (!pending_.empty()) [[unlikely]] { // remaining seeds join at depth 0
					vertex_id const s = pending_.back();
					pending_.pop_back();
					if (marks_.get().try_visit(s)) {
						order_.push_back({s, s, 0});
						return *this;
					}
				}
				for (;;) {
					while (edges_.exhausted()) {
						if (++scan_ == order_.size()) {
							is_done_ = true;
							return *this;
						}
						edges_ = detail::edge_scan<Graph>(*g_, order_[scan_].vertex);
					}
					vertex_id const v = graph::target(edges_.current());
					edges_.advance();
					if (marks_.get().try_visit(v)) {
						order_.push_back({v, order_[scan_].vertex, order_[scan_].depth + 1});
						return *this;
					}
				}
			}

			constexpr void operator++(int) { ++*this; }

			constexpr bool operator==(std::default_sentinel_t) const { return is_done_; }
		};

		constexpr bfs_by_allocated_seeds_view(Graph const &g, vertex_id source)
			requires(owns_marks)
			: g_(&g), sources_(1, source) {}

		constexpr bfs_by_allocated_seeds_view(Graph const &g, vertex_id source, marker_param &marker)
			requires(!owns_marks)
			: g_(&g), sources_(1, source), marker_(&marker) {}

		template<typename Sources>
			requires vertex_range<Sources const>
		constexpr bfs_by_allocated_seeds_view(Graph const &g, Sources const &sources)
			requires(owns_marks)
			: g_(&g) {
			for (auto const s : sources) sources_.push_back(vertex_id(s));
		}

		template<typename Sources>
			requires vertex_range<Sources const>
		constexpr bfs_by_allocated_seeds_view(Graph const &g, Sources const &sources, marker_param &marker)
			requires(!owns_marks)
			: g_(&g), marker_(&marker) {
			for (auto const s : sources) sources_.push_back(vertex_id(s));
		}

		constexpr iterator begin() const {
			if constexpr (owns_marks)
				return {*g_, sources_};
			else
				return {*g_, sources_, *marker_};
		}

		constexpr std::default_sentinel_t end() const { return {}; }
	};

	/** @brief Adapt allocated-seed BFS to an owned dense marker. */
	template<detail::scannable_graph Graph>
	constexpr auto bfs_by_allocated_seeds(Graph const &g, vertex_id source) {
		return bfs_by_allocated_seeds_view<Graph>(g, source);
	}

	/** @brief Adapt allocated-seed BFS to persistent marks. */
	template<detail::scannable_graph Graph, visit_marker Marker>
	constexpr auto bfs_by_allocated_seeds(Graph const &g, vertex_id source, Marker &marker) {
		return bfs_by_allocated_seeds_view<Graph, Marker>(g, source, marker);
	}

	template<detail::scannable_graph Graph, visit_marker Marker, bfs_step_event Step>
	constexpr void bfs_run_by_record_queue(Graph const &g, std::vector<visit> &queue, Marker &marks, Step &step) {
		for (std::size_t head = 0; head < queue.size(); ++head) {
			auto const [u, parent, depth] = queue[head];
			for (auto const &e : graph::out_edges(g, u))
				if (vertex_id const v = graph::target(e); marks.try_visit(v)) {
					queue.push_back({v, u, depth + 1});
					step(v, u, depth + 1);
				}
		}
	}

	/** @brief Eager BFS storing vertex, parent and depth together in every queue slot. */
	template<detail::scannable_graph Graph, visit_marker Marker, bfs_step_event Step>
	constexpr void bfs_visit_by_record_queue(Graph const &g, vertex_id source, Marker &marker, Step step) {
		atl_assert(graph::vertex_count(g) <= none);
		atl_assert(source < graph::vertex_count(g));
		std::vector<visit> queue;
		if (!marker.try_visit(source)) return;
		queue.push_back({source, source, 0});
		step(source, source, 0);
		bfs_run_by_record_queue(g, queue, marker, step);
	}
}

namespace {
	template<typename Run>
		requires std::is_invocable_r_v<atl::uint64, Run &>
	void measure(std::string name, atl::uint64 expected, Run run) {
		REQUIRE(std::invoke(run) == expected);
		BENCHMARK(std::move(name)) { return std::invoke(run); };
	}
}

namespace g = atl::graph;
using atl::uint32;
using atl::uint64;
using atl::uint8;

namespace {
	/** @brief One byte per vertex; the library dense marker stores one bit. */
	class visit_marker_by_bytes {
		std::vector<uint8> seen_;

	public:
		explicit visit_marker_by_bytes(std::size_t n) : seen_(n, 0) {}

		bool try_visit(g::vertex_id v) {
			if (seen_[v]) return false;
			seen_[v] = 1;
			return true;
		}
	};

	/** @brief A connected-ish random sparse digraph: a random tree plus random arcs up to @p m. */
	g::csr<g::vertex_id> random_digraph(std::size_t n, std::size_t m) {
		std::mt19937 rng(7);
		std::vector<g::arc<>> arcs;
		arcs.reserve(m);
		for (std::size_t i = 1; i < n; ++i) arcs.push_back({uint32(rng() % i), uint32(i)});
		while (arcs.size() < m) arcs.push_back({uint32(rng() % n), uint32(rng() % n)});
		return g::csr<g::vertex_id>(n, arcs);
	}

	/** @brief Hand-rolled BFS over raw csr spans: the baseline every row is held to. */
	uint64 bfs_by_raw_arrays(g::csr<g::vertex_id> const &h) {
		std::size_t const n = h.vertex_count();
		std::vector<uint32> dist(n, uint32(-1)), queue;
		queue.reserve(n);
		dist[0] = 0;
		queue.push_back(0);
		for (std::size_t head = 0; head < queue.size(); ++head) {
			uint32 const u = queue[head];
			for (uint32 const v : h.out_edges(u))
				if (dist[v] == uint32(-1)) {
					dist[v] = dist[u] + 1;
					queue.push_back(v);
				}
		}
		uint64 acc = 0;
		for (uint32 const d : dist) acc += d != uint32(-1) ? d : 0;
		return acc;
	}

	/** @brief Compare level-boundary and record queues with a raw-array BFS on one graph. */
	void bfs_visit_rows(std::string const &name, g::csr<g::vertex_id> const &h) {
		std::size_t const n = h.vertex_count();
		measure(name + " | bfs by raw arrays", bfs_by_raw_arrays(h), [&] { return bfs_by_raw_arrays(h); });
		measure(name + " | bfs_visit library", bfs_by_raw_arrays(h), [&] {
			uint64 acc = 0;
			g::bfs_visit(h, 0, [&](g::vertex_id, g::vertex_id, uint32 d) { acc += d; });
			return acc;
		});
		measure(name + " | bfs_visit library, caller marker, no reserve", bfs_by_raw_arrays(h), [&] {
			uint64 acc = 0;
			g::dense_marker marks(n);
			g::bfs_visit(h, 0, marks, [&](g::vertex_id, g::vertex_id, uint32 d) { acc += d; });
			return acc;
		});
		measure(name + " | bfs_visit by record queue", bfs_by_raw_arrays(h), [&] {
			uint64 acc = 0;
			g::dense_marker marks(n);
			g::bench_methods::bfs_visit_by_record_queue(h, 0, marks, [&](g::vertex_id, g::vertex_id, uint32 d) { acc += d; });
			return acc;
		});
		measure(name + " | bfs view library", bfs_by_raw_arrays(h), [&] {
			uint64 acc = 0;
			for (auto const &[v, parent, depth] : g::bfs(h, 0)) acc += depth;
			return acc;
		});
	}

	/** @brief The marker race: dense_marker (bit flags) vs byte flags under BFS and DFS. */
	void marker_rows(std::string const &name, g::csr<g::vertex_id> const &h) {
		std::size_t const n = h.vertex_count();
		measure(name + " | bfs_visit library, bit marker", bfs_by_raw_arrays(h), [&] {
			uint64 acc = 0;
			g::dense_marker marks(n);
			g::bfs_visit(h, 0, marks, [&](g::vertex_id, g::vertex_id, uint32 d) { acc += d; });
			return acc;
		});
		measure(name + " | bfs_visit by byte marker", bfs_by_raw_arrays(h), [&] {
			uint64 acc = 0;
			visit_marker_by_bytes marks(n);
			g::bfs_visit(h, 0, marks, [&](g::vertex_id, g::vertex_id, uint32 d) { acc += d; });
			return acc;
		});
		measure(name + " | dfs view library, bit marker", uint64(n) * (n - 1) / 2, [&] {
			uint64 acc = 0;
			g::dense_marker marks(n);
			for (auto const &[v, parent, depth] : g::dfs(h, 0, marks)) acc += v;
			return acc;
		});
		measure(name + " | dfs by byte marker", uint64(n) * (n - 1) / 2, [&] {
			uint64 acc = 0;
			visit_marker_by_bytes marks(n);
			for (auto const &[v, parent, depth] : g::dfs(h, 0, marks)) acc += v;
			return acc;
		});
	}
}

TEST_CASE("graph traversal benchmarks", "[graph]") {
	constexpr std::size_t n = 1'000'000, m = 4'000'000;
	std::mt19937 rng(1);
	std::vector<g::arc<>> arcs;
	arcs.reserve(m);
	for (std::size_t i = 1; i < n; ++i) arcs.push_back({uint32(rng() % i), uint32(i)}); // connected
	while (arcs.size() < m) arcs.push_back({uint32(rng() % n), uint32(rng() % n)});
	g::csr const compressed(n, arcs);
	std::vector<std::vector<uint32>> adj(n);
	for (auto const &a : arcs) adj[a.from].push_back(a.to);

	// raw CSR arrays for the hand-rolled baselines
	std::vector<uint32> off(n + 1, 0), heads(m);
	for (auto const &a : arcs) ++off[a.from + 1];
	for (std::size_t i = 0; i < n; ++i) off[i + 1] += off[i];
	{
		auto cursor = off;
		for (auto const &a : arcs) heads[cursor[a.from]++] = a.to;
	}

	uint64 const bfs_expected = bfs_by_raw_arrays(compressed);
	uint64 const vertex_sum = uint64(n) * (n - 1) / 2;
	uint64 euler_expected = n;
	std::size_t discovery = 0;
	for (auto const &node : g::dfs(compressed, 0)) {
		if (node.vertex == n - 1) euler_expected += discovery;
		++discovery;
	}

	measure("bfs by raw arrays", bfs_expected, [&] {
		std::vector<uint32> dist(n, uint32(-1)), queue;
		queue.reserve(n);
		dist[0] = 0;
		queue.push_back(0);
		for (std::size_t head = 0; head < queue.size(); ++head) {
			uint32 const u = queue[head];
			for (uint32 j = off[u]; j < off[u + 1]; ++j)
				if (uint32 const v = heads[j]; dist[v] == uint32(-1)) {
					dist[v] = dist[u] + 1;
					queue.push_back(v);
				}
		}
		uint64 acc = 0;
		for (uint32 const d : dist) acc += d != uint32(-1) ? d : 0;
		return acc;
	});

	measure("bfs view over csr", bfs_expected, [&] {
		uint64 acc = 0;
		for (auto const &[v, parent, depth] : g::bfs(compressed, 0)) acc += depth;
		return acc;
	});

	measure("bfs by allocated seeds over csr", bfs_expected, [&] {
		uint64 acc = 0;
		for (auto const &[v, parent, depth] : g::bench_methods::bfs_by_allocated_seeds(compressed, 0)) acc += depth;
		return acc;
	});

	measure("bfs_visit over csr", bfs_expected, [&] {
		uint64 acc = 0;
		g::bfs_visit(compressed, 0, [&](g::vertex_id, g::vertex_id, uint32 depth) { acc += depth; });
		return acc;
	});

	measure("bfs_visit over csr, caller's marker (no reserve)", bfs_expected, [&] {
		uint64 acc = 0;
		g::dense_marker marks(n);
		g::bfs_visit(compressed, 0, marks, [&](g::vertex_id, g::vertex_id, uint32 depth) { acc += depth; });
		return acc;
	});

	measure("bfs_visit by record queue over csr", bfs_expected, [&] {
		uint64 acc = 0;
		g::dense_marker marks(n);
		g::bench_methods::bfs_visit_by_record_queue(compressed, 0, marks, [&](g::vertex_id, g::vertex_id, uint32 depth) { acc += depth; });
		return acc;
	});

	measure("bfs view over vector<vector>", bfs_expected, [&] {
		uint64 acc = 0;
		for (auto const &[v, parent, depth] : g::bfs(adj, 0)) acc += depth;
		return acc;
	});

	measure("dfs by raw arrays", vertex_sum, [&] {
		std::vector<uint8> seen(n, 0);
		std::vector<uint32> stack;
		stack.reserve(n);
		stack.push_back(0);
		uint64 acc = 0;
		while (!stack.empty()) {
			uint32 const u = stack.back();
			stack.pop_back();
			if (seen[u]) continue;
			seen[u] = 1;
			acc += u;
			for (uint32 j = off[u]; j < off[u + 1]; ++j)
				if (!seen[heads[j]]) stack.push_back(heads[j]);
		}
		return acc;
	});

	measure("dfs view over csr", vertex_sum, [&] {
		uint64 acc = 0;
		for (auto const &[v, parent, depth] : g::dfs(compressed, 0)) acc += v;
		return acc;
	});

	measure("dfs_visit euler intervals over csr", euler_expected, [&] {
		std::vector<uint32> pos(n), sub_end(n);
		uint32 t = 0;
		g::dfs_visit(
			compressed, 0, [&](g::vertex_id v, g::vertex_id, uint32, uint32) { pos[v] = t++; },
			g::skip, [&](g::vertex_id v, g::vertex_id) { sub_end[v] = t; });
		return uint64(pos[n - 1]) + sub_end[0];
	});
}

TEST_CASE("implicit vs materialized grid", "[graph]") {
	constexpr int W = 2000, H = 2000;
	auto const moves = [](g::vertex_id u) {
		atl::static_vector<g::vertex_id, 4> out;
		int const x = int(u) % W, y = int(u) / W;
		if (x > 0) out.push_back(u - 1);
		if (x + 1 < W) out.push_back(u + 1);
		if (y > 0) out.push_back(u - W);
		if (y + 1 < H) out.push_back(u + W);
		return out;
	};
	auto const grid = g::implicit_graph(std::size_t(W) * H, moves);

	std::vector<g::arc<>> arcs;
	arcs.reserve(std::size_t(4) * W * H);
	for (g::vertex_id u = 0; u < g::vertex_id(W * H); ++u)
		for (auto const v : moves(u)) arcs.push_back({u, v});
	g::csr const materialized(std::size_t(W) * H, arcs);

	uint64 const grid_expected = uint64(W) * H * (W + H - 2) / 2;

	measure("bfs over implicit grid (no graph built)", grid_expected, [&] {
		uint64 acc = 0;
		for (auto const &[v, parent, depth] : g::bfs(grid, 0)) acc += depth;
		return acc;
	});

	measure("bfs over materialized csr grid", grid_expected, [&] {
		uint64 acc = 0;
		for (auto const &[v, parent, depth] : g::bfs(materialized, 0)) acc += depth;
		return acc;
	});
}

TEST_CASE("bfs_visit queue", "[graph]") {
	{
		std::size_t const n = 1'000'000;
		std::vector<g::arc<>> arcs;
		for (uint32 v = 1; v < n; ++v) arcs.push_back({v - 1, v});
		bfs_visit_rows("path n=1e6", g::csr<g::vertex_id>(n, arcs));
	}
	bfs_visit_rows("random n=4e6 m=1.6e7", random_digraph(4'000'000, 16'000'000));
}

TEST_CASE("bfs over tiny components", "[graph]") {
	std::size_t const n = 1'000'000;
	std::mt19937 rng(3);
	std::vector<g::arc<>> arcs;
	for (uint32 start = 0; start < n;) {
		uint32 const size = std::min<uint32>(uint32(1 + rng() % 4), uint32(n - start));
		for (uint32 v = start + 1; v < start + size; ++v) {
			arcs.push_back({v - 1, v});
			arcs.push_back({v, v - 1});
		}
		start += size;
	}
	g::csr<g::vertex_id> const h(n, arcs);
	auto bfs_by_reused_queue = [&] {
		g::epoch_marker marks(n);
		std::vector<uint32> queue, depth(n);
		uint64 acc = 0;
		for (uint32 s = 0; s < n; ++s) {
			marks.next_run();
			queue.clear();
			marks.try_visit(s);
			queue.push_back(s);
			depth[s] = 0;
			for (std::size_t head = 0; head < queue.size(); ++head) {
				uint32 const u = queue[head];
				acc += depth[u];
				for (uint32 const v : h.out_edges(u))
					if (marks.try_visit(v)) {
						depth[v] = depth[u] + 1;
						queue.push_back(v);
					}
			}
		}
		return acc;
	};
	uint64 const tiny_expected = bfs_by_reused_queue();

	measure("tiny components | bfs view library", tiny_expected, [&] {
		g::epoch_marker marks(n);
		uint64 acc = 0;
		for (uint32 s = 0; s < n; ++s) {
			marks.next_run();
			for (auto const &[v, parent, depth] : g::bfs(h, s, marks)) acc += depth;
		}
		return acc;
	});
	measure("tiny components | bfs by allocated seeds", tiny_expected, [&] {
		g::epoch_marker marks(n);
		uint64 acc = 0;
		for (uint32 s = 0; s < n; ++s) {
			marks.next_run();
			for (auto const &[v, parent, depth] : g::bench_methods::bfs_by_allocated_seeds(h, s, marks)) acc += depth;
		}
		return acc;
	});
	measure("tiny components | bfs_visit library", tiny_expected, [&] {
		g::epoch_marker marks(n);
		uint64 acc = 0;
		for (uint32 s = 0; s < n; ++s) {
			marks.next_run();
			g::bfs_visit(h, s, marks, [&](g::vertex_id, g::vertex_id, uint32 d) { acc += d; });
		}
		return acc;
	});
	measure("tiny components | bfs_visit by record queue", tiny_expected, [&] {
		g::epoch_marker marks(n);
		uint64 acc = 0;
		for (uint32 s = 0; s < n; ++s) {
			marks.next_run();
			g::bench_methods::bfs_visit_by_record_queue(h, s, marks, [&](g::vertex_id, g::vertex_id, uint32 d) { acc += d; });
		}
		return acc;
	});
	measure("tiny components | bfs by reused queue", tiny_expected, [&] {
		g::epoch_marker marks(n);
		std::vector<uint32> queue, depth(n);
		uint64 acc = 0;
		for (uint32 s = 0; s < n; ++s) {
			marks.next_run();
			queue.clear();
			marks.try_visit(s);
			queue.push_back(s);
			depth[s] = 0;
			for (std::size_t head = 0; head < queue.size(); ++head) {
				uint32 const u = queue[head];
				acc += depth[u];
				for (uint32 const v : h.out_edges(u))
					if (marks.try_visit(v)) {
						depth[v] = depth[u] + 1;
						queue.push_back(v);
					}
			}
		}
		return acc;
	});
}

TEST_CASE("marker width", "[graph]") {
	marker_rows("n=1e5 m=4e5", random_digraph(100'000, 400'000));
	marker_rows("n=1e6 m=4e6", random_digraph(1'000'000, 4'000'000));
	marker_rows("n=4e6 m=1.6e7", random_digraph(4'000'000, 16'000'000));
}
