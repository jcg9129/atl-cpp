#pragma once
#include <atl/ds/flat_hash_set.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::graph {
	/**
	 * @brief One traversal step: the tree edge `parent -> vertex` at @p depth. The
	 * source visits itself (`parent == vertex`, depth 0).
	 */
	struct visit {
		vertex_id vertex; ///< Discovered vertex.
		vertex_id parent; ///< Discovery parent; the source is its own parent.
		uint32 depth;	  ///< Tree distance from the source.
	};

	/**
	 * @brief Visited-state customization for traversals: one primitive,
	 *        `try_visit(v)` — mark v and report whether it was new.
	 *
	 * Traversals allocate a @ref atl::graph::dense_marker per run by default; pass your
	 * own marker (by reference, outliving the run) to control the cost model:
	 * an @ref atl::graph::epoch_marker amortizes repeated small traversals over one big
	 * graph to O(reached vertices + scanned edges) each, a
	 * @ref atl::graph::hash_marker traverses implicit graphs with up to 2^32 - 1 vertices,
	 * too many to allocate densely.
	 */
	template<typename M>
	concept visit_marker = requires(M &m, vertex_id v) {
		{ m.try_visit(v) } -> std::convertible_to<bool>;
	};

	/**
	 * @brief Bit flags — the default marker, O(n / 8) storage, O(n / 64) reset.
	 * One bit per vertex reduces flag storage relative to byte marks; cache
	 * effects under random probes are measured in
	 * benchmarks/graph/traversal.cpp, "marker width".
	 */
	class dense_marker {
		std::vector<uint64> words_;
		std::size_t size_ = 0;

	public:
		/** @brief Allocate clear marks for @p n vertices. @complexity O(n / 64 + 1) */
		constexpr explicit dense_marker(std::size_t n) : words_(n / 64 + std::size_t(n % 64 != 0), 0), size_(n) {}

		/** @brief Mark @p v and return whether it was clear. @pre `v < n` (asserted). @complexity O(1) */
		constexpr bool try_visit(vertex_id v) {
			atl_assert(v < size_);
			uint64 &word = words_[v >> 6];
			uint64 const bit = uint64(1) << (v & 63);
			// Avoid dirtying a cache line on repeated probes.
			if (word & bit) return false;
			word |= bit;
			return true;
		}

		/** @brief Whether @p v is marked. @pre `v < n` (asserted). @complexity O(1) */
		[[nodiscard]] constexpr bool visited(vertex_id v) const {
			atl_assert(v < size_);
			return ((words_[v >> 6] >> (v & 63)) & 1) != 0;
		}

		/** @brief Clear every bit. @complexity O(n / 64 + 1) */
		constexpr void reset() { std::ranges::fill(words_, uint64(0)); }
	};

	/**
	 * @brief Generation-stamped flags: `next_run()` invalidates every mark in O(1),
	 * so repeated traversals over one graph cost O(reached vertices + scanned edges) each following the
	 * single O(n) allocation — the marker for small-side component races.
	 */
	class epoch_marker {
		std::vector<uint32> stamp_;
		uint32 epoch_ = 1;

	public:
		/** @brief Allocate clear stamps for @p n vertices. @complexity O(n) */
		constexpr explicit epoch_marker(std::size_t n) : stamp_(n, 0) {}

		/** @brief Mark @p v and return whether it was clear. @pre `v < n` (asserted). @complexity O(1) */
		constexpr bool try_visit(vertex_id v) {
			atl_assert(v < stamp_.size());
			if (stamp_[v] == epoch_) return false;
			stamp_[v] = epoch_;
			return true;
		}

		/** @brief Whether @p v is marked. @pre `v < n` (asserted). @complexity O(1) */
		[[nodiscard]] constexpr bool visited(vertex_id v) const {
			atl_assert(v < stamp_.size());
			return stamp_[v] == epoch_;
		}

		/** @brief Forget every mark. O(1) (O(n) once per ~4·10⁹ runs, on epoch wrap). */
		constexpr void next_run() {
			if (++epoch_ == 0) {
				std::ranges::fill(stamp_, uint32(0));
				epoch_ = 1;
			}
		}
	};

	/**
	 * @brief Hash-set marks: O(visited) memory, no dependence on the vertex count —
	 * for implicit graphs over domains up to the full `vertex_id` range
	 * (2^32 - 1 vertices), too large to allocate densely.
	 */
	class hash_marker {
		flat_hash_set<vertex_id> seen_;

	public:
		/** @brief Mark @p v and return whether it was new.
		 * @complexity Expected amortized O(1), worst-case O(marked vertices).
		 * @note Not constexpr (the hash table uses runtime allocation).
		 */
		bool try_visit(vertex_id v) { return seen_.insert(v); }

		/** @brief Whether @p v is marked. @complexity Expected O(1), worst-case O(marked vertices).
		 * @note Not constexpr (queries the runtime hash table).
		 */
		[[nodiscard]] bool visited(vertex_id v) const { return seen_.contains(v); }

		/** @brief Clear all marks. @complexity O(table capacity).
		 * @note Not constexpr (mutates the runtime hash table).
		 */
		void clear() { seen_.clear(); }
	};

	namespace detail {
		struct nothing {};

		template<adjacency_graph Graph>
		inline constexpr bool has_edges_by_reference =
			std::is_lvalue_reference_v<decltype(graph::out_edges(std::declval<Graph const &>(), vertex_id{}))>;

		/**
		 * @brief What the lazy views and the visit functions need beyond
		 *        adjacency_graph: reference-returning out_edges may be any
		 *        forward range (iterator positions are stable);
		 *        prvalue-returning out_edges must be random-access and sized
		 *        (positions are indices into the frame's own copy).
		 */
		template<typename Graph>
		concept scannable_graph =
			adjacency_graph<Graph>
			&& ((has_edges_by_reference<Graph> && std::ranges::forward_range<out_edges_t<Graph> const>)
				|| (std::ranges::random_access_range<out_edges_t<Graph> const>
					&& std::ranges::sized_range<out_edges_t<Graph> const>
					&& (has_edges_by_reference<Graph>
						|| (std::default_initializable<out_edges_t<Graph>>
							&& std::assignable_from<out_edges_t<Graph> &, out_edges_t<Graph>>))));

		/** @brief Random-access const rows support a compact resumable index. */
		template<adjacency_graph Graph>
		inline constexpr bool has_indexable_edges =
			std::ranges::random_access_range<out_edges_t<Graph> const> && std::ranges::sized_range<out_edges_t<Graph> const>;

		/**
		 * @brief What a paused scan must remember to resume — the part stored
		 *        in traversal frames, so it must survive its frame moving
		 *        (stack / queue reallocation).
		 *
		 * When `out_edges` returns a reference, its iterators point into the
		 * graph's own storage and stay valid whatever the frame does — so the
		 * position is the iterator itself (plus the running edge index for
		 * event coordinates) — or, when @p PrefersIndex and the range is
		 * random-access and sized (see `has_indexable_edges`), a bare
		 * index — and *any forward range* qualifies: `set`, `map`,
		 * `list` adjacency all work. When it returns a prvalue, the range must
		 * live in the frame and iterators into it would dangle on move
		 * (inline-storage ranges, self-referencing views) — the position is an
		 * index, and the range must be random-access + sized.
		 */
		template<scannable_graph Graph, bool PrefersIndex = false>
		struct scan_position {
			struct by_iterator {
				std::ranges::iterator_t<out_edges_t<Graph> const> it{};
				uint32 edge_index = 0;
			};

			using type = std::conditional_t<has_edges_by_reference<Graph> && !(PrefersIndex && has_indexable_edges<Graph>),
											by_iterator, uint32>;
		};

		template<scannable_graph Graph, bool PrefersIndex = false>
		using scan_position_t = typename scan_position<Graph, PrefersIndex>::type;

		/**
		 * @brief Resumable scan over one vertex's out-edges; construct fresh or
		 *        from a saved `scan_position_t`, read the position back with
		 *        `pos()`.
		 * @tparam PrefersIndex Resume reference-returning random-access ranges
		 *         from an index rather than an iterator (see
		 *         `has_indexable_edges`).
		 */
		template<scannable_graph Graph, bool PrefersIndex = false>
		struct edge_scan {
			using position = scan_position_t<Graph, PrefersIndex>;
			static constexpr bool is_by_reference = has_edges_by_reference<Graph>;
			static constexpr bool is_iterator_mode = is_by_reference && !(PrefersIndex && has_indexable_edges<Graph>);

			[[no_unique_address]] std::conditional_t<
				is_iterator_mode, out_edges_t<Graph> const *,
				std::conditional_t<is_by_reference, std::ranges::iterator_t<out_edges_t<Graph> const>, out_edges_t<Graph>>>
				edges{};
			[[no_unique_address]] std::conditional_t<is_iterator_mode, std::ranges::sentinel_t<out_edges_t<Graph> const>,
													 uint32>
				last{};
			position at{};

			edge_scan() = default;

			constexpr edge_scan(Graph const &g, vertex_id u) : edge_scan(g, u, position{}) {
				if constexpr (is_iterator_mode) at.it = std::ranges::begin(*edges);
			}

			constexpr edge_scan(Graph const &g, vertex_id u, position resume) : at(resume) {
				if constexpr (is_iterator_mode) {
					edges = &graph::out_edges(g, u);
					last = std::ranges::end(*edges);
				} else if constexpr (is_by_reference) {
					auto const &range = graph::out_edges(g, u);
					atl_assert(std::ranges::size(range) <= none);
					edges = std::ranges::begin(range);
					last = uint32(std::ranges::size(range));
				} else {
					edges = graph::out_edges(g, u);
					atl_assert(std::ranges::size(edges) <= none);
					last = uint32(std::ranges::size(edges));
				}
			}

			/**
			 * @brief The position a fresh scan of @p u starts from — for index mode
			 * this is free (no out_edges call).
			 */
			[[nodiscard]] static constexpr position initial(Graph const &g, vertex_id u) {
				if constexpr (is_iterator_mode)
					return {std::ranges::begin(graph::out_edges(g, u)), 0};
				else
					return {};
			}

			[[nodiscard]] constexpr bool exhausted() const {
				if constexpr (is_iterator_mode)
					return at.it == last;
				else
					return at == last;
			}

			/** @brief The edge at the scan position. @pre `!exhausted()` */
			[[nodiscard]] constexpr decltype(auto) current() const {
				if constexpr (is_iterator_mode)
					return *at.it;
				else if constexpr (is_by_reference)
					return edges[at];
				else
					return std::ranges::begin(edges)[at];
			}

			/** @brief The scan position's edge index (the `k` of event coordinates). */
			[[nodiscard]] constexpr uint32 index() const {
				if constexpr (is_iterator_mode)
					return at.edge_index;
				else
					return at;
			}

			constexpr void advance() {
				if constexpr (is_iterator_mode) {
					++at.it;
					++at.edge_index;
				} else {
					++at;
				}
			}

			[[nodiscard]] constexpr position const &pos() const { return at; }
		};

		/**
		 * @brief Marker storage for a traversal iterator: an owned dense_marker
		 *        when the caller passed none (Marker = void), else a pointer to
		 *        theirs.
		 */
		template<typename Marker>
			requires(std::is_void_v<Marker> || visit_marker<Marker>)
		struct marks_of {
			static constexpr bool owns_marker = std::is_void_v<Marker>;
			[[no_unique_address]] std::conditional_t<owns_marker, dense_marker, nothing> owned_marker;
			[[no_unique_address]] std::conditional_t<owns_marker, nothing, Marker *> external_marker;

			[[nodiscard]] constexpr auto &get() {
				if constexpr (owns_marker)
					return owned_marker;
				else
					return *external_marker;
			}
		};
	}

	/**
	 * @brief Breadth-first traversal as a lazy range: yields each reachable
	 *        vertex exactly once, in nondecreasing depth order, as a
	 *        @ref atl::graph::visit. For unweighted graphs `depth` is the shortest
	 *        distance; with several sources, the distance to the nearest one. With prepopulated
	 * external marks, these statements apply to the search with marked vertices blocked.
	 *
	 * `operator++` is **edge-granular**: it scans forward only until the next
	 * undiscovered vertex — O(K + reached vertices + scanned edges) over a full run,
	 * where K counts supplied sources, including duplicates (K = 1 for a single source).
	 * Breaking immediately after the k-th visit performs no further increment.
	 * Fully consuming `| std::views::take(k)` increments the underlying iterator
	 * after the k-th visit: it may scan through several adjacency rows until
	 * discovering and marking one additional vertex, or exhaust the traversal.
	 * For k = 0, begin() still initializes the traversal and may mark a source.
	 *
	 * The view refers to the graph. A C++23 range-for over `bfs(temporary_graph, s)`
	 * keeps that temporary graph alive through the loop. Storing the view beyond
	 * that full-expression leaves a dangling graph reference. Otherwise the graph
	 * must outlive use of the view and its iterators; adjacency order and storage
	 * stay unchanged. Dereferenced visits are invalidated by incrementing or
	 * destroying the iterator. Targets are in `[0, vertex_count(g))`, vertex
	 * count is at most `none`, and each row has at most `none` edges.
	 * Graph access is assumed O(1); regenerating rows adds its own cost.
	 * O(K + reached vertices) storage for copied sources, pending seeds and the
	 * queue, plus the marker; construction copies K ids in O(K) time. An owned
	 * dense marker adds O(n / 64 + 1) initialization time and words. No recursion.
	 * Single-pass input range; each `begin()` runs a traversal against the
	 * marker: the default is a fresh internal @ref atl::graph::dense_marker; an external
	 * @ref atl::graph::visit_marker (see `bfs(g, s, marker)`) is used as-is, so its
	 * already-visited vertices are silently skipped and the marks it gains
	 * remain for the caller — the substrate for component races.
	 */
	template<detail::scannable_graph Graph, typename Marker = void>
		requires(std::is_void_v<Marker> || visit_marker<Marker>)
	class bfs_view : public std::ranges::view_interface<bfs_view<Graph, Marker>> {
		static constexpr bool owns_marks = std::is_void_v<Marker>;
		/**
		 * @brief Marker in signatures — a placeholder when Marker is void, so no
		 * `void&` is ever formed (the constrained overloads are gone anyway).
		 */
		using marker_param = std::conditional_t<owns_marks, detail::nothing, Marker>;

		Graph const *g_;
		/** @brief The seeds of a multi-source view; empty for a single source. */
		std::vector<vertex_id> sources_;
		/** @brief The seed of a single-source view, held inline: no allocation per view. */
		vertex_id source_ = none;
		[[no_unique_address]] std::conditional_t<owns_marks, detail::nothing, Marker *> marker_{};

		/**
		 * @brief The seeds: `sources_`, else the inline source (`none` marks an
		 * empty multi-source view).
		 */
		[[nodiscard]] constexpr std::span<vertex_id const> seeds() const {
			if (!sources_.empty()) return sources_;
			if (source_ == none) return {};
			return {&source_, 1};
		}

	public:
		/** @brief Single-pass traversal cursor; increments may allocate heap storage. */
		class iterator {
			Graph const *g_ = nullptr;
			/** @brief Every vertex discovered so far; `back()` is the current element. */
			std::vector<visit> order_;
			/**
			 * @brief Up to K supplied seeds not yet yielded, reversed; drained before any
			 *        expansion so all sources share depth 0.
			 */
			std::vector<vertex_id> pending_;
			/** @brief Index into `order_` of the vertex whose adjacency `edges_` is consuming. */
			std::size_t scan_ = 0;
			detail::edge_scan<Graph> edges_;
			[[no_unique_address]] detail::marks_of<Marker> marks_;
			bool is_done_ = false;

			constexpr void seed(std::span<vertex_id const> sources) {
				atl_assert(graph::vertex_count(*g_) <= none);
				for ([[maybe_unused]] vertex_id const s : sources) atl_assert(s < graph::vertex_count(*g_));
				std::size_t first = 0;
				while (first < sources.size() && !marks_.get().try_visit(sources[first])) ++first;
				if (first == sources.size()) {
					is_done_ = true;
					return;
				}
				pending_.assign(sources.rbegin(), sources.rend() - std::ptrdiff_t(first + 1));
				order_.push_back({sources[first], sources[first], 0});
				edges_ = detail::edge_scan<Graph>(*g_, sources[first]);
			}

		public:
			/** @brief Traversal record type. */
			using value_type = visit;
			/** @brief Signed iterator distance type. */
			using difference_type = std::ptrdiff_t;
			/** @brief Single-pass iterator category. */
			using iterator_concept = std::input_iterator_tag;

			/** @brief Default construction is available when marker storage is default-initializable; the cursor is unbound. */
			iterator() = default;

			/** @brief Start with the supplied graph, valid source(s), and optional persistent marker. */
			constexpr iterator(Graph const &g, std::span<vertex_id const> sources)
				requires(owns_marks)
				: g_(&g), marks_{.owned_marker = dense_marker(graph::vertex_count(g)), .external_marker = {}} {
				seed(sources);
			}

			/** @brief Start with the supplied graph, valid source(s), and optional persistent marker. */
			constexpr iterator(Graph const &g, std::span<vertex_id const> sources, bfs_view::marker_param &marker)
				requires(!owns_marks)
				: g_(&g), marks_{.owned_marker = {}, .external_marker = &marker} {
				seed(sources);
			}

			/** @brief Current discovery. @pre The cursor is bound and not at its sentinel. */
			[[nodiscard]] constexpr visit const &operator*() const { return order_.back(); }

			/** @brief Scan to the next discovery. @pre The cursor is bound and not at its sentinel. */
			constexpr iterator &operator++() {
				while (!pending_.empty()) [[unlikely]] {
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

			/** @brief Advance without retaining a readable cursor copy. */
			constexpr void operator++(int) { ++*this; }

			/** @brief Whether traversal is exhausted; the cursor must be bound. */
			[[nodiscard]] constexpr bool operator==(std::default_sentinel_t) const { return is_done_; }
		};

		/** @brief Start with the supplied graph, valid source(s), and optional persistent marker. */
		constexpr bfs_view(Graph const &g, vertex_id source)
			requires(owns_marks)
			: g_(&g), source_(source) { atl_assert(source < graph::vertex_count(g)); }

		/** @brief Start with the supplied graph, valid source(s), and optional persistent marker. */
		constexpr bfs_view(Graph const &g, vertex_id source, marker_param &marker)
			requires(!owns_marks)
			: g_(&g), source_(source), marker_(&marker) { atl_assert(source < graph::vertex_count(g)); }

		/** @brief Start with the supplied graph, valid source(s), and optional persistent marker. */
		template<typename Sources>
			requires vertex_range<Sources const>
		constexpr bfs_view(Graph const &g, Sources const &sources)
			requires(owns_marks)
			: g_(&g) {
			for (auto const &s : sources) sources_.push_back(vertex_id(s));
		}

		/** @brief Start with the supplied graph, valid source(s), and optional persistent marker. */
		template<typename Sources>
			requires vertex_range<Sources const>
		constexpr bfs_view(Graph const &g, Sources const &sources, marker_param &marker)
			requires(!owns_marks)
			: g_(&g), marker_(&marker) {
			for (auto const &s : sources) sources_.push_back(vertex_id(s));
		}

		/** @brief Start a traversal; external marks persist between calls. */
		constexpr iterator begin() const {
			if constexpr (owns_marks)
				return {*g_, seeds()};
			else
				return {*g_, seeds(), *marker_};
		}

		/** @brief Sentinel reached when no discoveries remain. */
		[[nodiscard]] constexpr std::default_sentinel_t end() const { return {}; }
	};

	/** @brief Breadth-first traversal of @p g from @p source. */
	template<detail::scannable_graph Graph>
	[[nodiscard]] constexpr auto bfs(Graph const &g, vertex_id source) {
		return bfs_view<Graph>(g, source);
	}

	/**
	 * @brief Multi-source breadth-first traversal: every source at depth 0 on one
	 * shared frontier — depths are distances to the *nearest* source.
	 */
	template<detail::scannable_graph Graph, typename Sources>
		requires vertex_range<Sources const>
	[[nodiscard]] constexpr auto bfs(Graph const &g, Sources const &sources) {
		return bfs_view<Graph>(g, sources);
	}

	/**
	 * @brief Breadth-first traversal against caller-owned visited state (see
	 * @ref atl::graph::visit_marker); the marker must outlive the iteration.
	 */
	template<detail::scannable_graph Graph, visit_marker Marker>
	[[nodiscard]] constexpr auto bfs(Graph const &g, vertex_id source, Marker &marker) {
		return bfs_view<Graph, Marker>(g, source, marker);
	}

	/** @brief Multi-source form with caller-owned visited state. */
	template<detail::scannable_graph Graph, typename Sources, visit_marker Marker>
		requires vertex_range<Sources const>
	[[nodiscard]] constexpr auto bfs(Graph const &g, Sources const &sources, Marker &marker) {
		return bfs_view<Graph, Marker>(g, sources, marker);
	}

	/**
	 * @brief Depth-first traversal as a lazy range: yields each reachable
	 *        vertex exactly once, in recursive preorder (children in
	 *        adjacency order), as a @ref atl::graph::visit.
	 *
	 * Iterative — one slim frame per path vertex, no recursion. `operator++`
	 * is edge-granular like @ref atl::graph::bfs_view; markers, lifetimes and graph
	 * preconditions work identically. O(reached vertices + scanned edges) time
	 * with O(1) graph access, O(depth) heap frames plus marker storage.
	 * The view refers to the graph: a C++23 range-for over `dfs(temporary_graph, s)`
	 * keeps that graph alive through the loop; storing the view beyond that
	 * full-expression dangles. No call-stack recursion, including on a million-vertex path.
	 */
	template<detail::scannable_graph Graph, typename Marker = void>
		requires(std::is_void_v<Marker> || visit_marker<Marker>)
	class dfs_view : public std::ranges::view_interface<dfs_view<Graph, Marker>> {
		static constexpr bool owns_marks = std::is_void_v<Marker>;
		using marker_param = std::conditional_t<owns_marks, detail::nothing, Marker>;

		Graph const *g_;
		vertex_id source_;
		[[no_unique_address]] std::conditional_t<owns_marks, detail::nothing, Marker *> marker_{};

	public:
		/** @brief Single-pass traversal cursor; increments may allocate heap storage. */
		class iterator {
			/**
			 * @brief Slim resumable frame: the edge range is *reloaded* on
			 *        resume (O(1): a span rebuild or a reference — out_edges
			 *        must be deterministic across calls, the library-wide
			 *        contract); the saved position is an index or a stable
			 *        iterator (see scan_position).
			 */
			struct frame {
				vertex_id v;
				[[no_unique_address]] detail::scan_position_t<Graph> at;
				uint32 depth;
			};

			Graph const *g_ = nullptr;
			std::vector<frame> stack_;
			visit current_{};
			[[no_unique_address]] detail::marks_of<Marker> marks_;
			bool is_done_ = false;

			constexpr void start(vertex_id source) {
				if (!marks_.get().try_visit(source)) {
					is_done_ = true;
					return;
				}
				current_ = {source, source, 0};
				stack_.push_back({source, detail::edge_scan<Graph>::initial(*g_, source), 0});
			}

		public:
			/** @brief Traversal record type. */
			using value_type = visit;
			/** @brief Signed iterator distance type. */
			using difference_type = std::ptrdiff_t;
			/** @brief Single-pass iterator category. */
			using iterator_concept = std::input_iterator_tag;

			/** @brief Default construction is available when marker storage is default-initializable; the cursor is unbound. */
			iterator() = default;

			/** @brief Start with the supplied graph, valid source(s), and optional persistent marker. */
			constexpr iterator(Graph const &g, vertex_id source)
				requires(owns_marks)
				: g_(&g), marks_{.owned_marker = dense_marker(graph::vertex_count(g)), .external_marker = {}} {
				atl_assert(graph::vertex_count(g) <= none);
				atl_assert(source < graph::vertex_count(g));
				start(source);
			}

			/** @brief Start with the supplied graph, valid source(s), and optional persistent marker. */
			constexpr iterator(Graph const &g, vertex_id source, dfs_view::marker_param &marker)
				requires(!owns_marks)
				: g_(&g), marks_{.owned_marker = {}, .external_marker = &marker} {
				atl_assert(graph::vertex_count(g) <= none);
				atl_assert(source < graph::vertex_count(g));
				start(source);
			}

			/** @brief Current discovery. @pre The cursor is bound and not at its sentinel. */
			[[nodiscard]] constexpr visit const &operator*() const { return current_; }

			/** @brief Scan to the next discovery. @pre The cursor is bound and not at its sentinel. */
			constexpr iterator &operator++() {
				while (!stack_.empty()) {
					frame &f = stack_.back();
					auto scan = detail::edge_scan<Graph>(*g_, f.v, f.at);
					while (!scan.exhausted()) {
						vertex_id const v = graph::target(scan.current());
						scan.advance();
						if (marks_.get().try_visit(v)) {
							f.at = scan.pos(); // write back before f can dangle
							current_ = {v, f.v, f.depth + 1};
							stack_.push_back({v, detail::edge_scan<Graph>::initial(*g_, v), current_.depth});
							return *this;
						}
					}
					stack_.pop_back();
				}
				is_done_ = true;
				return *this;
			}

			/** @brief Advance without retaining a readable cursor copy. */
			constexpr void operator++(int) { ++*this; }

			/** @brief Whether traversal is exhausted; the cursor must be bound. */
			[[nodiscard]] constexpr bool operator==(std::default_sentinel_t) const { return is_done_; }
		};

		/** @brief Start with the supplied graph, valid source(s), and optional persistent marker. */
		constexpr dfs_view(Graph const &g, vertex_id source)
			requires(owns_marks)
			: g_(&g), source_(source) { atl_assert(source < graph::vertex_count(g)); }

		/** @brief Start with the supplied graph, valid source(s), and optional persistent marker. */
		constexpr dfs_view(Graph const &g, vertex_id source, marker_param &marker)
			requires(!owns_marks)
			: g_(&g), source_(source), marker_(&marker) { atl_assert(source < graph::vertex_count(g)); }

		/** @brief Start a traversal; external marks persist between calls. */
		constexpr iterator begin() const {
			if constexpr (owns_marks)
				return {*g_, source_};
			else
				return {*g_, source_, *marker_};
		}

		/** @brief Sentinel reached when no discoveries remain. */
		[[nodiscard]] constexpr std::default_sentinel_t end() const { return {}; }
	};

	/** @brief Depth-first (preorder) traversal of @p g from @p source. */
	template<detail::scannable_graph Graph>
	[[nodiscard]] constexpr auto dfs(Graph const &g, vertex_id source) {
		return dfs_view<Graph>(g, source);
	}

	/** @brief Depth-first traversal against caller-owned visited state. */
	template<detail::scannable_graph Graph, visit_marker Marker>
	[[nodiscard]] constexpr auto dfs(Graph const &g, vertex_id source, Marker &marker) {
		return dfs_view<Graph, Marker>(g, source, marker);
	}

	/**
	 * @brief The bfs_visit event: `step(v, parent, depth)` per discovered vertex, in
	 * nondecreasing depth order (sources: `parent == v`, depth 0).
	 */
	template<typename F>
	concept bfs_step_event = std::invocable<F &, vertex_id, vertex_id, uint32>;

	namespace detail {
		/**
		 * @brief The BFS engine behind bfs_visit: expands @p queue in order,
		 *        appending and reporting every newly marked vertex. The queue
		 *        holds bare vertex ids; depths come from level boundaries
		 *        (a third of the queue traffic of {vertex, parent, depth}
		 *        records).
		 * @pre @p queue holds the seeds, each already marked in @p marks and
		 *      already reported to @p step at depth 0.
		 */
		template<scannable_graph Graph, visit_marker Marker, bfs_step_event Step>
		constexpr void bfs_run(Graph const &g, std::vector<vertex_id> &queue, Marker &marks, Step &step) {
			uint32 depth = 1;
			std::size_t level_end = queue.size();
			for (std::size_t head = 0; head < queue.size(); ++head) {
				if (head == level_end) {
					++depth;
					level_end = queue.size();
				}
				vertex_id const u = queue[head];
				for (auto const &e : graph::out_edges(g, u))
					if (vertex_id const v = graph::target(e); marks.try_visit(v)) {
						queue.push_back(v);
						step(vertex_id{v}, vertex_id{u}, uint32{depth});
					}
			}
		}

		/**
		 * @brief bfs_visit from a seed range; @p is_dense says the marker spans
		 *        the whole graph, so the traversal may reserve the full queue
		 *        up front (a caller's marker may be an epoch marker over tiny
		 *        components, where an O(n) reserve would dominate).
		 */
		template<scannable_graph Graph, typename Sources, visit_marker Marker, bfs_step_event Step>
			requires vertex_range<Sources const>
		constexpr void bfs_visit_from(Graph const &g, Sources const &sources, Marker &marker, Step &step,
									  bool is_dense) {
			atl_assert(graph::vertex_count(g) <= none);
			std::vector<vertex_id> queue;
			if (is_dense) queue.reserve(graph::vertex_count(g));
			for (auto const &s : sources) {
				atl_assert(vertex_id(s) < graph::vertex_count(g));
				if (marker.try_visit(vertex_id(s))) {
					queue.push_back(vertex_id(s));
					step(vertex_id(s), vertex_id(s), uint32{0});
				}
			}
			detail::bfs_run(g, queue, marker, step);
		}
	}

	/**
	 * @brief Eager BFS via internal iteration: one `step(v, parent, depth)`
	 *        call per reachable vertex, in nondecreasing depth order.
	 *
	 * The full-throughput counterpart of the @ref atl::graph::bfs_view — no per-vertex
	 * yield boundary, so it runs at hand-rolled-loop speed (see docs/graph.md
	 * for the measured ratios). Reach for the view when you need early exit
	 * or composition; reach for this when you consume the whole traversal.
	 * @complexity O(reached vertices + scanned edges), plus owned-marker initialization where applicable
	 */
	template<detail::scannable_graph Graph, visit_marker Marker, bfs_step_event Step>
	constexpr void bfs_visit(Graph const &g, vertex_id source, Marker &marker, Step step) {
		detail::bfs_visit_from(g, std::span<vertex_id const>(&source, 1), marker, step, false);
	}

	/** @brief bfs_visit with a fresh internal dense marker. */
	template<detail::scannable_graph Graph, bfs_step_event Step>
	constexpr void bfs_visit(Graph const &g, vertex_id source, Step step) {
		dense_marker marks(graph::vertex_count(g));
		detail::bfs_visit_from(g, std::span<vertex_id const>(&source, 1), marks, step, true);
	}

	/** @brief Multi-source bfs_visit: every source at depth 0 on one shared frontier.
	 * @complexity O(K + reached vertices + scanned edges) time for K supplied sources,
	 * duplicates included; O(reached vertices) queue storage plus the caller marker. */
	template<detail::scannable_graph Graph, typename Sources, visit_marker Marker, bfs_step_event Step>
		requires vertex_range<Sources const>
	constexpr void bfs_visit(Graph const &g, Sources const &sources, Marker &marker, Step step) {
		detail::bfs_visit_from(g, sources, marker, step, false);
	}

	/** @brief Multi-source bfs_visit with a fresh internal dense marker.
	 * @complexity O(n / 64 + K + reached vertices + scanned edges) time for K supplied sources,
	 * duplicates included; O(n) reserved queue storage plus O(n / 64 + 1) marker words. */
	template<detail::scannable_graph Graph, typename Sources, bfs_step_event Step>
		requires vertex_range<Sources const>
	constexpr void bfs_visit(Graph const &g, Sources const &sources, Step step) {
		dense_marker marks(graph::vertex_count(g));
		detail::bfs_visit_from(g, sources, marks, step, true);
	}

	/**
	 * @brief Accepts anything and does nothing — the filler for unused dfs_visit
	 * event slots.
	 */
	inline constexpr auto skip = [](auto &&...) noexcept {};

	/**
	 * @brief Preorder discovery: `enter(v, parent, parent_edge, depth)`; the root
	 * enters with `parent == v`, `parent_edge == none`, depth 0.
	 */
	template<typename F>
	concept dfs_enter_event = std::invocable<F &, vertex_id, vertex_id, uint32, uint32>;

	/**
	 * @brief Non-tree edge: `seen_edge(u, k, v)` — the k-th out-edge of u points at
	 * already-discovered v (back/forward/cross edges; on doubled undirected
	 * graphs this includes the reverse of the tree edge).
	 */
	template<typename F>
	concept dfs_seen_edge_event = std::invocable<F &, vertex_id, uint32, vertex_id>;

	/**
	 * @brief Postorder completion: `exit(v, parent)` — v's subtree is finished; the
	 * root exits with `parent == v`.
	 */
	template<typename F>
	concept dfs_exit_event = std::invocable<F &, vertex_id, vertex_id>;

	namespace detail {

		template<typename... Events>
		concept forwardable_events = (std::constructible_from<Events, Events &&> && ...);

		template<typename F>
		inline constexpr int dfs_roles =
			int(dfs_enter_event<F>) + int(dfs_seen_edge_event<F>) + int(dfs_exit_event<F>);

		template<scannable_graph Graph>
		struct dfs_frame {
			vertex_id v;
			[[no_unique_address]] scan_position_t<Graph> at;
			uint32 depth;
		};

		/**
		 * @brief The single iterative DFS engine behind dfs_visit: slim
		 *        resumable frames, edge ranges reloaded on resume (see
		 *        edge_scan). Every edge fires exactly one event: enter for
		 *        tree edges, seen_edge otherwise.
		 * @pre @p s is already marked in @p marks.
		 */
		template<scannable_graph Graph, visit_marker Marker, dfs_enter_event Enter, dfs_seen_edge_event SeenEdge,
				 dfs_exit_event Exit>
		constexpr void dfs_run(Graph const &g, vertex_id s, Marker &marks, std::vector<dfs_frame<Graph>> &stack, Enter &enter,
							   SeenEdge &seen_edge, Exit &exit) {
			enter(vertex_id{s}, vertex_id{s}, uint32{none}, uint32{0});
			stack.push_back({s, edge_scan<Graph>::initial(g, s), 0});
			while (!stack.empty()) {
				dfs_frame<Graph> &f = stack.back();
				auto scan = edge_scan<Graph>(g, f.v, f.at);
				vertex_id discovered = none;
				while (!scan.exhausted()) {
					vertex_id const v = graph::target(scan.current());
					uint32 const k = scan.index();
					scan.advance();
					if (!marks.try_visit(v)) {
						seen_edge(vertex_id{f.v}, uint32{k}, vertex_id{v});
						continue;
					}
					f.at = scan.pos(); // write back before f can dangle
					enter(vertex_id{v}, vertex_id{f.v}, uint32{k}, uint32{f.depth + 1});
					stack.push_back({v, edge_scan<Graph>::initial(g, v), f.depth + 1});
					discovered = v;
					break;
				}
				if (discovered == none) {
					dfs_frame<Graph> const &top = stack.back();
					exit(vertex_id{top.v}, vertex_id{stack.size() >= 2 ? stack[stack.size() - 2].v : top.v});
					stack.pop_back();
				}
			}
		}
	}

	/**
	 * @brief Event-driven DFS: the traversal is written once, callers observe
	 *        exactly the events they care about and store only what they need
	 *        — parents, Euler intervals, low-links, tree DP are all a couple
	 *        of lambdas (see @ref atl::graph::events for passing a subset).
	 *
	 * Iterative (slim frames, one per path vertex), edge-granular. Each event
	 * slot takes a plain callable with the exact role signature — @ref atl::graph::skip
	 * fills unused slots. A source already visited in a caller-owned marker
	 * is skipped entirely.
	 * @complexity O(reached vertices + scanned edges), plus owned-marker initialization where applicable
	 */
	template<detail::scannable_graph Graph, visit_marker Marker, dfs_enter_event Enter,
			 dfs_seen_edge_event SeenEdge, dfs_exit_event Exit>
	constexpr void dfs_visit(Graph const &g, vertex_id source, Marker &marker, Enter enter, SeenEdge seen_edge,
							 Exit exit) {
		atl_assert(graph::vertex_count(g) <= none);
		atl_assert(source < graph::vertex_count(g));
		if (!marker.try_visit(source)) return;
		std::vector<detail::dfs_frame<Graph>> stack;
		detail::dfs_run(g, source, marker, stack, enter, seen_edge, exit);
	}

	/** @brief dfs_visit with a fresh internal dense marker. */
	template<detail::scannable_graph Graph, dfs_enter_event Enter, dfs_seen_edge_event SeenEdge,
			 dfs_exit_event Exit>
		requires detail::forwardable_events<Enter, SeenEdge, Exit>
	constexpr void dfs_visit(Graph const &g, vertex_id source, Enter enter, SeenEdge seen_edge, Exit exit) {
		dense_marker marks(graph::vertex_count(g));
		dfs_visit(g, source, marks, std::move(enter), std::move(seen_edge), std::move(exit));
	}

	/**
	 * @brief Whole-graph dfs_visit against caller-owned visited state: every
	 * yet-unvisited vertex roots a new tree. @complexity O(n + m)
	 */
	template<detail::scannable_graph Graph, visit_marker Marker, dfs_enter_event Enter,
			 dfs_seen_edge_event SeenEdge, dfs_exit_event Exit>
	constexpr void dfs_visit(Graph const &g, Marker &marker, Enter enter, SeenEdge seen_edge, Exit exit) {
		atl_assert(graph::vertex_count(g) <= none);
		std::vector<detail::dfs_frame<Graph>> stack;
		for (vertex_id v = 0; v < graph::vertex_count(g); ++v)
			if (marker.try_visit(v)) detail::dfs_run(g, v, marker, stack, enter, seen_edge, exit);
	}

	/** @brief Whole-graph dfs_visit with a fresh internal dense marker. */
	template<detail::scannable_graph Graph, dfs_enter_event Enter, dfs_seen_edge_event SeenEdge,
			 dfs_exit_event Exit>
		requires detail::forwardable_events<Enter, SeenEdge, Exit>
	constexpr void dfs_visit(Graph const &g, Enter enter, SeenEdge seen_edge, Exit exit) {
		dense_marker marks(graph::vertex_count(g));
		dfs_visit(g, marks, std::move(enter), std::move(seen_edge), std::move(exit));
	}

	namespace detail {
		/**
		 * @brief One @ref atl::graph::dfs_fold stack frame: a vertex mid-visit — its
		 *        identity and depth, the resumable edge-scan position, the
		 *        accumulator being built, and the parent edge element stashed
		 *        at discovery (default-constructed at the root, where it is
		 *        never read).
		 */
		template<scannable_graph Graph, std::movable Acc>
		struct fold_frame {
			vertex_id v, parent;
			[[no_unique_address]] scan_position_t<Graph> at;
			uint32 depth;
			Acc acc;
			[[no_unique_address]] std::ranges::range_value_t<out_edges_t<Graph>> edge;
		};

		struct fold_no_finalize {
			template<std::movable Acc>
			constexpr void operator()(Acc &, visit) const noexcept {}
		};
	}

	/**
	 * @brief The @ref atl::graph::dfs_fold callback bundle over graph `Graph`: `init(visit) -> Acc`
	 * makes a vertex's base accumulator, `fold(Acc&, Acc, parent, child,
	 * Edge const&)` merges a finished child across its connecting edge, and
	 * `finalize(Acc&, visit const&)` seals a vertex after its last child. The edge
	 * element must be default-constructible and copyable because every frame
	 * stashes its parent edge by value (defaulted at the root).
	 */
	template<typename Graph, typename Init, typename Fold, typename Finalize>
	concept dfs_fold_callbacks =
		detail::scannable_graph<Graph>
		&& std::invocable<Init &, visit>
		&& std::movable<std::invoke_result_t<Init &, visit>>
		&& std::default_initializable<std::ranges::range_value_t<out_edges_t<Graph>>>
		&& std::copy_constructible<std::ranges::range_value_t<out_edges_t<Graph>>>
		&& std::constructible_from<std::ranges::range_value_t<out_edges_t<Graph>>,
								   decltype(std::declval<detail::edge_scan<Graph> const &>().current())>
		&& std::invocable<Fold &, std::invoke_result_t<Init &, visit> &,
						  std::invoke_result_t<Init &, visit>, visit, visit,
						  std::ranges::range_value_t<out_edges_t<Graph>> const &>
		&& std::invocable<Finalize &, std::invoke_result_t<Init &, visit> &, visit const &>;

	/**
	 * @brief Bottom-up fold over the DFS tree — the value-returning recursive
	 *        DFS, run iteratively so deep trees never overflow the stack.
	 *
	 * Each vertex gets an accumulator from @p init; as every child finishes,
	 * @p fold merges the child's accumulator into the parent's (across the
	 * connecting edge); once a vertex's children are all folded in, @p
	 * finalize seals its accumulator before it is handed up. The root's
	 * accumulator is returned. Only the accumulators of the vertices on the
	 * current root-to-node path are ever live at once — O(depth), not O(n) —
	 * because each child's result is folded and dropped the moment it
	 * completes.
	 *
	 * Callback contract (`Acc` = `init`'s result; `Edge` = the adjacency
	 * element type; each @ref atl::graph::visit is the vertex's position in the DFS tree):
	 * - `Acc init(visit u)` — u's base accumulator.
	 * - `void fold(Acc &acc, Acc child, visit u, visit child_node, Edge const &edge)`
	 *   — merge `child` (moved in) into u's `acc`; `edge` is u -> child_node.
	 * - `void finalize(Acc &acc, visit const &u)` — after all children folded
	 *   (a by-value visit parameter is also accepted).
	 *
	 * Info a fold needs flows through three channels: structural facts in the
	 * @ref atl::graph::visit arguments, computed/threaded parent state in `acc` (the
	 * mutable accumulator), and external data via the vertex ids. The edge is
	 * passed as the *element*, not an index, so it works on non-random-access
	 * adjacencies (`vector<set<edge>>`, `vector<map<..>>`) too.
	 * @pre Graph and marker storage stay valid; adjacency order is deterministic and
	 * unchanged during the run. Every target is below `vertex_count(g) <= none`;
	 * each row has at most `none` edges. The root is a valid vertex (asserted).
	 * @return The finalized root accumulator of the reachable DFS tree.
	 * @complexity O(n / 64 + reached vertices + scanned edges) plus callback
	 * costs; O(n / 64 + 1) marker words and O(depth) frames/accumulators.
	 * Graph access costs are additional.
	 */
	template<detail::scannable_graph Graph, typename Init, typename Fold, typename Finalize>
		requires dfs_fold_callbacks<Graph, Init, Fold, Finalize>
	[[nodiscard]] constexpr auto dfs_fold(Graph const &g, vertex_id root, Init init, Fold fold, Finalize finalize)
		-> std::invoke_result_t<Init &, visit> {
		using Acc = std::invoke_result_t<Init &, visit>;
		using Edge = std::ranges::range_value_t<out_edges_t<Graph>>;
		atl_assert(graph::vertex_count(g) <= none);
		atl_assert(root < graph::vertex_count(g));

		dense_marker marks(graph::vertex_count(g));
		marks.try_visit(root);
		std::vector<detail::fold_frame<Graph, Acc>> stack;
		stack.push_back({root, root, detail::edge_scan<Graph>::initial(g, root), 0,
						 std::invoke(init, visit{root, root, 0}), Edge{}});
		for (;;) {
			auto &f = stack.back();
			auto scan = detail::edge_scan<Graph>(g, f.v, f.at);
			bool has_descended = false;
			while (!scan.exhausted()) {
				auto &&e = scan.current();
				vertex_id const w = graph::target(e);
				scan.advance();
				if (!marks.try_visit(w)) continue;
				f.at = scan.pos(); // write back before push may dangle f
				uint32 const cd = f.depth + 1;
				vertex_id const fv = f.v;
				stack.push_back({w, fv, detail::edge_scan<Graph>::initial(g, w), cd,
								 std::invoke(init, visit{w, fv, cd}), Edge(e)});
				has_descended = true;
				break;
			}
			if (has_descended) continue;
			auto &top = stack.back();
			visit const un{top.v, top.parent, top.depth};
			std::invoke(finalize, top.acc, un);
			if (stack.size() == 1) return std::move(top.acc);
			auto &par = stack[stack.size() - 2];
			std::invoke(fold, par.acc, std::move(top.acc), visit{par.v, par.parent, par.depth}, visit{un}, std::as_const(top.edge));
			stack.pop_back();
		}
	}

	/** @brief @ref atl::graph::dfs_fold without a finalize step (init + fold only). */
	template<detail::scannable_graph Graph, typename Init, typename Fold>
		requires dfs_fold_callbacks<Graph, Init, Fold, detail::fold_no_finalize>
	[[nodiscard]] constexpr auto dfs_fold(Graph const &g, vertex_id root, Init init, Fold fold) {
		return dfs_fold(g, root, std::move(init), std::move(fold), detail::fold_no_finalize{});
	}

	/** @brief The three dfs events of an @ref atl::graph::events bundle, unused slots @ref atl::graph::skip. */
	template<dfs_enter_event Enter, dfs_seen_edge_event SeenEdge, dfs_exit_event Exit>
	struct event_set {
		[[no_unique_address]] Enter enter;		  ///< Discovery callback.
		[[no_unique_address]] SeenEdge seen_edge; ///< Non-tree-edge callback.
		[[no_unique_address]] Exit exit;		  ///< Completion callback.
	};

	namespace detail {
		/**
		 * @brief The first tuple element filling dfs role @p Role (0 enter, 1
		 *        seen_edge, 2 exit), or @ref atl::graph::skip if none does. The element
		 *        keeps its original value category (the tuple holds forwarding
		 *        references), so an rvalue callable is moved into the bundle,
		 *        never copied.
		 */
		template<int Role, std::size_t I = 0, typename Tuple>
			requires(Role >= 0 && Role <= 2) && requires { typename std::tuple_size<Tuple>::type; }
				 && (std::tuple_size_v<Tuple> <= 3 && I <= std::tuple_size_v<Tuple>)
		[[nodiscard]] constexpr auto dfs_event_for(Tuple &t) {
			constexpr std::size_t selected = []<std::size_t... Index>(std::index_sequence<Index...>) {
				std::size_t result = std::tuple_size_v<Tuple>;
				((Index >= I && result == std::tuple_size_v<Tuple>
						  && (Role == 0	  ? dfs_enter_event<std::remove_cvref_t<std::tuple_element_t<Index, Tuple>>>
							  : Role == 1 ? dfs_seen_edge_event<std::remove_cvref_t<std::tuple_element_t<Index, Tuple>>>
										  : dfs_exit_event<std::remove_cvref_t<std::tuple_element_t<Index, Tuple>>>)
					  ? result = Index
					  : result),
				 ...);
				return result;
			}(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
			if constexpr (selected == std::tuple_size_v<Tuple>)
				return skip;
			else
				return std::forward<std::tuple_element_t<selected, Tuple>>(std::get<selected>(t));
		}
	}

	/**
	 * @brief Bundle 1-3 dfs event callables given **in any order**, roles
	 *        inferred from their signatures (the three arities are distinct),
	 *        missing roles filled with @ref atl::graph::skip, as in
	 *        `dfs_visit(g, s, events([&](vertex_id v, vertex_id p) { ... }))`.
	 *
	 * Rejected at compile time: a callable matching no role, a catch-all
	 * matching several (ambiguous), and two callables claiming the same role.
	 */
	template<typename... Events>
		requires(sizeof...(Events) >= 1 && sizeof...(Events) <= 3
				 && (std::constructible_from<std::remove_cvref_t<Events>, Events &&> && ...)
				 && ((detail::dfs_roles<std::remove_cvref_t<Events>> == 1) && ...)
				 && (0 + ... + int(dfs_enter_event<std::remove_cvref_t<Events>>)) <= 1
				 && (0 + ... + int(dfs_seen_edge_event<std::remove_cvref_t<Events>>)) <= 1
				 && (0 + ... + int(dfs_exit_event<std::remove_cvref_t<Events>>)) <= 1)
	[[nodiscard]] constexpr auto events(Events &&...event) {
		auto refs = std::forward_as_tuple(std::forward<Events>(event)...);
		return event_set{detail::dfs_event_for<0>(refs), detail::dfs_event_for<1>(refs), detail::dfs_event_for<2>(refs)};
	}

	/** @brief dfs_visit over an @ref atl::graph::events bundle. */
	template<detail::scannable_graph Graph, dfs_enter_event Enter, dfs_seen_edge_event SeenEdge, dfs_exit_event Exit>
		requires detail::forwardable_events<Enter, SeenEdge, Exit>
	constexpr void dfs_visit(Graph const &g, vertex_id source, event_set<Enter, SeenEdge, Exit> bundle) {
		dfs_visit(g, source, std::move(bundle.enter), std::move(bundle.seen_edge), std::move(bundle.exit));
	}

	/** @brief dfs_visit over an @ref atl::graph::events bundle with caller-owned visited state. */
	template<detail::scannable_graph Graph, visit_marker Marker, dfs_enter_event Enter,
			 dfs_seen_edge_event SeenEdge, dfs_exit_event Exit>
		requires detail::forwardable_events<Enter, SeenEdge, Exit>
	constexpr void dfs_visit(Graph const &g, vertex_id source, Marker &marker, event_set<Enter, SeenEdge, Exit> bundle) {
		dfs_visit(g, source, marker, std::move(bundle.enter), std::move(bundle.seen_edge),
				  std::move(bundle.exit));
	}

	/** @brief Whole-graph dfs_visit over an @ref atl::graph::events bundle. */
	template<detail::scannable_graph Graph, dfs_enter_event Enter, dfs_seen_edge_event SeenEdge, dfs_exit_event Exit>
		requires detail::forwardable_events<Enter, SeenEdge, Exit>
	constexpr void dfs_visit(Graph const &g, event_set<Enter, SeenEdge, Exit> bundle) {
		dfs_visit(g, std::move(bundle.enter), std::move(bundle.seen_edge), std::move(bundle.exit));
	}

	/**
	 * @brief Whole-graph dfs_visit over an @ref atl::graph::events bundle with caller-owned
	 * visited state.
	 */
	template<detail::scannable_graph Graph, visit_marker Marker, dfs_enter_event Enter,
			 dfs_seen_edge_event SeenEdge, dfs_exit_event Exit>
		requires detail::forwardable_events<Enter, SeenEdge, Exit>
	constexpr void dfs_visit(Graph const &g, Marker &marker, event_set<Enter, SeenEdge, Exit> bundle) {
		dfs_visit(g, marker, std::move(bundle.enter), std::move(bundle.seen_edge), std::move(bundle.exit));
	}
}
