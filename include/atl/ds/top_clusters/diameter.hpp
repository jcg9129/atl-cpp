#pragma once
#include <atl/ds/top_clusters/common.hpp>
#include <atl/ds/top_tree_engine.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <concepts>
#include <utility>

namespace atl::top_clusters {
	/** @brief A vertex at a distance; `vertex == graph::none` means "no vertex" (an empty cluster). */
	template<detail::length W>
	struct farthest_vertex {
		W distance{};				 ///< the distance
		uint32 vertex = graph::none; ///< the vertex, `graph::none` when there is none

		/** @brief Compare every stored field. */
		[[nodiscard]] constexpr bool operator==(farthest_vertex const &) const = default;
	};

	/** @brief A longest path: its length and endpoints (`graph::none` in an empty cluster). */
	template<detail::length W>
	struct diameter_span {
		W length{};					 ///< the length (sum of edge lengths)
		uint32 first = graph::none;	 ///< one endpoint
		uint32 second = graph::none; ///< the other endpoint

		/** @brief Compare every stored field. */
		[[nodiscard]] constexpr bool operator==(diameter_span const &) const = default;
	};

	/**
	 * @brief A node value of the @ref atl::top_clusters::diameter set on the edge tier: an edge
	 *        (its length; converts implicitly from `W`, so `link(u, v, w)` and
	 *        `set(e, w)` take a bare length) or a vertex (length 0, its id).
	 */
	template<detail::length W>
	struct diameter_value {
		W length{};					 ///< the edge length; 0 on a vertex
		uint32 vertex = graph::none; ///< the vertex id; `graph::none` on an edge

		/** @brief A zero-length placeholder with no vertex id. */
		constexpr diameter_value() = default;

		/** @brief An edge of length @p edge_length. */
		constexpr diameter_value(W edge_length) : length(std::move(edge_length)) {} // NOLINT(google-explicit-constructor): an edge length converts

		/** @brief The value of vertex @p v (what a `diameter_top_tree` starts each vertex with). */
		[[nodiscard]] static constexpr diameter_value of_vertex(uint32 v) {
			diameter_value x;
			x.vertex = v;
			return x;
		}

		/** @brief Compare every stored field. */
		[[nodiscard]] constexpr bool operator==(diameter_value const &) const = default;
	};

	/**
	 * @brief The @ref atl::top_clusters::diameter path cluster: the spine's length, the farthest
	 *        vertex from each end (distances include both end nodes), and the
	 *        longest path inside.
	 */
	template<detail::length W>
	struct diameter_path {
		W length{};						///< the spine length
		farthest_vertex<W> from_top;	///< the farthest vertex from the top end
		farthest_vertex<W> from_bottom; ///< the farthest vertex from the bottom end
		diameter_span<W> diameter;		///< the longest path inside the cluster

		/** @brief Compare every stored field. */
		[[nodiscard]] constexpr bool operator==(diameter_path const &) const = default;
	};

	/** @brief The @ref atl::top_clusters::diameter hanging set: the deepest vertex below the attachment and the longest path inside. */
	template<detail::length W>
	struct diameter_point {
		farthest_vertex<W> deepest; ///< the farthest vertex from the attachment (exclusive)
		diameter_span<W> diameter;	///< the longest path inside the set

		/** @brief Compare every stored field. */
		[[nodiscard]] constexpr bool operator==(diameter_point const &) const = default;
	};

	/**
	 * @brief Edge lengths with the diameter, eccentricity and farthest vertex
	 *        of every tree, and path lengths (`diameter_top_tree<W>`, edge
	 *        tier): dynamic diameter and farthest-vertex queries on a
	 *        changing tree. Ties pick either candidate.
	 * @pre lengths and their sums stay in range for `W`; vertex nodes keep
	 *      length 0 (a longest path through a vertex adds nothing for it)
	 */
	template<detail::length W>
	struct diameter {
		/** @brief Vertex or edge datum accepted by this set. */
		using value_type = diameter_value<W>;
		/** @brief Summary of a spine and its hanging subtrees. */
		using path_type = diameter_path<W>;
		/** @brief Summary of subtrees attached at one boundary. */
		using point_type = diameter_point<W>;
		/** @brief Lazy action on the covered values. */
		using tag_type = void;
		/** @brief Value mapping slot, disabled for this distance set. */
		using mapping = void;
		/** @brief Path mapping slot, disabled for this distance set. */
		using map_path = void;
		/** @brief Hanging-set mapping slot, disabled for this distance set. */
		using map_point = void;
		/** @brief Tag composition slot, disabled for this distance set. */
		using compose = void;

		/** @brief The farther of two candidates (`a` on ties, either when one is absent). */
		[[nodiscard]] static constexpr farthest_vertex<W> farther(farthest_vertex<W> const &a, farthest_vertex<W> const &b) {
			if (b.vertex == graph::none) return a;
			if (a.vertex == graph::none) return b;
			return a.distance < b.distance ? b : a;
		}

		/** @brief Candidate @p c seen @p by further away. */
		[[nodiscard]] static constexpr farthest_vertex<W> shifted(farthest_vertex<W> const &c, W const &by) {
			return c.vertex == graph::none ? c : farthest_vertex<W>{c.distance + by, c.vertex};
		}

		/** @brief The longer of two paths. */
		[[nodiscard]] static constexpr diameter_span<W> longer(diameter_span<W> const &a, diameter_span<W> const &b) {
			if (b.first == graph::none) return a;
			if (a.first == graph::none) return b;
			return a.length < b.length ? b : a;
		}

		/** @brief The path joining the vertices of @p a and @p b through the point their distances meet at. */
		[[nodiscard]] static constexpr diameter_span<W> joined(farthest_vertex<W> const &a, farthest_vertex<W> const &b) {
			if (a.vertex == graph::none || b.vertex == graph::none) return {};
			return {a.distance + b.distance, a.vertex, b.vertex};
		}

		/** @brief Join two path clusters. */
		struct compress {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr path_type operator()(path_type const &upper, path_type const &lower) const {
				return {upper.length + lower.length, farther(upper.from_top, shifted(lower.from_top, upper.length)),
						farther(lower.from_bottom, shifted(upper.from_bottom, lower.length)),
						longer(longer(upper.diameter, lower.diameter), joined(upper.from_bottom, lower.from_top))};
			}
		};

		/** @brief Merge two hanging sets (at a vertex, length 0). */
		struct rake {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr point_type operator()(point_type const &a, point_type const &b) const {
				return {farther(a.deepest, b.deepest), longer(longer(a.diameter, b.diameter), joined(a.deepest, b.deepest))};
			}
		};

		/** @brief A node over its hanging set. */
		struct add_vertex {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr path_type operator()(point_type const &light, value_type const &node) const {
				farthest_vertex<W> const self{node.length, node.vertex};
				farthest_vertex<W> const reach = farther(self, shifted(light.deepest, node.length));
				diameter_span<W> const alone =
					node.vertex == graph::none ? diameter_span<W>{} : diameter_span<W>{node.length, node.vertex, node.vertex};
				return {node.length, reach, reach, longer(longer(light.diameter, alone), joined(self, light.deepest))};
			}
		};

		/** @brief A child's cluster as a hanging set. */
		struct add_edge {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr point_type operator()(path_type const &child) const { return {child.from_top, child.diameter}; }
		};

		/** @brief Reverse a path cluster: swap its ends. */
		struct reverse {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr path_type operator()(path_type const &p) const { return {p.length, p.from_bottom, p.from_top, p.diameter}; }
		};
	};

	/** @brief This set's path type is ready-made. */
	template<detail::length W>
	inline constexpr bool is_ready_made<diameter_path<W>> = true;

}

namespace atl::detail {
	template<typename Derived, std::copyable Value, typename T, typename Tag>
	class top_named_members<Derived, Value, top_clusters::diameter_path<T>, Tag> {
		constexpr Derived &self() { return static_cast<Derived &>(*this); }

	public:
		/** @brief The length of the longest path in v's tree (0 for a lone vertex). @complexity amortized O(log n) */
		[[nodiscard]] constexpr auto diameter(uint32 v) {
			return self().component_fold(v).diameter.length;
		}

		/** @brief The endpoints of a longest path in v's tree. @complexity amortized O(log n) */
		[[nodiscard]] constexpr std::pair<uint32, uint32> diameter_endpoints(uint32 v) {
			auto const span = self().component_fold(v).diameter;
			return {span.first, span.second};
		}

		/** @brief The distance from v to the farthest vertex of its tree. @complexity amortized O(log n) */
		[[nodiscard]] constexpr auto eccentricity(uint32 v) {
			return self().component_fold(v).from_bottom.distance;
		}

		/** @brief The farthest vertex from v in its tree, with its distance. @complexity amortized O(log n) */
		[[nodiscard]] constexpr auto farthest(uint32 v) {
			return self().component_fold(v).from_bottom;
		}

		/** @brief The length of the u - v path. @pre connected (asserted) @complexity amortized O(log n) */
		[[nodiscard]] constexpr auto path_length(uint32 u, uint32 v) {
			return self().fold(u, v).length;
		}
	};
}

namespace atl {
	/** @brief Edge lengths with diameter, eccentricity, farthest vertex, path length (@ref atl::top_clusters::diameter). */
	template<typename W>
	using diameter_top_tree = detail::top_tree_of_set<top_clusters::diameter<W>, edge_values_t>;
}

namespace atl::top_clusters::detail {
	/** @brief The path type of @ref atl::top_clusters::diameter. */
	template<typename Path>
	concept diameter_family = is_ready_made<Path> && requires(Path const &p) {
		p.diameter;
		p.from_bottom;
	};
}
