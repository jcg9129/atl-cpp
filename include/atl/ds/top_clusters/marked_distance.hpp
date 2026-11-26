#pragma once
#include <atl/ds/top_clusters/common.hpp>
#include <atl/ds/top_tree_engine.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <concepts>
#include <cstddef>
#include <utility>

namespace atl::top_clusters {
	/**
	 * @brief A node value of the @ref atl::top_clusters::marked_distance set on the edge tier:
	 *        an edge (its length; converts implicitly from `W`) or a vertex
	 *        (length 0, marked or not).
	 */
	template<detail::length W>
	struct marked_value {
		W length{};				///< the edge length; 0 on a vertex
		bool is_marked = false; ///< a marked vertex

		/** @brief An unmarked zero-length node. */
		constexpr marked_value() = default;

		/** @brief An edge of length @p edge_length. */
		constexpr marked_value(W edge_length) : length(std::move(edge_length)) {} // NOLINT(google-explicit-constructor): an edge length converts

		/** @brief Compare every stored field. */
		[[nodiscard]] constexpr bool operator==(marked_value const &) const = default;
	};

	/**
	 * @brief The @ref atl::top_clusters::marked_distance path cluster: spine length, the number
	 *        of marked vertices, and the sums of their distances to each end.
	 */
	template<detail::length W>
	struct marked_path {
		W length{};		///< the spine length
		W top_sum{};	///< sum of the marked vertices' distances to the top end
		W bottom_sum{}; ///< sum of the marked vertices' distances to the bottom end
		uint32 count{}; ///< the number of marked vertices

		/** @brief Compare every stored field. */
		[[nodiscard]] constexpr bool operator==(marked_path const &) const = default;
	};

	/** @brief The @ref atl::top_clusters::marked_distance hanging set: marked count and distance sum to the attachment. */
	template<detail::length W>
	struct marked_point {
		W sum{};		///< sum of the marked vertices' distances to the attachment (exclusive)
		uint32 count{}; ///< the number of marked vertices

		/** @brief Compare every stored field. */
		[[nodiscard]] constexpr bool operator==(marked_point const &) const = default;
	};

	/**
	 * @brief Marked vertices and edge lengths with the sum of distances from a
	 *        vertex to every marked vertex of its tree
	 *        (`marked_distance_top_tree<W>`, edge tier).
	 * @pre the sums stay in range for `W`
	 */
	template<detail::length W>
	struct marked_distance {
		/** @brief Vertex or edge datum accepted by this set. */
		using value_type = marked_value<W>;
		/** @brief Summary of a spine and its hanging subtrees. */
		using path_type = marked_path<W>;
		/** @brief Summary of subtrees attached at one boundary. */
		using point_type = marked_point<W>;
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

		/** @brief Join two path clusters. */
		struct compress {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr path_type operator()(path_type const &upper, path_type const &lower) const {
				return {upper.length + lower.length, upper.top_sum + lower.top_sum + upper.length * W(lower.count),
						lower.bottom_sum + upper.bottom_sum + lower.length * W(upper.count), upper.count + lower.count};
			}
		};

		/** @brief Merge two hanging sets. */
		struct rake {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr point_type operator()(point_type const &a, point_type const &b) const { return {a.sum + b.sum, a.count + b.count}; }
		};

		/** @brief A node over its hanging set. */
		struct add_vertex {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr path_type operator()(point_type const &light, value_type const &node) const {
				uint32 const count = light.count + uint32(node.is_marked);
				W const sum = light.sum + node.length * W(count);
				return {node.length, sum, sum, count};
			}
		};

		/** @brief A child's cluster as a hanging set. */
		struct add_edge {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr point_type operator()(path_type const &child) const { return {child.top_sum, child.count}; }
		};

		/** @brief Reverse a path cluster: swap its ends. */
		struct reverse {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr path_type operator()(path_type const &p) const { return {p.length, p.bottom_sum, p.top_sum, p.count}; }
		};
	};

	/** @brief This set's path type is ready-made. */
	template<detail::length W>
	inline constexpr bool is_ready_made<marked_path<W>> = true;
}

namespace atl::detail {
	template<typename Derived, std::copyable Value, typename T, typename Tag>
	class top_named_members<Derived, Value, top_clusters::marked_path<T>, Tag> {
		constexpr Derived &self() { return static_cast<Derived &>(*this); }

	public:
		/** @brief Mark or unmark vertex @p v. @complexity amortized O(log n) */
		constexpr void set_marked(uint32 v, bool is_marked) {
			self().update_vertex(v, [is_marked](Value x) {
				x.is_marked = is_marked;
				return x;
			});
		}

		/** @brief Whether vertex @p v is marked. @complexity O(1) */
		[[nodiscard]] constexpr bool is_marked(uint32 v) {
			return self().at_vertex(v).is_marked;
		}

		/** @brief The sum of the distances from v to every marked vertex of its tree. @complexity amortized O(log n) */
		[[nodiscard]] constexpr auto marked_distance_sum(uint32 v) {
			return self().component_fold(v).bottom_sum;
		}

		/** @brief The number of marked vertices in v's tree. @complexity amortized O(log n) */
		[[nodiscard]] constexpr std::size_t marked_count(uint32 v) {
			return self().component_fold(v).count;
		}
	};
}

namespace atl {
	/** @brief Marked vertices with distance sums to them (@ref atl::top_clusters::marked_distance). */
	template<typename W>
	using marked_distance_top_tree = detail::top_tree_of_set<top_clusters::marked_distance<W>, edge_values_t>;
}

namespace atl::top_clusters::detail {
	/** @brief The path type of @ref atl::top_clusters::marked_distance. */
	template<typename Path>
	concept marked_family = is_ready_made<Path> && requires(Path const &p) {
		p.bottom_sum;
		p.count;
	};
}
