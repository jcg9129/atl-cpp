#pragma once
#include <atl/ds/detail/top_aggregate_members.hpp>
#include <atl/ds/top_clusters/common.hpp>
#include <atl/ds/top_tree_engine.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <concepts>
#include <utility>

namespace atl::top_clusters {
	/** @brief A sum and the number of vertices it covers. */
	template<detail::summable T>
	struct sum_aggregate {
		T sum{};		///< the fold of the covered vertex values
		uint32 count{}; ///< the number of covered vertices

		/** @brief Compare every stored field. */
		[[nodiscard]] constexpr bool operator==(sum_aggregate const &) const = default;
	};

	/**
	 * @brief The @ref atl::top_clusters::sum path cluster: the spine's vertices and everything
	 *        hanging off the spine, kept apart so a path tag and a subtree tag
	 *        map them separately.
	 */
	template<detail::summable T>
	struct sum_path {
		sum_aggregate<T> spine; ///< the vertices of the spine
		sum_aggregate<T> light; ///< every vertex hanging off the spine

		/** @brief Spine and light together: the whole cluster. */
		[[nodiscard]] constexpr sum_aggregate<T> total() const { return {spine.sum + light.sum, spine.count + light.count}; }

		/** @brief Compare every stored field. */
		[[nodiscard]] constexpr bool operator==(sum_path const &) const = default;
	};

	/** @brief The @ref atl::top_clusters::sum tag: add `add` to every covered vertex. */
	template<detail::summable T>
	struct sum_tag {
		T add{}; ///< the amount added to each vertex

		/** @brief The tag adding @p x. */
		[[nodiscard]] static constexpr sum_tag add_of(T x) { return {std::move(x)}; }

		/** @brief Compare every stored field. */
		[[nodiscard]] constexpr bool operator==(sum_tag const &) const = default;
	};

	/**
	 * @brief Vertex values with sums and sizes over paths, subtrees and whole
	 *        trees, and lazy adds on paths and subtrees (`sum_top_tree<T>`).
	 *        The fold is a sum, so it commutes (@ref atl::commutative_fold_t).
	 * @pre sums stay in range for `T` (an add tag multiplies by a count)
	 */
	template<detail::summable T>
	struct sum {
		/** @brief Vertex or edge datum accepted by this set. */
		using value_type = T;
		/** @brief Summary of a spine and its hanging subtrees. */
		using path_type = sum_path<T>;
		/** @brief Summary of subtrees attached at one boundary. */
		using point_type = sum_aggregate<T>;
		/** @brief Lazy action on the covered values. */
		using tag_type = sum_tag<T>;
		/** @brief Selector or callable reversing the boundary order. */
		using reverse = commutative_fold_t;

		/** @brief Sum two aggregates. */
		[[nodiscard]] static constexpr point_type merged(point_type const &a, point_type const &b) {
			return {a.sum + b.sum, a.count + b.count};
		}

		/** @brief Add @p add to every vertex an aggregate covers. */
		[[nodiscard]] static constexpr point_type mapped(T const &add, point_type const &a) {
			return {a.sum + add * T(a.count), a.count};
		}

		/** @brief Join two path clusters. */
		struct compress {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr path_type operator()(path_type const &upper, path_type const &lower) const {
				return {merged(upper.spine, lower.spine), merged(upper.light, lower.light)};
			}
		};

		/** @brief Merge two hanging sets. */
		struct rake {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr point_type operator()(point_type const &a, point_type const &b) const { return merged(a, b); }
		};

		/** @brief A vertex over its hanging set. */
		struct add_vertex {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr path_type operator()(point_type const &light, T const &value) const { return {{value, 1}, light}; }
		};

		/** @brief A child's cluster as a hanging set. */
		struct add_edge {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr point_type operator()(path_type const &child) const { return child.total(); }
		};

		/** @brief A tag on one vertex value. */
		struct mapping {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr T operator()(tag_type const &f, T const &value) const { return value + f.add; }
		};

		/** @brief A (spine, light) tag pair on a path cluster. */
		struct map_path {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr path_type operator()(tag_type const &spine, tag_type const &light, path_type const &p) const {
				return {mapped(spine.add, p.spine), mapped(light.add, p.light)};
			}
		};

		/** @brief A tag on a hanging set. */
		struct map_point {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr point_type operator()(tag_type const &f, point_type const &p) const { return mapped(f.add, p); }
		};

		/** @brief Tag composition, "g first, then f". */
		struct compose {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr tag_type operator()(tag_type const &f, tag_type const &g) const { return {f.add + g.add}; }
		};
	};

	/** @brief This set's path type is ready-made. */
	template<detail::summable T>
	inline constexpr bool is_ready_made<sum_path<T>> = true;

}

namespace atl::detail {
	template<typename Derived, std::copyable Value, typename T, typename Tag>
	class top_named_members<Derived, Value, top_clusters::sum_path<T>, Tag> : public top_aggregate_members<Derived, Value, Tag> {
	};
}

namespace atl {
	/** @brief Vertex values with path / subtree / component sums and sizes and lazy adds (@ref atl::top_clusters::sum). */
	template<typename T>
	using sum_top_tree = detail::top_tree_of_set<top_clusters::sum<T>>;
}
