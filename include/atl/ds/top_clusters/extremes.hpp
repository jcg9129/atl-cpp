#pragma once
#include <atl/ds/detail/top_aggregate_members.hpp>
#include <atl/ds/top_clusters/common.hpp>
#include <atl/ds/top_tree_engine.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <concepts>
#include <limits>
#include <utility>

namespace atl::top_clusters::detail {
	/** @brief A `summable` value with an order and numeric limits (the min / max identities). */
	template<typename T>
	concept extremal = summable<T> && std::totally_ordered<T> && std::numeric_limits<T>::is_specialized;

}

namespace atl::top_clusters {
	/** @brief Sum, minimum and maximum of the covered vertex values and their number. */
	template<detail::extremal T>
	struct extremes_aggregate {
		T sum{};								  ///< the sum of the covered values
		T min = (std::numeric_limits<T>::max)();  ///< the minimum (`max()` when empty)
		T max = std::numeric_limits<T>::lowest(); ///< the maximum (`lowest()` when empty)
		uint32 count{};							  ///< the number of covered vertices

		/** @brief Compare every stored field. */
		[[nodiscard]] constexpr bool operator==(extremes_aggregate const &) const = default;
	};

	/** @brief The @ref atl::top_clusters::extremes path cluster: spine and hanging vertices apart. */
	template<detail::extremal T>
	struct extremes_path {
		extremes_aggregate<T> spine; ///< the vertices of the spine
		extremes_aggregate<T> light; ///< every vertex hanging off the spine

		/** @brief Spine and light together: the whole cluster. */
		[[nodiscard]] constexpr extremes_aggregate<T> total() const {
			return {spine.sum + light.sum, light.min < spine.min ? light.min : spine.min,
					spine.max < light.max ? light.max : spine.max, spine.count + light.count};
		}

		/** @brief Compare every stored field. */
		[[nodiscard]] constexpr bool operator==(extremes_path const &) const = default;
	};

	/**
	 * @brief The @ref atl::top_clusters::extremes tag: `x -> (is_assign ? assigned : x) + add`,
	 *        one composable tag for assignments, adds and their mixes.
	 */
	template<detail::extremal T>
	struct extremes_tag {
		bool is_assign = false; ///< every covered value is first replaced by `assigned`
		T assigned{};			///< the assigned value (when `is_assign`)
		T add{};				///< then added to every covered value

		/** @brief The tag adding @p x. */
		[[nodiscard]] static constexpr extremes_tag add_of(T x) { return {false, T{}, std::move(x)}; }

		/** @brief The tag assigning @p x. */
		[[nodiscard]] static constexpr extremes_tag assign_of(T x) { return {true, std::move(x), T{}}; }

		/** @brief Compare every stored field. */
		[[nodiscard]] constexpr bool operator==(extremes_tag const &) const = default;
	};

	/**
	 * @brief Vertex values with sum, min and max over paths, subtrees and
	 *        whole trees, lazy add and assign on paths and subtrees
	 *        (`extremes_top_tree<T>`).
	 * @pre sums stay in range for `T`
	 */
	template<detail::extremal T>
	struct extremes {
		/** @brief Vertex or edge datum accepted by this set. */
		using value_type = T;
		/** @brief Summary of covered vertex values. */
		using aggregate = extremes_aggregate<T>;
		/** @brief Summary of a spine and its hanging subtrees. */
		using path_type = extremes_path<T>;
		/** @brief Summary of subtrees attached at one boundary. */
		using point_type = aggregate;
		/** @brief Lazy action on the covered values. */
		using tag_type = extremes_tag<T>;
		/** @brief Selector or callable reversing the boundary order. */
		using reverse = commutative_fold_t;

		/** @brief Merge two aggregates. */
		[[nodiscard]] static constexpr aggregate merged(aggregate const &a, aggregate const &b) {
			return {a.sum + b.sum, b.min < a.min ? b.min : a.min, a.max < b.max ? b.max : a.max, a.count + b.count};
		}

		/** @brief One tag on an aggregate. */
		[[nodiscard]] static constexpr aggregate mapped(tag_type const &f, aggregate const &a) {
			if (a.count == 0) return a;
			if (f.is_assign) {
				T const x = f.assigned + f.add;
				return {x * T(a.count), x, x, a.count};
			}
			return {a.sum + f.add * T(a.count), a.min + f.add, a.max + f.add, a.count};
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
			[[nodiscard]] constexpr path_type operator()(point_type const &light, T const &value) const {
				return {{value, value, value, 1}, light};
			}
		};

		/** @brief A child's cluster as a hanging set. */
		struct add_edge {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr point_type operator()(path_type const &child) const { return child.total(); }
		};

		/** @brief A tag on one vertex value. */
		struct mapping {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr T operator()(tag_type const &f, T const &value) const { return (f.is_assign ? f.assigned : value) + f.add; }
		};

		/** @brief A (spine, light) tag pair on a path cluster. */
		struct map_path {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr path_type operator()(tag_type const &spine, tag_type const &light, path_type const &p) const {
				return {mapped(spine, p.spine), mapped(light, p.light)};
			}
		};

		/** @brief A tag on a hanging set. */
		struct map_point {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr point_type operator()(tag_type const &f, point_type const &p) const { return mapped(f, p); }
		};

		/** @brief Tag composition, "g first, then f": an assignment in f discards g. */
		struct compose {
			/** @brief Evaluate this cluster operation; inputs obey the enclosing set's laws. */
			[[nodiscard]] constexpr tag_type operator()(tag_type const &f, tag_type const &g) const {
				if (f.is_assign) return f;
				return {g.is_assign, g.assigned, g.add + f.add};
			}
		};
	};

	/** @brief This set's path type is ready-made. */
	template<detail::extremal T>
	inline constexpr bool is_ready_made<extremes_path<T>> = true;

}

namespace atl::detail {
	template<typename Derived, std::copyable Value, typename T, typename Tag>
	class top_named_members<Derived, Value, top_clusters::extremes_path<T>, Tag> : public top_aggregate_members<Derived, Value, Tag> {
		constexpr Derived &self() { return static_cast<Derived &>(*this); }

	public:
		/** @brief The minimum vertex value on the u - v path. @complexity amortized O(log n) */
		[[nodiscard]] constexpr Value path_min(uint32 u, uint32 v) {
			return self().fold(u, v).spine.min;
		}

		/** @brief The maximum vertex value on the u - v path. @complexity amortized O(log n) */
		[[nodiscard]] constexpr Value path_max(uint32 u, uint32 v) {
			return self().fold(u, v).spine.max;
		}

		/** @brief The minimum over v's subtree. @complexity amortized O(log n) */
		[[nodiscard]] constexpr Value subtree_min(uint32 v) {
			return self().subtree_fold(v).total().min;
		}

		/** @brief The minimum over v's side of the edge {v, p}. @complexity amortized O(log n) */
		[[nodiscard]] constexpr Value subtree_min(uint32 v, uint32 p) {
			return self().subtree_fold(v, p).total().min;
		}

		/** @brief The maximum over v's subtree. @complexity amortized O(log n) */
		[[nodiscard]] constexpr Value subtree_max(uint32 v) {
			return self().subtree_fold(v).total().max;
		}

		/** @brief The maximum over v's side of the edge {v, p}. @complexity amortized O(log n) */
		[[nodiscard]] constexpr Value subtree_max(uint32 v, uint32 p) {
			return self().subtree_fold(v, p).total().max;
		}

		/** @brief The minimum over v's tree. @complexity amortized O(log n) */
		[[nodiscard]] constexpr Value component_min(uint32 v) {
			return self().component_fold(v).total().min;
		}

		/** @brief The maximum over v's tree. @complexity amortized O(log n) */
		[[nodiscard]] constexpr Value component_max(uint32 v) {
			return self().component_fold(v).total().max;
		}

		/** @brief Set every vertex on the u - v path to @p x. @complexity amortized O(log n) */
		constexpr void path_assign(uint32 u, uint32 v, Value const &x) {
			self().apply(u, v, Tag::assign_of(x));
		}

		/** @brief Set every vertex of v's subtree to @p x. @complexity amortized O(log n) */
		constexpr void subtree_assign(uint32 v, Value const &x) {
			self().subtree_apply(v, Tag::assign_of(x));
		}

		/** @brief Set every vertex on v's side of the edge {v, p} to @p x. @complexity amortized O(log n) */
		constexpr void subtree_assign(uint32 v, uint32 p, Value const &x) {
			self().subtree_apply(v, p, Tag::assign_of(x));
		}
	};
}

namespace atl {
	/** @brief Sums, minima and maxima with lazy add and assign (@ref atl::top_clusters::extremes). */
	template<typename T>
	using extremes_top_tree = detail::top_tree_of_set<top_clusters::extremes<T>>;
}

namespace atl::top_clusters::detail {
	/** @brief The path type of @ref atl::top_clusters::extremes : the min / max / assign named members. */
	template<typename Path>
	concept extremes_family = sum_family<Path> && requires(Path const &p) { p.spine.min; };
}
