#pragma once
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <concepts>
#include <limits>
#include <utility>

/**
 * @brief Ready-made cluster algebras for @ref atl::top_tree : library `Value`,
 *        `Path`, `Point` and `Tag` types with their merges, identities
 *        given by the types' default values. A @ref atl::top_tree over one of
 *        them gets the set's named members (`path_sum`, `diameter`,
 *        `marked_distance_sum`, ...), switched on by requires-clauses, and
 *        constructs from a vertex count (or values) alone; each set has an
 *        alias spelling every slot (`atl::sum_top_tree<T>`, ...).
 */
namespace atl::top_clusters {
	namespace detail {
		/**
		 * @brief A value the summing sets can hold: regular, closed under `+`
		 *        and `*`, and constructible from a vertex count (the add tags
		 *        multiply by the number of vertices they cover).
		 */
		template<typename T>
		concept summable = std::regular<T> && std::constructible_from<T, uint32> && requires(T const &a, T const &b) {
			{ a + b } -> std::convertible_to<T>;
			{ a * b } -> std::convertible_to<T>;
		};

		/** @brief A `summable` value with an order and numeric limits (the min / max identities). */
		template<typename T>
		concept extremal = summable<T> && std::totally_ordered<T> && std::numeric_limits<T>::is_specialized;

		/** @brief An edge length of the distance sets: `summable` and ordered. */
		template<typename W>
		concept length = summable<W> && std::totally_ordered<W>;
	}

	/**
	 * @brief Whether @p Path is the path type of a ready-made set: then a
	 *        @ref atl::top_tree over it takes every identity from the types'
	 *        default values and constructs from a vertex count alone.
	 */
	template<typename Path>
	inline constexpr bool is_ready_made = false;

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

	namespace detail {
		/** @brief The path type of @ref atl::top_clusters::sum or @ref atl::top_clusters::extremes : the summing named members. */
		template<typename Path>
		concept sum_family = is_ready_made<Path> && requires(Path const &p) {
			p.spine.sum;
			p.light.count;
		};

		/** @brief The path type of @ref atl::top_clusters::extremes : the min / max / assign named members. */
		template<typename Path>
		concept extremes_family = sum_family<Path> && requires(Path const &p) { p.spine.min; };

		/** @brief The path type of @ref atl::top_clusters::diameter. */
		template<typename Path>
		concept diameter_family = is_ready_made<Path> && requires(Path const &p) {
			p.diameter;
			p.from_bottom;
		};

		/** @brief The path type of @ref atl::top_clusters::marked_distance. */
		template<typename Path>
		concept marked_family = is_ready_made<Path> && requires(Path const &p) {
			p.bottom_sum;
			p.count;
		};

		/** @brief A value with a per-vertex starting value (`Value::of_vertex(v)`). */
		template<typename Value>
		concept per_vertex_value = requires(uint32 v) {
			{ Value::of_vertex(v) } -> std::convertible_to<Value>;
		};
	}
}
