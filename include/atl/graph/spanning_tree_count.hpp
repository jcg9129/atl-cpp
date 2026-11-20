#pragma once

#include <atl/graph/concepts.hpp>
#include <atl/nt/basic.hpp>
#include <atl/nt/gaussian_elimination.hpp>
#include <atl/nt/matrix.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <cstddef>
#include <limits>
#include <utility>

namespace atl::graph {
	namespace detail {
		/**
		 * @brief The determinant of a reduced Laplacian modulo `Z::mod()`:
		 *        field elimination when the modulus is prime, Euclid row
		 *        reduction (@ref atl::determinant_any_modulus) otherwise.
		 */
		template<modular_integer Z>
		[[nodiscard]] constexpr Z laplacian_determinant(matrix<Z> laplacian) {
			if (is_prime(uint64(Z::mod()))) return atl::determinant(std::move(laplacian));
			return atl::determinant_any_modulus(std::move(laplacian));
		}

		enum class laplacian_kind {
			undirected, ///< both endpoints' degrees; A symmetric
			in_degree,	///< `u -> v` counts toward `v`: out-arborescences
			out_degree	///< `u -> v` counts toward `u`: in-arborescences
		};

		/**
		 * @brief The Laplacian `D - A` of a multigraph with row and column `root`
		 *        deleted: `(n-1) x (n-1)`, vertex `v` at index `v - (v > root)`.
		 *        Self-loops add nothing (they cancel in `D - A`).
		 * @pre `vertex_count >= 1`, `root < vertex_count`, endpoints `< vertex_count` (asserted).
		 */
		template<laplacian_kind Kind, modular_integer Z, edge_list EdgeList>
		[[nodiscard]] constexpr matrix<Z> reduced_laplacian(std::size_t vertex_count, EdgeList const &edges, vertex_id root) {
			atl_assert(vertex_count >= 1 && vertex_count <= std::numeric_limits<uint32>::max());
			atl_assert(root < vertex_count);
			atl_assert(Z::mod() > 0 && Z::mod() <= std::numeric_limits<uint64>::max());
			matrix<Z> laplacian(vertex_count - 1, vertex_count - 1);
			Z const one = Z(uint64(1));
			auto const add_degree = [&](vertex_id v) {
				if (v != root) laplacian(v - uint32(v > root), v - uint32(v > root)) += one;
			};
			auto const sub_adjacency = [&](vertex_id u, vertex_id v) {
				if (u != root && v != root) laplacian(u - uint32(u > root), v - uint32(v > root)) -= one;
			};
			for (auto const &e : edges) {
				vertex_id const u = graph::source(e), v = graph::target(e);
				atl_assert(u < vertex_count && v < vertex_count);
				if (u == v) continue;
				if constexpr (Kind != laplacian_kind::out_degree) add_degree(v);
				if constexpr (Kind != laplacian_kind::in_degree) add_degree(u);
				sub_adjacency(u, v);
				if constexpr (Kind == laplacian_kind::undirected) sub_adjacency(v, u);
			}
			return laplacian;
		}
	}

	/**
	 * @brief The number of spanning trees of an undirected multigraph, modulo
	 *        `Z::mod()` (Kirchhoff's matrix-tree theorem: any cofactor of the
	 *        Laplacian). Parallel edges count as distinct, self-loops never
	 *        lie in a tree; a disconnected graph has 0.
	 *
	 * Any modulus: prime ones eliminate over the field, others take the
	 * Euclid row reduction.
	 *
	 * @tparam Z Modular integer type; prime moduli use a field, composite moduli use Euclidean reduction.
	 * @tparam EdgeList Const forward range of list edges, including proxy references.
	 * @param vertex_count The number of vertices `n`; edges use ids `[0, n)`.
	 * @param edges Each element one undirected edge, endpoints via
	 *              `graph::source` / `graph::target`.
	 * @pre 1 <= vertex_count <= UINT32_MAX, every endpoint < vertex_count,
	 * and (vertex_count-1)^2 <= `vector<Z>::max_size()` (asserted).
	 * The modulus is valid for Z and at most UINT64_MAX (asserted).
	 * @return Count modulo Z::mod(); the singleton graph has one empty tree.
	 * Input is neither modified nor retained; allocation failures propagate.
	 * @complexity O(n³ + m) for prime modulus; O(n³ + n² log(mod+1) + m) otherwise,
	 * O(n²) heap storage and O(1) call stack; no recursion per vertex.
	 */
	template<modular_integer Z, edge_list EdgeList>
	[[nodiscard]] constexpr Z count_spanning_trees(std::size_t vertex_count, EdgeList const &edges) {
		return detail::laplacian_determinant(detail::reduced_laplacian<detail::laplacian_kind::undirected, Z>(vertex_count, edges, 0));
	}

	/**
	 * @brief The number of spanning out-arborescences rooted at `root` of a
	 *        directed multigraph (every vertex reachable from `root`, every
	 *        edge pointing away from it), modulo `Z::mod()`: the directed
	 *        matrix-tree theorem on the in-degree Laplacian.
	 *
	 * @tparam Z Modular integer type with modulus at most UINT64_MAX.
	 * @tparam EdgeList Const forward range of list edges, including proxy references.
	 * @param vertex_count Number of vertices.
	 * @param root Root vertex.
	 * @param edges Each element one arc `source -> target`.
	 * @pre 1 <= vertex_count <= UINT32_MAX, root < vertex_count, endpoints < vertex_count,
	 * (vertex_count-1)^2 <= `vector<Z>::max_size()`, modulus <= UINT64_MAX (asserted).
	 * Z obeys modular arithmetic laws and has a valid modulus.
	 * @return Count modulo Z::mod(); the singleton graph has one empty tree.
	 * Input is neither modified nor retained; allocation failures propagate.
	 * @complexity O(n³ + m) for prime modulus; O(n³ + n² log(mod+1) + m) otherwise,
	 * O(n²) heap storage and O(1) call stack; no recursion per vertex.
	 */
	template<modular_integer Z, edge_list EdgeList>
	[[nodiscard]] constexpr Z count_out_arborescences(std::size_t vertex_count, EdgeList const &edges, vertex_id root) {
		return detail::laplacian_determinant(detail::reduced_laplacian<detail::laplacian_kind::in_degree, Z>(vertex_count, edges, root));
	}

	/**
	 * @brief The number of spanning in-arborescences rooted at `root` of a
	 *        directed multigraph (every vertex reaches `root`, every edge
	 *        pointing toward it), modulo `Z::mod()`: the directed matrix-tree
	 *        theorem on the out-degree Laplacian.
	 *
	 * @tparam Z Modular integer type with modulus at most UINT64_MAX.
	 * @tparam EdgeList Const forward range of list edges, including proxy references.
	 * @param vertex_count Number of vertices.
	 * @param root Root vertex.
	 * @param edges Each element one arc `source -> target`.
	 * @pre 1 <= vertex_count <= UINT32_MAX, root < vertex_count, endpoints < vertex_count,
	 * (vertex_count-1)^2 <= `vector<Z>::max_size()`, modulus <= UINT64_MAX (asserted).
	 * Z obeys modular arithmetic laws and has a valid modulus.
	 * @return Count modulo Z::mod(); the singleton graph has one empty tree.
	 * Input is neither modified nor retained; allocation failures propagate.
	 * @complexity O(n³ + m) for prime modulus; O(n³ + n² log(mod+1) + m) otherwise,
	 * O(n²) heap storage and O(1) call stack; no recursion per vertex.
	 */
	template<modular_integer Z, edge_list EdgeList>
	[[nodiscard]] constexpr Z count_in_arborescences(std::size_t vertex_count, EdgeList const &edges, vertex_id root) {
		return detail::laplacian_determinant(
			detail::reduced_laplacian<detail::laplacian_kind::out_degree, Z>(vertex_count, edges, root));
	}
}
