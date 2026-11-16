#pragma once
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/** @brief What a weighted-DSU `unite(a, b, w)` did with its constraint. */
	enum class unite_outcome {
		merged,		///< `a` and `b` were in different sets, now joined under the constraint
		consistent, ///< already in one set, and the constraint agrees with it (no constraint added)
		contradicts ///< already in one set, and the constraint disagrees with it (no constraint added)
	};

	/**
	 * @brief Union-find over a group of potentials: each element carries an unknown
	 *        `pot(x)`, `unite(a, b, w)` records `pot(b) = pot(a) · w`, and
	 *        `difference(a, b)` answers `pot(a)⁻¹ · pot(b)` whenever the recorded
	 *        constraints determine it.
	 *
	 * The potential group `(Potential, op, identity, inverse)` need not be
	 * commutative (2x2 matrices, permutations); additively the constraint reads
	 * `x_b - x_a = w` and the answer `x_b - x_a`. Typical uses: difference
	 * constraints, parity / bipartiteness over Z/2, detecting a contradicting
	 * constraint the moment it arrives.
	 *
	 * Union by size with full path compression: amortized O(α(n)) group operations per call.
	 * `find` relinks the path it walks, so it is not const; @ref rollback_weighted_dsu
	 * is the undoable sibling with pure reads.
	 *
	 * Each node stores its parent-or-size `int32` (the same encoding as @ref dsu) and its
	 * weight relative to its parent, `pot(x) = pot(parent) · w[x]`; a root's weight
	 * is the identity. Elements are numbered `[0, n)`; `n` must fit `int32`.
	 *
	 * @pre Documented-only: `op` is associative with two-sided identity `identity`;
	 *      `op(x, inverse(x)) == op(inverse(x), x) == identity`. Equality must
	 *      represent group equality. All evaluated operations and conversions must
	 *      be defined and remain in range, including intermediate sums and inverses.
	 *      No group law is asserted; concepts check const-callability and result conversion.
	 * @note The one-argument constructor requires `Potential{}` to be the identity.
	 *       Storage is O(n).
	 *
	 * @tparam Potential The group's element type.
	 * @tparam Op        The group operation (associative, with identity `identity`).
	 * @tparam Inverse   The group inverse.
	 */
	template<std::copyable Potential = int64, binary_operation<Potential> Op = std::plus<>,
			 inverse_operation<Potential> Inverse = std::negate<>>
		requires std::equality_comparable<Potential>
	class weighted_dsu {
		/**
		 * @brief Whether links and weights share one array of nodes (else two arrays).
		 *
		 * With recursive compression, one
		 * array of `{int32, Potential}` nodes wins for 4-byte potentials (the node is 8
		 * bytes, one load per hop); wider potentials win split, since the descent then
		 * reads only the dense int32 links and touches each weight once, on the way back.
		 * Split links improved int64 random queries from 79.4 to 70.0 ms; compact Z/2
		 * regressed from 57.5 to 66.4 ms (benchmark metadata in docs/weighted-dsu.md).
		 */
		static constexpr bool is_interleaved = sizeof(Potential) <= sizeof(int32);

		struct node {
			int32 parent_or_size; ///< >= 0: parent index; < 0: a root with size -value
			Potential weight;	  ///< pot(x) = pot(parent) · weight
		};

		struct split_nodes {
			std::vector<int32> parent_or_size;
			std::vector<Potential> weight;
		};

		std::conditional_t<is_interleaved, std::vector<node>, split_nodes> nodes_;
		std::size_t count_ = 0;
		Potential identity_{};
		[[no_unique_address]] Op op_{};
		[[no_unique_address]] Inverse inverse_{};

		constexpr int32 &link(std::size_t x) {
			if constexpr (is_interleaved)
				return nodes_[x].parent_or_size;
			else
				return nodes_.parent_or_size[x];
		}

		constexpr Potential &weight(std::size_t x) {
			if constexpr (is_interleaved)
				return nodes_[x].weight;
			else
				return nodes_.weight[x];
		}

		constexpr std::size_t storage_size() const noexcept {
			if constexpr (is_interleaved)
				return nodes_.size();
			else
				return nodes_.parent_or_size.size();
		}

		/** @brief `op(a, b)` converted to `Potential` (the concept allows any convertible result). */
		constexpr Potential combine(Potential const &a, Potential const &b) const { return Potential(std::invoke(op_, a, b)); }

		/**
		 * @brief Point every node from `x` up to the root's child at the root, with its
		 *        weight made root-relative; returns the root.
		 * @pre `x` is not a root. The recursion depth is at most log₂ n ≤ 30, which
		 *      union by size guarantees.
		 */
		constexpr std::size_t compress(std::size_t x) {
			std::size_t const p = std::size_t(link(x));
			if (link(p) < 0) return p;
			std::size_t const root = compress(p);
			weight(x) = combine(weight(p), weight(x));
			link(x) = int32(root);
			return root;
		}

		/** @brief The root of `x` and `pot(x)` relative to that root, compressing the walked path. */
		constexpr std::pair<std::size_t, Potential> locate(std::size_t x) {
			atl_assert(x < storage_size());
			if (link(x) < 0) return {x, identity_};
			return {compress(x), weight(x)};
		}

		// Bind constructor arguments directly to avoid an extra operation or identity transfer.
		struct initialize_t {};

		constexpr weighted_dsu(initialize_t, std::size_t n, Potential &&identity, Op &&op, Inverse &&inverse)
			: count_(n), identity_(std::move(identity)), op_(std::move(op)), inverse_(std::move(inverse)) {
			atl_assert(n <= std::size_t((std::numeric_limits<int32>::max)()));
			if constexpr (is_interleaved)
				nodes_.assign(n, node{int32(-1), identity_});
			else
				nodes_.parent_or_size.assign(n, int32(-1)), nodes_.weight.assign(n, identity_);
		}

	public:
		/** @brief An empty weighted DSU (0 elements). */
		weighted_dsu() = default;

		/**
		 * @brief `n` singletons with identity `Potential{}` (right for additive groups;
		 *        a multiplicative group passes its identity to the other constructor).
		 * @param n Number of elements.
		 * @pre `n <= 2³¹−1` (asserted); the supplied operations satisfy the class group contract.
		 * @complexity O(n)
		 * @note Omitted operations are value-initialized and directly constructed into storage from their rvalues.
		 */
		constexpr explicit weighted_dsu(std::size_t n)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
				  && requires { Potential{}; Op{}; Inverse{}; }
			: weighted_dsu(initialize_t{}, n, Potential{}, Op{}, Inverse{}) {}

		/**
		 * @brief `n` singletons over the group `(Potential, op, identity, inverse)`.
		 * @param n Number of elements.
		 * @param identity Two-sided group identity.
		 * @param op Group composition, directly constructed into storage from its rvalue.
		 * @param inverse Group inversion, directly constructed into storage from its rvalue.
		 * @pre `n <= 2³¹−1`
		 * @complexity O(n)
		 */
		constexpr weighted_dsu(std::size_t n, Potential identity, Op op, Inverse inverse)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
			: weighted_dsu(initialize_t{}, n, std::move(identity), std::move(op), std::move(inverse)) {}

		/**
		 * @brief `n` singletons with a supplied identity and value-initialized operations.
		 * @param n Number of elements.
		 * @param identity Two-sided group identity.
		 * @pre The size bound and group laws of the four-argument constructor apply.
		 * @complexity O(n)
		 */
		constexpr weighted_dsu(std::size_t n, Potential identity)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
				  && requires { Op{}; Inverse{}; }
			: weighted_dsu(initialize_t{}, n, std::move(identity), Op{}, Inverse{}) {}

		/**
		 * @brief `n` singletons with a supplied composition and value-initialized inverse.
		 * @param n Number of elements.
		 * @param identity Two-sided group identity.
		 * @param op Group composition, directly constructed into storage from its rvalue.
		 * @pre The size bound and group laws of the four-argument constructor apply.
		 * @complexity O(n)
		 */
		constexpr weighted_dsu(std::size_t n, Potential identity, Op op)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
				  && requires { Inverse{}; }
			: weighted_dsu(initialize_t{}, n, std::move(identity), std::move(op), Inverse{}) {}

		/** @brief Number of elements. */
		constexpr std::size_t elements() const noexcept { return storage_size(); }

		/** @brief Number of disjoint sets. */
		constexpr std::size_t count() const noexcept { return count_; }

		/**
		 * @brief The representative (root) of `x`'s set.
		 * @param x Element index.
		 * @pre `x < elements()` (asserted).
		 * @complexity O(α(n)) amortized
		 */
		constexpr std::size_t find(std::size_t x) { return locate(x).first; }

		/**
		 * @brief Whether `a` and `b` belong to the same set.
		 * @param a First element index.
		 * @param b Second element index.
		 * @pre `a < elements()` and `b < elements()` (asserted).
		 * @complexity O(α(n)) amortized
		 */
		[[nodiscard]] constexpr bool same(std::size_t a, std::size_t b) { return find(a) == find(b); }

		/**
		 * @brief The number of elements in the set containing `x`.
		 * @param x Element index.
		 * @pre `x < elements()` (asserted).
		 * @complexity O(α(n)) amortized
		 */
		[[nodiscard]] constexpr std::size_t size(std::size_t x) { return std::size_t(-link(find(x))); }

		/**
		 * @brief Record the constraint `pot(b) = pot(a) · w` (additively `x_b - x_a = w`).
		 * @param a Source element index.
		 * @param b Destination element index.
		 * @param w Relative potential, multiplied on the right.
		 * @pre Both indices are less than `elements()` (asserted).
		 * @return @ref unite_outcome::merged if it joined two sets; otherwise whether it
		 *         agrees with what the set already implies (a contradicting constraint
		 *         is dropped; paths may still be compressed).
		 * @complexity O(α(n)) amortized
		 */
		constexpr unite_outcome unite(std::size_t a, std::size_t b, Potential w) {
			auto [ra, pa] = locate(a);
			auto [rb, pb] = locate(b);
			Potential through = combine(pa, w); // pot(b) relative to ra, as the constraint has it
			if (ra == rb) return through == pb ? unite_outcome::consistent : unite_outcome::contradicts;
			// rb's weight under ra is through · pb⁻¹; ra's under rb is its inverse
			Potential lift = combine(through, Potential(std::invoke(inverse_, pb)));
			if (link(ra) > link(rb)) {
				weight(ra) = Potential(std::invoke(inverse_, lift));
				link(rb) += link(ra);
				link(ra) = int32(rb);
			} else {
				weight(rb) = std::move(lift);
				link(ra) += link(rb);
				link(rb) = int32(ra);
			}
			--count_;
			return unite_outcome::merged;
		}

		/**
		 * @brief `pot(a)⁻¹ · pot(b)` (additively `x_b - x_a`), or none if `a` and `b` are
		 *        in different sets and the constraints leave it undetermined.
		 * @param a Source element index.
		 * @param b Destination element index.
		 * @pre Both indices are less than `elements()` (asserted).
		 * @return Relative potential, or `std::nullopt` for different components.
		 * @complexity O(α(n)) amortized
		 */
		[[nodiscard]] constexpr std::optional<Potential> difference(std::size_t a, std::size_t b) {
			auto [ra, pa] = locate(a);
			auto [rb, pb] = locate(b);
			if (ra != rb) return std::nullopt;
			return combine(Potential(std::invoke(inverse_, pa)), pb);
		}
	};
}
