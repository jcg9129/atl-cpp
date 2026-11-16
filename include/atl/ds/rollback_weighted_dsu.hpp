#pragma once
#include <atl/ds/rollback_dsu.hpp>
#include <atl/ds/weighted_dsu.hpp>
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
	/**
	 * @brief @ref weighted_dsu with rollback: merges can be undone in LIFO order.
	 *
	 * Same constraints and answers as @ref weighted_dsu (`unite(a, b, w)` records
	 * `pot(b) = pot(a) · w`, `difference(a, b)` is `pot(a)⁻¹ · pot(b)`), with
	 * @ref rollback_dsu's history: union by size and **no path compression**, so
	 * reads are const and O(log n) group operations, and `time` / `undo` /
	 * `rollback` work exactly as there. Undoing a merge resets the child root's
	 * weight to the identity (a root's weight is always the identity), so the log
	 * holds the same 12-byte @ref detail::dsu_undo entries as @ref rollback_dsu.
	 * Typical use: offline dynamic bipartiteness over Z/2.
	 *
	 * @pre Documented-only: `op` is associative with two-sided identity `identity`;
	 *      `op(x, inverse(x)) == op(inverse(x), x) == identity`. Equality must
	 *      represent group equality. All evaluated operations and conversions must
	 *      be defined and remain in range, including intermediate sums and inverses.
	 *      No group law is asserted; concepts check const-callability and result conversion.
	 * @note The one-argument constructor requires `Potential{}` to be the identity.
	 *       Storage is O(n), including at most n-1 live rollback entries.
	 *
	 * @tparam Potential The group's element type.
	 * @tparam Op The group operation (associative, with identity `identity`).
	 * @tparam Inverse The group inverse.
	 */
	template<std::copyable Potential = int64, binary_operation<Potential> Op = std::plus<>,
			 inverse_operation<Potential> Inverse = std::negate<>>
		requires std::equality_comparable<Potential>
	class rollback_weighted_dsu {
		/** @brief Keep compact potentials with their links; wider potentials use dense split links. */
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
		std::vector<detail::dsu_undo> history_;
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

		constexpr int32 link(std::size_t x) const {
			if constexpr (is_interleaved)
				return nodes_[x].parent_or_size;
			else
				return nodes_.parent_or_size[x];
		}

		constexpr Potential const &weight(std::size_t x) const {
			if constexpr (is_interleaved)
				return nodes_[x].weight;
			else
				return nodes_.weight[x];
		}

		constexpr void assign(std::size_t x, int32 parent_or_size, Potential w) {
			if constexpr (is_interleaved)
				nodes_[x] = node{parent_or_size, std::move(w)};
			else
				nodes_.parent_or_size[x] = parent_or_size, nodes_.weight[x] = std::move(w);
		}

		constexpr std::size_t storage_size() const noexcept {
			if constexpr (is_interleaved)
				return nodes_.size();
			else
				return nodes_.parent_or_size.size();
		}

		/** @brief `op(a, b)` converted to `Potential`. */
		constexpr Potential combine(Potential const &a, Potential const &b) const { return Potential(std::invoke(op_, a, b)); }

		/** @brief The root of `x` and `pot(x)` relative to that root. */
		constexpr std::pair<std::size_t, Potential> locate(std::size_t x) const {
			atl_assert(x < storage_size());
			if (link(x) < 0) return {x, identity_};
			Potential total = weight(x);
			x = std::size_t(link(x));
			while (link(x) >= 0) {
				total = combine(weight(x), total);
				x = std::size_t(link(x));
			}
			return {x, std::move(total)};
		}

		// Bind constructor arguments directly to avoid an extra operation or identity transfer.
		struct initialize_t {};

		constexpr rollback_weighted_dsu(initialize_t, std::size_t n, Potential &&identity, Op &&op, Inverse &&inverse)
			: count_(n), identity_(std::move(identity)), op_(std::move(op)), inverse_(std::move(inverse)) {
			atl_assert(n <= std::size_t((std::numeric_limits<int32>::max)()));
			if constexpr (is_interleaved)
				nodes_.assign(n, node{int32(-1), identity_});
			else
				nodes_.parent_or_size.assign(n, int32(-1)), nodes_.weight.assign(n, identity_);
			history_.reserve(n); // a spanning forest merges at most n-1 times
		}

	public:
		/** @brief An empty weighted DSU (0 elements). */
		rollback_weighted_dsu() = default;

		/**
		 * @brief `n` singletons with identity `Potential{}` (right for additive groups;
		 *        a multiplicative group passes its identity to the other constructor).
		 * @param n Number of elements.
		 * @pre `n <= 2³¹−1` (asserted); the supplied operations satisfy the class group contract.
		 * @complexity O(n)
		 * @note Omitted operations are value-initialized and directly constructed into storage from their rvalues.
		 */
		constexpr explicit rollback_weighted_dsu(std::size_t n)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
				  && requires { Potential{}; Op{}; Inverse{}; }
			: rollback_weighted_dsu(initialize_t{}, n, Potential{}, Op{}, Inverse{}) {}

		/**
		 * @brief `n` singletons over the group `(Potential, op, identity, inverse)`.
		 * @param n Number of elements.
		 * @param identity Two-sided group identity.
		 * @param op Group composition, directly constructed into storage from its rvalue.
		 * @param inverse Group inversion, directly constructed into storage from its rvalue.
		 * @pre `n <= 2³¹−1` (the int32 element storage, which also bounds the uint32 undo log)
		 * @complexity O(n)
		 */
		constexpr rollback_weighted_dsu(std::size_t n, Potential identity, Op op, Inverse inverse)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
			: rollback_weighted_dsu(initialize_t{}, n, std::move(identity), std::move(op), std::move(inverse)) {}

		/**
		 * @brief `n` singletons with a supplied identity and value-initialized operations.
		 * @param n Number of elements.
		 * @param identity Two-sided group identity.
		 * @pre The size bound and group laws of the four-argument constructor apply.
		 * @complexity O(n)
		 */
		constexpr rollback_weighted_dsu(std::size_t n, Potential identity)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
				  && requires { Op{}; Inverse{}; }
			: rollback_weighted_dsu(initialize_t{}, n, std::move(identity), Op{}, Inverse{}) {}

		/**
		 * @brief `n` singletons with a supplied composition and value-initialized inverse.
		 * @param n Number of elements.
		 * @param identity Two-sided group identity.
		 * @param op Group composition, directly constructed into storage from its rvalue.
		 * @pre The size bound and group laws of the four-argument constructor apply.
		 * @complexity O(n)
		 */
		constexpr rollback_weighted_dsu(std::size_t n, Potential identity, Op op)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
				  && requires { Inverse{}; }
			: rollback_weighted_dsu(initialize_t{}, n, std::move(identity), std::move(op), Inverse{}) {}

		/** @brief Number of elements. */
		constexpr std::size_t elements() const noexcept { return storage_size(); }

		/** @brief Number of disjoint sets. */
		constexpr std::size_t count() const noexcept { return count_; }

		/**
		 * @brief The representative (root) of `x`'s set.
		 * @param x Element index.
		 * @pre `x < elements()` (asserted).
		 * @complexity O(log n)
		 */
		[[nodiscard]] constexpr std::size_t find(std::size_t x) const {
			atl_assert(x < storage_size());
			while (link(x) >= 0) x = std::size_t(link(x));
			return x;
		}

		/**
		 * @brief Whether `a` and `b` belong to the same set.
		 * @param a First element index.
		 * @param b Second element index.
		 * @pre `a < elements()` and `b < elements()` (asserted).
		 * @complexity O(log n)
		 */
		[[nodiscard]] constexpr bool same(std::size_t a, std::size_t b) const { return find(a) == find(b); }

		/**
		 * @brief The number of elements in the set containing `x`.
		 * @param x Element index.
		 * @pre `x < elements()` (asserted).
		 * @complexity O(log n)
		 */
		[[nodiscard]] constexpr std::size_t size(std::size_t x) const { return std::size_t(-link(find(x))); }

		/**
		 * @brief Record the constraint `pot(b) = pot(a) · w` (additively `x_b - x_a = w`).
		 * @param a Source element index.
		 * @param b Destination element index.
		 * @param w Relative potential, multiplied on the right.
		 * @pre Both indices are less than `elements()` (asserted).
		 * @return @ref unite_outcome::merged if it joined two sets (only these are
		 *         recorded, so only these are undone by @ref undo); otherwise whether
		 *         it agrees with what the set already implies.
		 * @complexity O(log n)
		 */
		constexpr unite_outcome unite(std::size_t a, std::size_t b, Potential w) {
			auto [ra, pa] = locate(a);
			auto [rb, pb] = locate(b);
			Potential through = combine(pa, w); // pot(b) relative to ra, as the constraint has it
			if (ra == rb) return through == pb ? unite_outcome::consistent : unite_outcome::contradicts;
			// rb's weight under ra is through · pb⁻¹; ra's under rb is its inverse
			Potential lift = combine(through, Potential(std::invoke(inverse_, pb)));
			int32 sa = -link(ra), sb = -link(rb);
			if (sa < sb) {
				std::swap(ra, rb), std::swap(sa, sb);
				lift = Potential(std::invoke(inverse_, lift));
			}
			history_.push_back({uint32(rb), uint32(ra), uint32(sb)});
			assign(rb, int32(ra), std::move(lift));
			link(ra) = -(sa + sb);
			--count_;
			return unite_outcome::merged;
		}

		/**
		 * @brief `pot(a)⁻¹ · pot(b)` (additively `x_b - x_a`), or none if `a` and `b` are
		 *        in different sets.
		 * @param a Source element index.
		 * @param b Destination element index.
		 * @pre Both indices are less than `elements()` (asserted).
		 * @return Relative potential, or `std::nullopt` for different components.
		 * @complexity O(log n) group operations.
		 */
		[[nodiscard]] constexpr std::optional<Potential> difference(std::size_t a, std::size_t b) const {
			auto [ra, pa] = locate(a);
			auto [rb, pb] = locate(b);
			if (ra != rb) return std::nullopt;
			return combine(Potential(std::invoke(inverse_, pa)), pb);
		}

		/**
		 * @brief An opaque token for the current history length; pass it to
		 *        @ref rollback to return here later. Equals the number of merges
		 *        performed and not yet undone.
		 */
		constexpr std::size_t time() const noexcept { return history_.size(); }

		/**
		 * @brief Undo the most recent merge.
		 * @pre `time() > 0` (asserted).
		 * @complexity O(1)
		 */
		constexpr void undo() {
			atl_assert(!history_.empty());
			auto const [child, root, child_size] = history_.back();
			history_.pop_back();
			link(root) += int32(child_size);
			assign(child, -int32(child_size), identity_);
			++count_;
		}

		/**
		 * @brief Undo merges until `time() == t`, where `t` came from an earlier
		 *        @ref time call.
		 * @param t Token from this object's still-live history prefix.
		 * @pre `t <= time()` (asserted); provenance and a still-live prefix are documented-only.
		 * Tokens from an abandoned branch do not identify its state after new merges.
		 * @complexity O(time() - t)
		 */
		constexpr void rollback(std::size_t t) {
			atl_assert(t <= history_.size());
			while (history_.size() > t) undo();
		}
	};
}
