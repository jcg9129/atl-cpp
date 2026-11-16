/**
 * @file
 * @brief Weighted DSU by selectable compression and node layout: two-pass buffer,
 * two-pass inverse, halving, splitting, or recursion, with split or interleaved storage.
 * Root-child exit can be disabled to isolate that branch. Each strategy preserves
 * the library's group contract. Compression-strategy race: not measured yet.
 * Rollback by interleaved nodes retains a comparison against dense split links;
 * recorded numbers and the mapping to the current library: docs/weighted-dsu.md.
 */
#pragma once

#include <atl/ds/rollback_dsu.hpp>
#include <atl/ds/weighted_dsu.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::experimental {
	/** @brief How a find walk relinks the path it walks. */
	enum class weighted_dsu_compression {
		/** @brief Every node on the path points at the root: the path goes into a fixed 32-slot
		 * buffer, then the root-relative potentials are rebuilt downward from the root. */
		two_pass_buffer,
		/** @brief Every node on the path points at the root, the `dsu` find kept as is: pass
		 * one multiplies the whole path, pass two peels each node's own potential off the
		 * running product with one `inverse` per relinked node. */
		two_pass_inverse,
		/** @brief Path halving: each visited node jumps to its grandparent, one pass, no inverse. */
		halving,
		/** @brief Path splitting: every node on the path jumps to its grandparent (halving relinks
		 * every other one), one pass, no inverse. */
		splitting,
		/** @brief Every node on the path points at the root, by recursion (depth <= 30 under
		 * union by size). */
		recursive,
	};

	/** @brief Where a node's parent link and potential live. */
	enum class weighted_dsu_layout {
		interleaved, ///< one array of `{int32, Potential}` nodes
		split,		 ///< an `int32` array and a `Potential` array
	};

	/**
	 * @brief @ref atl::weighted_dsu with the find strategy and the node layout as
	 *        template parameters; same interface and semantics.
	 * @tparam Compression Path relinking strategy.
	 * @tparam Layout Placement of links and weights.
	 * @tparam Potential Group element type.
	 * @tparam Op Associative group composition.
	 * @tparam Inverse Two-sided inverse.
	 * @tparam HasRootChildExit Whether to return early for a root child.
	 * @pre The group laws and value bounds of atl::weighted_dsu apply; they are not asserted.
	 *
	 * `HasRootChildExit` adds an early return when `x`'s parent is already the root
	 * (the common case once paths are compressed), the tuning knob of the race.
	 */
	template<weighted_dsu_compression Compression, weighted_dsu_layout Layout, std::copyable Potential = int64,
			 binary_operation<Potential> Op = std::plus<>, inverse_operation<Potential> Inverse = std::negate<>,
			 bool HasRootChildExit = true>
		requires std::equality_comparable<Potential>
	class weighted_dsu_by_compression_and_layout {
		struct node {
			int32 parent_or_size; ///< >= 0: parent index; < 0: a root with size -value
			Potential weight;	  ///< pot(x) = pot(parent) · weight
		};

		struct split_nodes {
			std::vector<int32> parent_or_size;
			std::vector<Potential> weight;
		};

		/** @brief Whether links and weights share one array of nodes (else two arrays). */
		static constexpr bool is_interleaved = Layout == weighted_dsu_layout::interleaved;

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

		/** @brief Point every node above `x` (exclusive of the root's child) at the root `x` reaches; returns that root. */
		constexpr std::size_t compress_recursive(std::size_t x) {
			std::size_t const p = std::size_t(link(x));
			if (link(p) < 0) return p;
			std::size_t const root = compress_recursive(p);
			weight(x) = combine(weight(p), weight(x));
			link(x) = int32(root);
			return root;
		}

		/** @brief The root of `x` and `x`'s potential relative to it, relinking the walked path. */
		constexpr std::pair<std::size_t, Potential> locate(std::size_t x) {
			atl_assert(x < storage_size());
			if (link(x) < 0) return {x, identity_};
			if constexpr (HasRootChildExit)
				if (std::size_t const p = std::size_t(link(x)); link(p) < 0) return {p, weight(x)};
			if constexpr (Compression == weighted_dsu_compression::two_pass_buffer) {
				std::array<uint32, 32> path;
				std::size_t depth = 0;
				std::size_t root = x;
				while (link(root) >= 0) {
					atl_assert(depth < path.size()); // union by size keeps depth <= 30
					path[depth++] = uint32(root);
					root = std::size_t(link(root));
				}
				// path[depth - 1] is the root's child, already root-relative
				for (std::size_t i = depth - 1; i-- > 0;) {
					weight(path[i]) = combine(weight(path[i + 1]), weight(path[i]));
					link(path[i]) = int32(root);
				}
				return {root, weight(x)};
			} else if constexpr (Compression == weighted_dsu_compression::two_pass_inverse) {
				Potential total = weight(x);
				std::size_t root = std::size_t(link(x));
				while (link(root) >= 0) total = combine(weight(root), total), root = std::size_t(link(root));
				Potential running = total;
				for (std::size_t y = x; std::size_t(link(y)) != root;) {
					std::size_t const next = std::size_t(link(y));
					Potential const own = std::move(weight(y));
					weight(y) = running;
					link(y) = int32(root);
					running = combine(running, Potential(std::invoke(inverse_, own)));
					y = next;
				}
				return {root, std::move(total)};
			} else if constexpr (Compression == weighted_dsu_compression::halving) {
				auto halve = [&](std::size_t y) {
					std::size_t const p = std::size_t(link(y));
					if (link(p) >= 0) {
						weight(y) = combine(weight(p), weight(y));
						link(y) = link(p);
					}
				};
				halve(x);
				Potential total = weight(x);
				x = std::size_t(link(x));
				while (link(x) >= 0) {
					halve(x);
					total = combine(weight(x), total);
					x = std::size_t(link(x));
				}
				return {x, std::move(total)};
			} else if constexpr (Compression == weighted_dsu_compression::splitting) {
				Potential total = weight(x);
				for (;;) {
					std::size_t const p = std::size_t(link(x));
					if (link(p) < 0) return {p, std::move(total)};
					// x skips p; p's own weight joins the running total before p moves on
					total = combine(weight(p), total);
					weight(x) = combine(weight(p), weight(x));
					link(x) = link(p);
					x = p;
				}
			} else {
				std::size_t const root = compress_recursive(x);
				return {root, weight(x)};
			}
		}

		// Bind constructor arguments directly to avoid an extra operation or identity transfer.
		struct initialize_t {};

		constexpr weighted_dsu_by_compression_and_layout(initialize_t, std::size_t n, Potential &&identity, Op &&op, Inverse &&inverse)
			: count_(n), identity_(std::move(identity)), op_(std::move(op)), inverse_(std::move(inverse)) {
			atl_assert(n <= std::size_t((std::numeric_limits<int32>::max)()));
			if constexpr (is_interleaved)
				nodes_.assign(n, node{int32(-1), identity_});
			else
				nodes_.parent_or_size.assign(n, int32(-1)), nodes_.weight.assign(n, identity_);
		}

	public:
		/** @brief An empty weighted DSU (0 elements). */
		weighted_dsu_by_compression_and_layout() = default;

		/**
		 * @brief `n` singletons with identity `Potential{}` (additive groups).
		 * @param n Number of elements.
		 * @pre `n <= 2³¹−1`
		 * @complexity O(n)
		 * @note Omitted operations are value-initialized and directly constructed into storage from their rvalues.
		 */
		constexpr explicit weighted_dsu_by_compression_and_layout(std::size_t n)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
				  && requires { Potential{}; Op{}; Inverse{}; }
			: weighted_dsu_by_compression_and_layout(initialize_t{}, n, Potential{}, Op{}, Inverse{}) {}

		/**
		 * @brief `n` singletons over the group `(Potential, op, identity, inverse)`.
		 * @param n Element count. @param identity Two-sided identity.
		 * @param op Group composition, directly constructed into storage from its rvalue. @param inverse Two-sided inversion.
		 * @pre `n <= 2³¹−1`
		 * @complexity O(n)
		 */
		constexpr weighted_dsu_by_compression_and_layout(std::size_t n, Potential identity, Op op, Inverse inverse)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
			: weighted_dsu_by_compression_and_layout(initialize_t{}, n, std::move(identity), std::move(op), std::move(inverse)) {}

		/**
		 * @brief `n` singletons with a supplied identity and value-initialized operations.
		 * @param n Number of elements.
		 * @param identity Two-sided group identity.
		 * @pre The size bound and group laws of the four-argument constructor apply.
		 * @complexity O(n)
		 */
		constexpr weighted_dsu_by_compression_and_layout(std::size_t n, Potential identity)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
				  && requires { Op{}; Inverse{}; }
			: weighted_dsu_by_compression_and_layout(initialize_t{}, n, std::move(identity), Op{}, Inverse{}) {}

		/**
		 * @brief `n` singletons with a supplied composition and value-initialized inverse.
		 * @param n Number of elements.
		 * @param identity Two-sided group identity.
		 * @param op Group composition, directly constructed into storage from its rvalue.
		 * @pre The size bound and group laws of the four-argument constructor apply.
		 * @complexity O(n)
		 */
		constexpr weighted_dsu_by_compression_and_layout(std::size_t n, Potential identity, Op op)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
				  && requires { Inverse{}; }
			: weighted_dsu_by_compression_and_layout(initialize_t{}, n, std::move(identity), std::move(op), Inverse{}) {}

		/** @brief Number of elements. */
		constexpr std::size_t elements() const noexcept { return storage_size(); }

		/** @brief Number of disjoint sets. */
		constexpr std::size_t count() const noexcept { return count_; }

		/**
		 * @brief The representative (root) of `x`'s set.
		 * @param x Element index.
		 * @pre `x < elements()` (asserted).
		 * @complexity O(α(n)) amortized (`halving`, `splitting`: the same bound)
		 */
		constexpr std::size_t find(std::size_t x) { return locate(x).first; }

		/** @brief Whether `a` and `b` belong to the same set.
		 * @param a First index. @param b Second index.
		 * @pre Both indices are less than elements() (asserted).
		 * @complexity O(α(n)) amortized. */
		[[nodiscard]] constexpr bool same(std::size_t a, std::size_t b) { return find(a) == find(b); }

		/** @brief The number of elements in the set containing `x`.
		 * @param x Element index. @pre `x < elements()` (asserted).
		 * @complexity O(α(n)) amortized. */
		[[nodiscard]] constexpr std::size_t size(std::size_t x) { return std::size_t(-link(find(x))); }

		/**
		 * @brief Record the constraint `pot(b) = pot(a) · w`; see @ref atl::weighted_dsu::unite.
		 * @param a Source index. @param b Destination index. @param w Relative potential.
		 * @pre Both indices are less than elements() (asserted).
		 * @return Merged, consistent or contradicts; compression may relink paths in every case.
		 * @complexity O(α(n)) amortized
		 */
		constexpr unite_outcome unite(std::size_t a, std::size_t b, Potential w) {
			auto [ra, pa] = locate(a);
			auto [rb, pb] = locate(b);
			Potential through = combine(pa, w);
			if (ra == rb) return through == pb ? unite_outcome::consistent : unite_outcome::contradicts;
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
		 * @brief `pot(a)⁻¹ · pot(b)`, or none across sets.
		 * @param a Source index. @param b Destination index.
		 * @pre Both indices are less than elements() (asserted).
		 * @return Relative potential or std::nullopt.
		 * @complexity O(α(n)) amortized
		 */
		[[nodiscard]] constexpr std::optional<Potential> difference(std::size_t a, std::size_t b) {
			auto [ra, pa] = locate(a);
			auto [rb, pb] = locate(b);
			if (ra != rb) return std::nullopt;
			return combine(Potential(std::invoke(inverse_, pa)), pb);
		}
	};

	/**
	 * @brief Interleaved rollback DSU (slower for wide potentials).
	 * Dense split links won the 2026-11-01 Xeon Platinum 8280 race
	 * (benchmarks/ds/weighted_dsu.cpp, ms, -O2 / native): int64 random
	 * 103 / 103 versus 84.1 / 86.2, binomial queries 249 / 254 versus
	 * 176 / 168, SL2 9.1 / 9.4 versus 8.5 / 8.6. Compact potentials retain
	 * this layout in the library class; parity split regressed 79.9 to 83.0
	 * under plain -O2. The tests use a constraint-forest oracle.
	 * @tparam Potential Group element type.
	 * @tparam Op Associative group composition.
	 * @tparam Inverse Two-sided inverse.
	 * @pre The group laws and value bounds of atl::rollback_weighted_dsu apply.
	 */
	template<std::copyable Potential = int64, binary_operation<Potential> Op = std::plus<>,
			 inverse_operation<Potential> Inverse = std::negate<>>
		requires std::equality_comparable<Potential>
	class rollback_weighted_dsu_by_interleaved_nodes {
		struct node {
			int32 parent_or_size; ///< >= 0: parent index; < 0: a root with size -value
			Potential weight;	  ///< pot(x) = pot(parent) · weight
		};

		std::vector<node> nodes_;
		std::vector<atl::detail::dsu_undo> history_;
		std::size_t count_ = 0;
		Potential identity_{};
		[[no_unique_address]] Op op_{};
		[[no_unique_address]] Inverse inverse_{};

		/** @brief `op(a, b)` converted to `Potential` (the concept allows any convertible result). */
		constexpr Potential combine(Potential const &a, Potential const &b) const { return Potential(std::invoke(op_, a, b)); }

		/** @brief The root of `x` and `pot(x)` relative to that root. */
		constexpr std::pair<std::size_t, Potential> locate(std::size_t x) const {
			atl_assert(x < nodes_.size());
			if (nodes_[x].parent_or_size < 0) return {x, identity_};
			Potential total = nodes_[x].weight;
			x = std::size_t(nodes_[x].parent_or_size);
			while (nodes_[x].parent_or_size >= 0) {
				total = combine(nodes_[x].weight, total);
				x = std::size_t(nodes_[x].parent_or_size);
			}
			return {x, std::move(total)};
		}

		// Bind constructor arguments directly to avoid an extra operation or identity transfer.
		struct initialize_t {};

		constexpr rollback_weighted_dsu_by_interleaved_nodes(initialize_t, std::size_t n, Potential &&identity, Op &&op, Inverse &&inverse)
			: count_(n), identity_(std::move(identity)), op_(std::move(op)), inverse_(std::move(inverse)) {
			atl_assert(n <= std::size_t((std::numeric_limits<int32>::max)()));
			nodes_.assign(n, node{int32(-1), identity_});
			history_.reserve(n); // a spanning forest merges at most n-1 times
		}

	public:
		/** @brief An empty weighted DSU (0 elements). */
		rollback_weighted_dsu_by_interleaved_nodes() = default;

		/**
		 * @brief `n` singletons with identity `Potential{}` (right for additive groups;
		 *        a multiplicative group passes its identity to the other constructor).
		 * @param n Number of elements.
		 * @pre `n <= 2³¹−1`
		 * @complexity O(n)
		 * @note Omitted operations are value-initialized and directly constructed into storage from their rvalues.
		 */
		constexpr explicit rollback_weighted_dsu_by_interleaved_nodes(std::size_t n)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
				  && requires { Potential{}; Op{}; Inverse{}; }
			: rollback_weighted_dsu_by_interleaved_nodes(initialize_t{}, n, Potential{}, Op{}, Inverse{}) {}

		/**
		 * @brief `n` singletons over the group `(Potential, op, identity, inverse)`.
		 * @param n Element count. @param identity Two-sided identity.
		 * @param op Group composition, directly constructed into storage from its rvalue. @param inverse Two-sided inversion.
		 * @pre `n <= 2³¹−1` (the int32 element storage, which also bounds the uint32 undo log)
		 * @complexity O(n)
		 */
		constexpr rollback_weighted_dsu_by_interleaved_nodes(std::size_t n, Potential identity, Op op, Inverse inverse)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
			: rollback_weighted_dsu_by_interleaved_nodes(initialize_t{}, n, std::move(identity), std::move(op), std::move(inverse)) {}

		/**
		 * @brief `n` singletons with a supplied identity and value-initialized operations.
		 * @param n Number of elements.
		 * @param identity Two-sided group identity.
		 * @pre The size bound and group laws of the four-argument constructor apply.
		 * @complexity O(n)
		 */
		constexpr rollback_weighted_dsu_by_interleaved_nodes(std::size_t n, Potential identity)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
				  && requires { Op{}; Inverse{}; }
			: rollback_weighted_dsu_by_interleaved_nodes(initialize_t{}, n, std::move(identity), Op{}, Inverse{}) {}

		/**
		 * @brief `n` singletons with a supplied composition and value-initialized inverse.
		 * @param n Number of elements.
		 * @param identity Two-sided group identity.
		 * @param op Group composition, directly constructed into storage from its rvalue.
		 * @pre The size bound and group laws of the four-argument constructor apply.
		 * @complexity O(n)
		 */
		constexpr rollback_weighted_dsu_by_interleaved_nodes(std::size_t n, Potential identity, Op op)
			requires std::constructible_from<Op, Op &&> && std::constructible_from<Inverse, Inverse &&>
				  && requires { Inverse{}; }
			: rollback_weighted_dsu_by_interleaved_nodes(initialize_t{}, n, std::move(identity), std::move(op), Inverse{}) {}

		/** @brief Number of elements. */
		constexpr std::size_t elements() const noexcept { return nodes_.size(); }

		/** @brief Number of disjoint sets. */
		constexpr std::size_t count() const noexcept { return count_; }

		/**
		 * @brief The representative (root) of `x`'s set.
		 * @param x Element index.
		 * @pre `x < elements()` (asserted).
		 * @complexity O(log n)
		 */
		[[nodiscard]] constexpr std::size_t find(std::size_t x) const {
			atl_assert(x < nodes_.size());
			while (nodes_[x].parent_or_size >= 0) x = std::size_t(nodes_[x].parent_or_size);
			return x;
		}

		/**
		 * @brief Whether `a` and `b` belong to the same set.
		 * @complexity O(log n)
		 */
		[[nodiscard]] constexpr bool same(std::size_t a, std::size_t b) const { return find(a) == find(b); }

		/**
		 * @brief The number of elements in the set containing `x`.
		 * @complexity O(log n)
		 */
		[[nodiscard]] constexpr std::size_t size(std::size_t x) const { return std::size_t(-nodes_[find(x)].parent_or_size); }

		/**
		 * @brief Record the constraint `pot(b) = pot(a) · w` (additively `x_b - x_a = w`).
		 * @param a Source index. @param b Destination index. @param w Relative potential.
		 * @pre Both indices are less than elements() (asserted).
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
			int32 sa = -nodes_[ra].parent_or_size, sb = -nodes_[rb].parent_or_size;
			if (sa < sb) {
				std::swap(ra, rb), std::swap(sa, sb);
				lift = Potential(std::invoke(inverse_, lift));
			}
			history_.push_back({uint32(rb), uint32(ra), uint32(sb)});
			nodes_[rb] = node{int32(ra), std::move(lift)};
			nodes_[ra].parent_or_size = -(sa + sb);
			--count_;
			return unite_outcome::merged;
		}

		/**
		 * @brief `pot(a)⁻¹ · pot(b)` (additively `x_b - x_a`), or none if `a` and `b` are
		 *        in different sets.
		 * @complexity O(log n)
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
		 * @pre `time() > 0`.
		 * @complexity O(1)
		 */
		constexpr void undo() {
			atl_assert(!history_.empty());
			auto const [child, root, child_size] = history_.back();
			history_.pop_back();
			nodes_[root].parent_or_size += int32(child_size);
			nodes_[child] = node{-int32(child_size), identity_};
			++count_;
		}

		/**
		 * @brief Undo merges until `time() == t`, where `t` came from an earlier
		 *        @ref time call.
		 * @param t Token from this object's still-live prefix.
		 * @pre `t <= time()` (asserted); provenance is documented-only.
		 * @complexity O(time() - t)
		 */
		constexpr void rollback(std::size_t t) {
			atl_assert(t <= history_.size());
			while (history_.size() > t) undo();
		}
	};
}
