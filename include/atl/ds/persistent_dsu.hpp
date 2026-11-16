#pragma once
#include <atl/ds/persistent_array.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/traits/integral.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace atl {
	namespace detail {
		/** @brief The signed parent-or-size element a `persistent_dsu` backing yields from `b[i]`. */
		template<typename Backing>
		using persistent_dsu_value_t = std::remove_cvref_t<decltype(std::declval<Backing const &>()[std::size_t()])>;

		/**
		 * @brief A persistent array of a signed integer usable as `persistent_dsu` storage:
		 *        indexable, sized, copyable (a version is returned by value),
		 *        default-constructible (the empty DSU), constructible as
		 *        `Backing(n, value)`, and a chained pair of rvalue `set` calls implicitly convertible to a new version.
		 */
		template<typename Backing>
		concept persistent_signed_array =
			std::copyable<Backing> && std::default_initializable<Backing> && requires(Backing const &b, Backing &mutable_backing, std::size_t i, std::size_t j) {
				requires traits::is_signed_integral_v<persistent_dsu_value_t<Backing>>;
				{ b.size() } -> std::convertible_to<std::size_t>;
				{ std::move(mutable_backing).set(i, persistent_dsu_value_t<Backing>()).set(j, persistent_dsu_value_t<Backing>()) } -> std::convertible_to<Backing>;
			} && std::constructible_from<Backing, std::size_t, persistent_dsu_value_t<Backing>>;
	}

	/**
	 * @brief Persistent disjoint-set union (union-find) over a persistent array.
	 *
	 * Each `unite` returns a **new** version; every past version stays queryable
	 * (`find`/`same`/`size`), with full persistence you may branch a new DSU off any version — the
	 * classic persistent-array application. Uses **union by size** and **no path compression** (compression
	 * mutates on reads, which an immutable array cannot do), so `find` is
	 * O(log² n): a height-O(log n) walk, each lookup O(log n) in the backing array.
	 *
	 * A single backing array holds both fields: a non-negative value is
	 * the parent index; a negative value marks a root and encodes its set size as
	 * `-value`. The backing element type must therefore be **signed** and large
	 * enough to index `n` elements (default `int32` → n ≤ 2³¹−1; use
	 * `persistent_array<int64>` for more).
	 *
	 * Persistence is inherited from `Backing`: `persistent_array<int32>` (default)
	 * gives full persistence (branch off any version); `semi_persistent_array<int32>`
	 * gives partial persistence (`unite` only the newest version).
	 *
	 *     persistent_dsu<> dsu(n); // full persistence
	 *     auto a = dsu.unite(0, 1); // a new version; dsu is unchanged
	 *     bool c = a.same(0, 1); // true
	 *
	 * @note Not constexpr (the supplied backing arrays use shared runtime allocation).
	 * @note With the supplied arrays, version copies are O(1); updates share an
	 *       append-only pool. Dropped versions do not reclaim individual nodes.
	 *       Nodes are freed when the last handle in the family dies; concurrent
	 *       updates to a shared pool require external synchronization.
	 * @tparam Backing The persistent array of a signed integer holding parent_or_size.
	 * @pre Backing provides the documented persistence and value semantics; the
	 *      concept checks expressions, not immutability or complexity.
	 */
	template<detail::persistent_signed_array Backing = persistent_array<int32>>
	class persistent_dsu {
		using value_type = detail::persistent_dsu_value_t<Backing>;

		// parent_or_size_[x] >= 0: parent index; < 0: x is a root with size -value.
		Backing parent_or_size_;
		std::size_t count_ = 0;

		/**
		 * @brief Wraps an already-built version; `count` is the number of disjoint sets
		 *        in `data`, carried along because it cannot be recomputed in O(1).
		 */
		persistent_dsu(Backing data, std::size_t count) : parent_or_size_(std::move(data)), count_(count) {}

		persistent_dsu unite_in(Backing data, std::size_t a, std::size_t b) const {
			atl_assert(a < data.size() && b < data.size());
			auto root_of = [&data](std::size_t x) {
				while (data[x] >= 0) x = std::size_t(data[x]);
				return x;
			};
			std::size_t ra = root_of(a), rb = root_of(b);
			if (ra == rb) return {std::move(data), count_};
			value_type sa = -data[ra], sb = -data[rb];
			if (sa < sb) std::swap(ra, rb), std::swap(sa, sb);

			return {std::move(data).set(rb, value_type(ra)).set(ra, value_type(-(sa + sb))), count_ - 1};
		}

		/** @brief Returns `n` after asserting it fits `value_type`, so the check runs before the O(n) build. */
		static std::size_t checked_elements(std::size_t n) {
			if constexpr (std::numeric_limits<value_type>::digits < std::numeric_limits<std::size_t>::digits)
				atl_assert(n <= std::size_t((std::numeric_limits<value_type>::max)()));
			return n;
		}

	public:
		/** @brief An empty DSU (0 elements).
		 * @note Not constexpr (the backing uses shared runtime storage).
		 */
		persistent_dsu() = default;

		/**
		 * @brief A DSU of `n` singleton sets {0}, {1}, …, {n-1}.
		 * @param n Number of elements.
		 * @pre `n <= numeric_limits<value_type>::max()` (asserted).
		 * @complexity O(n)
		 * @note Not constexpr (the backing uses shared runtime storage).
		 */
		explicit persistent_dsu(std::size_t n) : parent_or_size_(checked_elements(n), value_type(-1)), count_(n) {}

		/** @brief Number of elements.

		 * @note Not constexpr (the backing uses shared runtime storage).
		 */
		std::size_t elements() const noexcept { return parent_or_size_.size(); }

		/** @brief Number of disjoint sets.

		 * @note Not constexpr (the backing uses shared runtime storage).
		 */
		std::size_t count() const noexcept { return count_; }

		/**
		 * @brief The representative (root) of `x`'s set.
		 * @param x Element index.
		 * @pre `x < elements()` (asserted).
		 * @complexity O(log² n)
		 * @note Not constexpr (the backing uses shared runtime storage).
		 */
		[[nodiscard]] std::size_t find(std::size_t x) const {
			atl_assert(x < parent_or_size_.size());
			while (parent_or_size_[x] >= 0) x = std::size_t(parent_or_size_[x]);
			return x;
		}

		/**
		 * @brief Whether `a` and `b` belong to the same set.
		 * @param a First element index.
		 * @param b Second element index.
		 * @pre `a < elements()` and `b < elements()` (asserted).
		 * @complexity O(log² n)
		 * @note Not constexpr (the backing uses shared runtime storage).
		 */
		[[nodiscard]] bool same(std::size_t a, std::size_t b) const { return find(a) == find(b); }

		/**
		 * @brief The number of elements in the set containing `x`.
		 * @param x Element index.
		 * @pre `x < elements()` (asserted).
		 * @complexity O(log² n)
		 * @note Not constexpr (the backing uses shared runtime storage).
		 */
		[[nodiscard]] std::size_t size(std::size_t x) const { return std::size_t(-parent_or_size_[find(x)]); }

		/**
		 * @brief A new DSU with the sets of `a` and `b` merged (union by size). If
		 *        they are already together, the same version is returned unchanged.
		 * @param a First element index.
		 * @param b Second element index.
		 * @pre Both indices are in range (asserted). With partial persistence, only
		 *      the newest backing version may be updated.
		 * @return New version, sharing backing storage with this family.
		 * @complexity O(log² n) with the supplied arrays.
		 * @note Not constexpr (the backing uses shared runtime storage).
		 */
		[[nodiscard]] persistent_dsu unite(std::size_t a, std::size_t b) const & { return unite_in(parent_or_size_, a, b); }

		/**
		 * @brief `unite` consuming this version: the backing is moved, not copied, so
		 *        `d = std::move(d).unite(a, b)` is the cheap linear idiom. The moved-from
		 *        DSU may only be destroyed or assigned to.
		 * @param a First element index.
		 * @param b Second element index.
		 * @pre Both indices are in range (asserted). With partial persistence, only
		 *      the newest backing version may be updated.
		 * @return New version, sharing backing storage with this family.
		 * @complexity O(log² n) with the supplied arrays.
		 * @note Not constexpr (the backing uses shared runtime storage).
		 */
		[[nodiscard]] persistent_dsu unite(std::size_t a, std::size_t b) && {
			return unite_in(std::move(parent_or_size_), a, b);
		}
	};
}
