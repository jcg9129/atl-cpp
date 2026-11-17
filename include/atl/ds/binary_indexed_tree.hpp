#pragma once
#include <atl/ds/detail/forward_element.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/traits/integral.hpp>
#include <atl/utils/assert.hpp>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	namespace detail {
		/**
		 * @brief Nesting depth of a range type, with the scalar reached at the bottom.
		 *
		 * `::value` is how many range layers wrap the scalar and `::scalar` is that
		 * innermost element type, letting the nested-range constructor and deduction
		 * guides map a container onto a tree dimension:
		 * `range_rank<vector<vector<int>>>::value == 2`, `::scalar == int`.
		 *
		 * This primary template is the base case: a non-range type has rank 0 and is
		 * its own scalar.
		 */
		template<typename R>
		struct range_rank : std::integral_constant<std::size_t, 0> {
			using scalar = R;
		};

		template<std::ranges::range R>
		struct range_rank<R>
			: std::integral_constant<std::size_t, 1 + range_rank<std::ranges::range_value_t<R>>::value> {
			using scalar = typename range_rank<std::ranges::range_value_t<R>>::scalar;
		};

		/** @brief The nesting depth of `R` (cvref-stripped); see `range_rank`. */
		template<typename R>
		inline constexpr std::size_t range_rank_v = range_rank<std::remove_cvref_t<R>>::value;

		/** @brief The scalar element type at the bottom of `R` (cvref-stripped); see `range_rank`. */
		template<typename R>
		using range_scalar_t = typename range_rank<std::remove_cvref_t<R>>::scalar;

		/** @brief Shape discovery needs const size at every layer and const iteration only above the scalar layer. */
		template<typename R, std::size_t Depth>
		inline constexpr bool is_sized_nested_range = [] {
			if constexpr (Depth == 0)
				return true;
			else if constexpr (std::ranges::sized_range<R> && requires(R const &r) { std::ranges::size(r); }) {
				if constexpr (Depth == 1)
					return true;
				else if constexpr (std::ranges::range<R const>)
					return is_sized_nested_range<std::remove_cvref_t<std::ranges::range_reference_t<R const>>, Depth - 1>;
				else
					return false;
			} else
				return false;
		}();

		/**
		 * @brief A factory's extent argument as `std::size_t`.
		 * @pre `size >= 0` and representable in size_t (asserted) — a negative extent would otherwise wrap to a huge one.
		 */
		template<integral Size>
		constexpr std::size_t checked_extent(Size size) {
			if constexpr (traits::is_signed_integral_v<Size>) atl_assert(size >= 0);
			if constexpr (std::numeric_limits<Size>::digits > std::numeric_limits<std::size_t>::digits)
				atl_assert(size <= Size((std::numeric_limits<std::size_t>::max)()));
			return std::size_t(size);
		}
	}

	/**
	 * @brief Binary Indexed Tree (Fenwick tree), in one or more dimensions.
	 *
	 * Represents a `Dim`-dimensional array of `T` and supports combining a value
	 * into a cell and querying the aggregate over a prefix box
	 * `[0, i0) x [0, i1) x ...`, using a binary operation `op`. All indices are
	 * 0-based. The values live in a single flat vector (row-major).
	 *
	 * @tparam T   The element type.
	 * @tparam Dim The number of dimensions (default 1).
	 * @tparam Op  The binary operation combining two elements (default std::plus).
	 *
	 * The operation and identity are supplied at construction (`op` defaults to a
	 * value-initialized `Op`, `identity` to `T{}`), which lets the same class
	 * serve sums, products, min/max, xor, … — pass the identity that matches your
	 * operation (e.g. `T{1}` for a product, `+inf` for min).
	 *
	 * @pre (T, op, identity) forms a commutative monoid: `op` is associative and
	 *      commutative and `identity` is its identity element. Commutativity is
	 *      required because each cell aggregates a range that extends past the
	 *      updated index and prefix folds combine cells out of order (string
	 *      concatenation, for example, gives wrong results). This is not (and
	 *      cannot be) enforced. Only prefix queries are provided; a box range
	 *      query needs an abelian group (an invertible `op`) and is left to the
	 *      caller (inclusion-exclusion).
	 *
	 * @pre All operations and conversions are valid and representable. The product
	 *      of extents and every row-major stride fit size_t (asserted).
	 *      An update index is less than its extent; a prefix endpoint is at most
	 *      its extent. Index conversion to size_t must be exact (asserted).
	 * @complexity Storage O(product of extents); build O(Dim * product of extents).
	 *
	 * [Fenwick tree](https://en.wikipedia.org/wiki/Fenwick_tree)
	 * [Fenwick tree algorithms](https://cp-algorithms.com/data_structures/fenwick.html)
	 */
	template<std::copyable T, std::size_t Dim = 1, binary_operation<T> Op = std::plus<>>
	class binary_indexed_tree {
		std::array<std::size_t, Dim> sizes_;
		std::array<std::size_t, Dim> strides_{};
		std::vector<T> data_;
		[[no_unique_address]] Op op_;
		T identity_;

		constexpr std::size_t init_strides() {
			std::size_t total = 1;
			for (std::size_t d = Dim; d-- > 0;) {
				strides_[d] = total;
				atl_assert(sizes_[d] == 0 || total <= (std::numeric_limits<std::size_t>::max)() / sizes_[d]);
				total *= sizes_[d];
			}
			return total;
		}

		// Turn raw cell values in data_ into a Fenwick tree in O(Dim * total): along each
		// axis, fold every cell into its Fenwick parent (indices processed low-to-high
		// so a parent has collected all its children before it is folded in turn).
		constexpr void build_tree_layout() {
			std::size_t const total = data_.size();
			if (total == 0) return;
			// odometer over (block, index along axis d, inner offset): the axis index is
			// a loop counter, not a division and modulo per cell
			for (std::size_t d = 0; d < Dim; ++d) {
				std::size_t const stride = strides_[d], extent = sizes_[d], span = extent * stride;
				for (std::size_t block = 0; block < total; block += span)
					for (std::size_t index = 0; index < extent; ++index) {
						std::size_t const parent = index | (index + 1);
						if (parent >= extent) continue;
						std::size_t const from = block + index * stride, to = block + parent * stride;
						for (std::size_t j = 0; j < stride; ++j) data_[to + j] = op_(data_[to + j], data_[from + j]);
					}
			}
		}

		/** @brief The first-element chain defines extents; flattening checks rectangularity against them. */
		template<std::size_t D, std::ranges::sized_range R>
			requires(D < Dim) && detail::is_sized_nested_range<R, Dim - D>
		constexpr void fill_shape(R const &r) {
			sizes_[D] = detail::checked_extent(std::ranges::size(r));
			if constexpr (D + 1 < Dim) {
				if (sizes_[D] == 0)
					for (std::size_t d = D + 1; d < Dim; ++d) sizes_[d] = 0;
				else
					fill_shape<D + 1>(*std::ranges::begin(r));
			}
		}

		/** @brief Preserve source elements unless the whole chain owns them or the leaf yields a prvalue. */
		template<bool ShouldMove, std::size_t D, std::ranges::sized_range R>
			requires(D < Dim)
		constexpr void flatten_into(R &&r) {
			atl_assert(std::ranges::size(r) == sizes_[D]);
			if constexpr (D + 1 == Dim)
				for (auto &&e : r) {
					if constexpr (ShouldMove || !std::is_reference_v<std::ranges::range_reference_t<R>>)
						data_.emplace_back(std::move(e));
					else
						data_.emplace_back(std::as_const(e));
				}
			else
				for (auto &&e : r)
					flatten_into<(ShouldMove || !std::is_reference_v<std::ranges::range_reference_t<R>>) && detail::moves_range_elements<std::remove_cvref_t<decltype(e)>>, D + 1>(e);
		}

		template<std::size_t D>
			requires(D <= Dim)
		constexpr void update_impl(T const &value, std::array<std::size_t, Dim> const &idx, std::size_t flat) {
			if constexpr (D == Dim)
				data_[flat] = op_(data_[flat], value);
			else {
				atl_assert(idx[D] < sizes_[D]);
				for (std::size_t j = idx[D]; j < sizes_[D]; j = j | (j + 1))
					update_impl<D + 1>(value, idx, flat + j * strides_[D]);
			}
		}

		template<std::size_t D>
			requires(D <= Dim)
		constexpr void prefix_fold_impl(std::array<std::size_t, Dim> const &idx, std::size_t flat, T &result) const {
			if constexpr (D == Dim)
				result = op_(result, data_[flat]);
			else {
				atl_assert(idx[D] <= sizes_[D]);
				for (std::size_t i = idx[D]; i > 0; i = i & (i - 1))
					prefix_fold_impl<D + 1>(idx, flat + (i - 1) * strides_[D], result);
			}
		}

	public:
		/**
		 * @brief Build a tree over a `Dim`-dimensional box of the given sizes.
		 * @param sizes Extent of each dimension.
		 * @param identity Neutral element used to fill the box.
		 * @param op Commutative monoid operation.
		 * @complexity O(product of extents) time and storage.
		 */
		constexpr explicit binary_indexed_tree(std::array<std::size_t, Dim> sizes, T identity = T{}, Op op = Op{})
			: sizes_(sizes), op_(std::move(op)), identity_(std::move(identity)) {
			data_.assign(init_strides(), identity_);
		}

		/**
		 * @brief Convenience 1-D constructor, so `binary_indexed_tree<T>(n)` works.
		 */
		constexpr explicit binary_indexed_tree(std::size_t n, T identity = T{}, Op op = Op{})
			requires(Dim == 1)
			: binary_indexed_tree(std::array<std::size_t, 1>{n}, std::move(identity), std::move(op)) {}

		/**
		 * @brief Build from a nested range of initial cell values, e.g. a
		 *        `vector<vector<T>>` for `Dim == 2`. The dimension must match the
		 *        range's nesting depth; the extents are read from the range (which
		 *        must be rectangular). @complexity O(Dim * total).
		 */
		template<std::ranges::sized_range Range>
			requires(detail::range_rank_v<Range> == Dim
					 && detail::is_sized_nested_range<std::remove_reference_t<Range>, Dim>
					 && std::convertible_to<detail::range_scalar_t<Range>, T>
					 && !std::same_as<std::remove_cvref_t<Range>, std::array<std::size_t, Dim>>)
		constexpr explicit binary_indexed_tree(Range &&values, T identity = T{}, Op op = Op{})
			: op_(std::move(op)), identity_(std::move(identity)) {
			fill_shape<0>(values);
			std::size_t total = init_strides();
			data_.reserve(total);
			flatten_into<detail::moves_range_elements<Range>, 0>(values);
			atl_assert(data_.size() == total);
			build_tree_layout();
		}

		/**
		 * @brief Build from initial cell values already laid out flat and row-major
		 *        (size == product of `sizes`), in O(Dim * total). Prefer the nested-range
		 *        constructor unless the data is already flat (e.g. mdspan-backed).
		 */
		template<std::ranges::sized_range Range>
			requires std::constructible_from<T, detail::forwarded_element_t<Range>>
		constexpr binary_indexed_tree(std::array<std::size_t, Dim> sizes, Range &&values, T identity = T{}, Op op = Op{})
			: sizes_(sizes), op_(std::move(op)), identity_(std::move(identity)) {
			std::size_t total = init_strides();
			atl_assert(std::ranges::size(values) == total);
			data_.reserve(total);
			for (auto &&e : values) data_.emplace_back(detail::forward_element<Range>(e));
			build_tree_layout();
		}

		/**
		 * @brief Combine `value` into the cell at `(indices...)`.
		 * @param value Element to combine using op.
		 * @param indices Zero-based coordinate in each dimension.
		 * @pre Each index is nonnegative, fits size_t, and is below its extent.
		 * @complexity O(log^Dim n)
		 */
		template<integral... Indices>
			requires(sizeof...(Indices) == Dim)
		constexpr void update(T const &value, Indices... indices) {
			std::array<std::size_t, Dim> const idx{detail::checked_extent(indices)...};
			for (std::size_t d = 0; d < Dim; ++d) atl_assert(idx[d] < sizes_[d]);
			update_impl<0>(value, idx, 0);
		}

		/**
		 * @brief Aggregate over the prefix box `[0, i0) x [0, i1) x ...`.
		 * @param indices Half-open prefix endpoint in each dimension.
		 * @pre Each endpoint is nonnegative, fits size_t, and is at most its extent.
		 * @return The combined value, or `identity` when the box is empty.
		 * @complexity O(log^Dim n)
		 */
		template<integral... Indices>
			requires(sizeof...(Indices) == Dim)
		constexpr T prefix_fold(Indices... indices) const {
			T result = identity_;
			std::array<std::size_t, Dim> const idx{detail::checked_extent(indices)...};
			for (std::size_t d = 0; d < Dim; ++d) atl_assert(idx[d] <= sizes_[d]);
			prefix_fold_impl<0>(idx, 0, result);
			return result;
		}

		/**
		 * @brief Largest `r` in `[0, size()]` with `pred(prefix_fold(r))` true — the
		 *        k-th / lower-bound query on prefix folds, as one top-down
		 *        power-of-two descent instead of a binary search over
		 *        `prefix_fold`.
		 * @pre `pred` is monotone (true on a prefix of `r`, then false) and
		 *      `pred(identity)` holds.
		 * @complexity O(log n) calls of `op` and `pred`
		 */
		template<std::predicate<T const &> Pred>
			requires(Dim == 1)
		constexpr std::size_t max_right(Pred pred) const {
			atl_assert(pred(identity_));
			std::size_t const n = sizes_[0];
			std::size_t r = 0;
			T acc = identity_;
			// data_[r + step - 1] folds exactly [r, r + step) whenever step is a
			// power of two no larger than the lowest set bit of r.
			for (std::size_t step = std::bit_floor(n); step > 0; step >>= 1) {
				if (r + step > n) continue;
				if (T t = op_(acc, data_[r + step - 1]); pred(t)) {
					acc = std::move(t);
					r += step;
				}
			}
			return r;
		}

		/** @brief The number of cells (1-D). */
		constexpr std::size_t size() const noexcept
			requires(Dim == 1)
		{
			return sizes_[0];
		}

		/** @brief The extent of each axis. */
		constexpr std::array<std::size_t, Dim> const &sizes() const noexcept
			requires(Dim > 1)
		{
			return sizes_;
		}
	};

	/** @brief Deduce stored element, operation and dimensions from construction arguments. */
	template<std::copyable T, binary_operation<T> Op>
	binary_indexed_tree(std::size_t, T, Op) -> binary_indexed_tree<T, 1, Op>;
	/** @brief Deduce stored element, operation and dimensions from construction arguments. */
	template<std::copyable T>
	binary_indexed_tree(std::size_t, T) -> binary_indexed_tree<T, 1>;

	/** @brief Deduce stored element, operation and dimensions from construction arguments. */
	template<std::size_t Dim, std::copyable T, binary_operation<T> Op>
	binary_indexed_tree(std::array<std::size_t, Dim>, T, Op) -> binary_indexed_tree<T, Dim, Op>;
	/** @brief Deduce stored element, operation and dimensions from construction arguments. */
	template<std::size_t Dim, std::copyable T>
	binary_indexed_tree(std::array<std::size_t, Dim>, T) -> binary_indexed_tree<T, Dim>;

	/** @brief Deduce dimensions and scalar type from nested range layers. */
	template<std::ranges::sized_range Range>
		requires(!std::same_as<std::remove_cvref_t<Range>, std::array<std::size_t, detail::range_rank_v<Range>>>)
	binary_indexed_tree(Range &&) -> binary_indexed_tree<detail::range_scalar_t<Range>, detail::range_rank_v<Range>>;

	/**
	 * @brief Build a binary_indexed_tree without spelling `Dim` in the type: the
	 *        dimension is deduced from the number of sizes, `T` from the identity.
	 *
	 *     auto grid = make_binary_indexed_tree(0LL, 100, 200);            // 2D sum
	 *     auto prod = make_binary_indexed_tree<std::multiplies<>>(1, n);  // 1D product
	 */
	template<typename Op = std::plus<>, std::copyable T, integral... Sizes>
		requires binary_operation<Op, T>
	constexpr auto make_binary_indexed_tree(T identity, Sizes... sizes) {
		return binary_indexed_tree<T, sizeof...(Sizes), Op>(
			std::array<std::size_t, sizeof...(Sizes)>{detail::checked_extent(sizes)...}, std::move(identity));
	}

	/**
	 * @brief Build a binary_indexed_tree from initial values. With no sizes given,
	 *        the range is taken as nested and both `T` and `Dim` are deduced from it
	 *        (`vector<vector<int>>` -> 2D). With sizes given, the range is taken as
	 *        flat and row-major and `Dim` is the number of sizes.
	 *
	 *     auto a = make_binary_indexed_tree_from(values);         // 1D from a range
	 *     auto g = make_binary_indexed_tree_from(grid);           // 2D from vector<vector>
	 *     auto f = make_binary_indexed_tree_from(flat, 100, 200); // 2D from a flat range
	 *
	 * This overload selects std::plus<> with identity T{}; the caller must ensure
	 * that T{} is its identity. Any other `Op` goes through the overload
	 * that takes the identity first.
	 */
	template<typename Op = std::plus<>, std::ranges::sized_range Range, integral... Sizes>
		requires std::same_as<Op, std::plus<>> && binary_operation<Op, detail::range_scalar_t<Range>>
	constexpr auto make_binary_indexed_tree_from(Range &&values, Sizes... sizes) {
		if constexpr (sizeof...(Sizes) == 0)
			return binary_indexed_tree<detail::range_scalar_t<Range>, detail::range_rank_v<Range>, Op>(
				std::forward<Range>(values));
		else
			return binary_indexed_tree<std::ranges::range_value_t<Range>, sizeof...(Sizes), Op>(
				std::array<std::size_t, sizeof...(Sizes)>{detail::checked_extent(sizes)...}, std::forward<Range>(values));
	}

	/**
	 * @brief Build from values with an explicit identity for an arbitrary operation: the
	 *        identity comes first, as in `make_binary_indexed_tree`, then the
	 *        values and the optional sizes.
	 *
	 *     auto prod = make_binary_indexed_tree_from<std::multiplies<>>(1LL, values);
	 */
	template<typename Op = std::plus<>, typename Identity, std::ranges::sized_range Range, integral... Sizes>
		requires(!std::ranges::range<Identity>) && binary_operation<Op, detail::range_scalar_t<Range>>
			 && std::convertible_to<Identity, detail::range_scalar_t<Range>>
	constexpr auto make_binary_indexed_tree_from(Identity identity, Range &&values, Sizes... sizes) {
		using T = detail::range_scalar_t<Range>;
		if constexpr (sizeof...(Sizes) == 0)
			return binary_indexed_tree<T, detail::range_rank_v<Range>, Op>(std::forward<Range>(values),
																		   T(std::move(identity)));
		else
			return binary_indexed_tree<T, sizeof...(Sizes), Op>(
				std::array<std::size_t, sizeof...(Sizes)>{detail::checked_extent(sizes)...}, std::forward<Range>(values),
				T(std::move(identity)));
	}
}
