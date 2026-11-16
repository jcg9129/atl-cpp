#pragma once

#include <atl/ds/detail/forward_element.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <compare>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <ranges>
#include <type_traits>
#include <utility>

namespace atl {
	/**
	 * @brief Fixed-capacity vector with inline storage; the container itself never allocates.
	 *
	 * Stores up to Capacity elements of T in place; push_back beyond Capacity is
	 * a precondition violation. It is value-semantic, so it supports a
	 * by-value return where the number of elements is small and bounded (e.g. a
	 * factorization's few prime factors), avoiding a heap allocation per call.
	 *
	 * The inline storage is value-initialized up front and push_back assigns into
	 * it (rather than constructing on demand as std::vector does), so T must be
	 * default-constructible and copyable — enforced via std::semiregular. This
	 * suits small, trivial element types, for which constructing, copying, and
	 * destroying the whole capacity are cheap.
	 *
	 * @tparam T        Element type; std::semiregular (default-constructible, copyable).
	 * @tparam Capacity Maximum number of elements.
	 */
	template<std::semiregular T, std::size_t Capacity>
	class static_vector {
		std::array<T, Capacity> data_{};
		std::size_t size_ = 0;

		/**
		 * @brief Return slot @p index (no longer live) to the value-initialized state, so a
		 * nontrivially destructible element can release resources through assignment;
		 * trivially destructible types retain the slot's old value.
		 */
		constexpr void release(std::size_t index) {
			if constexpr (!std::is_trivially_destructible_v<T>) data_[index] = T{};
		}

	public:
		/** @brief Element type. */
		using value_type = T;
		/** @brief Element count type. */
		using size_type = std::size_t;
		/** @brief Iterator distance type. */
		using difference_type = std::ptrdiff_t;
		/** @brief Mutable element reference. */
		using reference = T &;
		/** @brief Read-only element reference. */
		using const_reference = T const &;
		/** @brief Mutable element pointer. */
		using pointer = T *;
		/** @brief Read-only element pointer. */
		using const_pointer = T const *;
		/** @brief Contiguous iterator over live elements. */
		using iterator = T *;
		/** @brief Read-only contiguous iterator over live elements. */
		using const_iterator = T const *;

		/** @brief An empty vector. */
		constexpr static_vector() = default;

		/**
		 * @brief A vector holding the elements of @p init, in order.
		 * @pre `init.size() <= Capacity`
		 */
		constexpr static_vector(std::initializer_list<T> init) {
			for (T const &value : init) push_back(value);
		}

		/**
		 * @brief A vector holding the elements of @p values, in order. Elements are
		 *        moved out only of an owning rvalue range (see
		 *        `detail::forward_element`), copied otherwise.
		 * @pre the range has at most `Capacity` elements
		 * @complexity O(Capacity)
		 */
		template<std::ranges::input_range R>
			requires(!std::same_as<std::remove_cvref_t<R>, static_vector>
					 && std::convertible_to<detail::forwarded_element_t<R>, T>)
		constexpr explicit static_vector(R &&values) {
			for (auto &&value : values) push_back(static_cast<T>(detail::forward_element<R>(std::forward<decltype(value)>(value))));
		}

		/**
		 * @brief Append a copy of @p value.
		 * @pre `size() < Capacity`
		 */
		constexpr void push_back(T const &value) {
			atl_assert(size_ < Capacity);
			data_[size_++] = value;
		}

		/**
		 * @brief Append @p value, moved in.
		 * @pre `size() < Capacity`
		 */
		constexpr void push_back(T &&value) {
			atl_assert(size_ < Capacity);
			data_[size_++] = std::move(value);
		}

		/**
		 * @brief Append an element built from `args`: constructed once as a
		 *        temporary, then move-assigned into its (already-constructed) slot.
		 * @pre `size() < Capacity`
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		constexpr void emplace_back(Args &&...args) {
			atl_assert(size_ < Capacity);
			data_[size_++] = T(std::forward<Args>(args)...);
		}

		/**
		 * @brief Remove the last element; nontrivially destructible T is reset to T{}.
		 * Resource release follows T's assignment semantics. @pre `!empty()`
		 */
		constexpr void pop_back() {
			atl_assert(size_ > 0);
			release(--size_);
		}

		/** @brief Remove every element. @complexity O(size()) */
		constexpr void clear() {
			while (size_ > 0) release(--size_);
		}

		/**
		 * @brief Grow to @p count elements with value-initialized ones appended, or
		 *        shrink to the first @p count. @pre `count <= Capacity`
		 */
		constexpr void resize(std::size_t count) { resize(count, T{}); }

		/**
		 * @brief Grow to @p count elements with copies of @p value appended, or
		 *        shrink to the first @p count. @pre `count <= Capacity`
		 */
		constexpr void resize(std::size_t count, T const &value) {
			atl_assert(count <= Capacity);
			if (count < size_) {
				// Release back to front through pointers: GCC -O3 cannot prove size_ <= Capacity
				// and would flag std::array::operator[] here (a known -Warray-bounds false positive).
				if constexpr (!std::is_trivially_destructible_v<T>)
					for (T *slot = data_.data() + size_; slot != data_.data() + count;) *--slot = T{};
				size_ = count;
			} else {
				while (size_ < count) data_[size_++] = value;
			}
		}

		/** @brief Number of live elements. */
		constexpr std::size_t size() const noexcept { return size_; }

		/** @brief Whether there are no live elements. */
		constexpr bool empty() const noexcept { return size_ == 0; }

		/** @brief Maximum number of live elements. */
		static constexpr std::size_t capacity() noexcept { return Capacity; }

		/** @brief The first element. @pre `!empty()` */
		constexpr T &front() {
			atl_assert(size_ > 0);
			return data_[0];
		}

		/** @brief The first element. @pre `!empty()` */
		constexpr T const &front() const {
			atl_assert(size_ > 0);
			return data_[0];
		}

		/** @brief The last element. @pre `!empty()` */
		constexpr T &back() {
			atl_assert(size_ > 0);
			return data_[size_ - 1];
		}

		/** @brief The last element. @pre `!empty()` */
		constexpr T const &back() const {
			atl_assert(size_ > 0);
			return data_[size_ - 1];
		}

		/** @brief Pointer to the inline storage. */
		constexpr T *data() noexcept { return data_.data(); }

		/** @brief Read-only pointer to the inline storage. */
		constexpr T const *data() const noexcept { return data_.data(); }

		/** @brief Element at index. @pre index < size() (asserted). */
		constexpr T &operator[](std::size_t index) {
			atl_assert(index < size_);
			return data_[index];
		}

		/** @brief Read-only element at index. @pre index < size() (asserted). */
		constexpr T const &operator[](std::size_t index) const {
			atl_assert(index < size_);
			return data_[index];
		}

		/** @brief Iterator to the first live element. */
		constexpr T *begin() { return data_.data(); }

		/** @brief Past-the-end iterator. */
		constexpr T *end() { return data_.data() + size_; }

		/** @brief Read-only iterator to the first live element. */
		constexpr T const *begin() const { return data_.data(); }

		/** @brief Read-only past-the-end iterator. */
		constexpr T const *end() const { return data_.data() + size_; }

		/** @brief Element-wise equality of the live elements (the spare capacity is ignored). */
		friend constexpr bool operator==(static_vector const &a, static_vector const &b)
			requires std::equality_comparable<T>
		{
			return std::ranges::equal(a, b);
		}

		/** @brief Lexicographic order of the live elements. */
		friend constexpr auto operator<=>(static_vector const &a, static_vector const &b)
			requires std::three_way_comparable<T>
		{
			return std::lexicographical_compare_three_way(a.begin(), a.end(), b.begin(), b.end());
		}
	};
}
