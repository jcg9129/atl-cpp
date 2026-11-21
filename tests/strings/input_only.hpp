#pragma once
#include <concepts>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

namespace test {
	/**
	 * @brief A single-pass view of a vector's elements: models
	 *        std::ranges::input_range and nothing stronger, so it exercises the
	 *        input-only paths of word-collection consumers.
	 */
	template<typename T>
		requires std::is_object_v<T> && std::destructible<T> && std::same_as<T, std::remove_cv_t<T>> && (!std::same_as<T, bool>)
	class input_only {
		std::vector<T> const *items_;

	public:
		/** @brief The single-pass iterator: no copy-stable multi-pass guarantee advertised. */
		class iterator {
			T const *cur_ = nullptr, *end_ = nullptr;

		public:
			using value_type = T;
			using difference_type = std::ptrdiff_t;
			using iterator_concept = std::input_iterator_tag;

			iterator() = default;

			iterator(T const *cur, T const *end) : cur_(cur), end_(end) {}

			iterator(iterator &&) = default;
			iterator &operator=(iterator &&) = default;

			T const &operator*() const { return *cur_; }

			iterator &operator++() {
				++cur_;
				return *this;
			}

			void operator++(int) { ++cur_; }

			bool operator==(std::default_sentinel_t) const { return cur_ == end_; }
		};

		explicit input_only(std::vector<T> const &items) : items_(&items) {}

		iterator begin() const { return {items_->data(), items_->empty() ? items_->data() : items_->data() + items_->size()}; }

		std::default_sentinel_t end() const { return {}; }
	};
}
