#pragma once

#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

namespace atl_test {
	/** @brief A range over a copy of its elements that offers single-pass input iterators and nothing more, also through a const reference. */
	template<typename Element>
	class input_only {
		std::vector<Element> elements_;

	public:
		/** @brief Input iterator: post-increment returns void and iterators are compared only with the end sentinel. */
		class iterator {
			Element const *position_ = nullptr;
			Element const *last_ = nullptr;

		public:
			using value_type = Element;
			using difference_type = std::ptrdiff_t;
			using iterator_concept = std::input_iterator_tag;

			iterator() = default;

			iterator(Element const *position, Element const *last) : position_(position), last_(last) {}

			Element const &operator*() const { return *position_; }

			iterator &operator++() {
				++position_;
				return *this;
			}

			void operator++(int) { ++position_; }

			friend bool operator==(iterator const &position, std::default_sentinel_t) { return position.position_ == position.last_; }
		};

		/** @brief Copy the elements. */
		explicit input_only(std::vector<Element> elements) : elements_(std::move(elements)) {}

		/** @brief Iterator at the first element. */
		iterator begin() const { return iterator(elements_.data(), elements_.data() + elements_.size()); }

		/** @brief End sentinel. */
		std::default_sentinel_t end() const { return {}; }
	};
}
