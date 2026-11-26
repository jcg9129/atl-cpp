#pragma once
#include <atl/ds/top_clusters/common.hpp>
#include <atl/types/alias.hpp>
#include <concepts>
#include <cstddef>

namespace atl::detail {
	/** @brief Named operations shared by cluster sets that summarize vertex sums and counts. */
	template<typename Derived, std::copyable Value, typename Tag>
	class top_aggregate_members {
		constexpr Derived &self() { return static_cast<Derived &>(*this); }

		/** @brief The tag adding @p x (the summing sets). */
		static constexpr Tag add_tag(Value const &x) {
			return Tag::add_of(x);
		}

	public:
		/** @brief The sum of the vertex values on the u - v path. @complexity amortized O(log n) */
		[[nodiscard]] constexpr Value path_sum(uint32 u, uint32 v) {
			return self().fold(u, v).spine.sum;
		}

		/** @brief The sum over v's subtree under the current root. @complexity amortized O(log n) */
		[[nodiscard]] constexpr Value subtree_sum(uint32 v) {
			return self().subtree_fold(v).total().sum;
		}

		/** @brief The sum over v's side of the edge {v, p}. @complexity amortized O(log n) */
		[[nodiscard]] constexpr Value subtree_sum(uint32 v, uint32 p) {
			return self().subtree_fold(v, p).total().sum;
		}

		/** @brief The sum over v's whole tree. @complexity amortized O(log n) */
		[[nodiscard]] constexpr Value component_sum(uint32 v) {
			return self().component_fold(v).total().sum;
		}

		/** @brief The number of vertices in v's subtree under the current root. @complexity amortized O(log n) */
		[[nodiscard]] constexpr std::size_t subtree_size(uint32 v) {
			return self().subtree_fold(v).total().count;
		}

		/** @brief The number of vertices on v's side of the edge {v, p}. @complexity amortized O(log n) */
		[[nodiscard]] constexpr std::size_t subtree_size(uint32 v, uint32 p) {
			return self().subtree_fold(v, p).total().count;
		}

		/** @brief The number of vertices in v's tree. @complexity amortized O(log n) */
		[[nodiscard]] constexpr std::size_t component_size(uint32 v) {
			return self().component_fold(v).total().count;
		}

		/** @brief Add @p x to every vertex on the u - v path. @complexity amortized O(log n) */
		constexpr void path_add(uint32 u, uint32 v, Value const &x) {
			self().apply(u, v, add_tag(x));
		}

		/** @brief Add @p x to every vertex of v's subtree. @complexity amortized O(log n) */
		constexpr void subtree_add(uint32 v, Value const &x) {
			self().subtree_apply(v, add_tag(x));
		}

		/** @brief Add @p x to every vertex on v's side of the edge {v, p}. @complexity amortized O(log n) */
		constexpr void subtree_add(uint32 v, uint32 p, Value const &x) {
			self().subtree_apply(v, p, add_tag(x));
		}
	};
}

namespace atl::top_clusters::detail {
	/** @brief The path type of @ref atl::top_clusters::sum or @ref atl::top_clusters::extremes : the summing named members. */
	template<typename Path>
	concept sum_family = is_ready_made<Path> && requires(Path const &p) {
		p.spine.sum;
		p.light.count;
	};
}
