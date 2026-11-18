#pragma once

namespace atl::traits::detail {
	/**
	 * @brief Registration point for class-type integers, so the integer traits
	 *        never name them: a specialization sets `is_signed` or `is_unsigned`
	 *        and names `signed_type` and `unsigned_type`. The software 128-bit
	 *        classes register themselves in `<atl/types/int128.hpp>`.
	 */
	template<typename T>
	struct extended_integer {
		static constexpr bool is_signed = false;
		static constexpr bool is_unsigned = false;
	};
}
