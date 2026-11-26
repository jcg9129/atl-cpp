#pragma once

#include <atl/nt/convolution_mod.hpp>
#include <atl/nt/detail/power_series_native_base.hpp>
#include <atl/nt/ntt_engine_core.hpp>
#include <atl/nt/prime_mod_int_core.hpp>
#include <atl/nt/series_calculus_core.hpp>
#include <atl/types/traits/widen_mul.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace atl::detail {
	/** @brief Full convolution for native fields whose capped products may exceed the direct transform order. */
	template<native_series_coefficient Coefficient>
		requires(!direct_series_coefficient<Coefficient>)
	struct native_series_convolution<Coefficient> {
		template<ntt_engine Engine>
		static constexpr auto multiply(std::span<Coefficient const> left, std::span<Coefficient const> right) {
			return convolution<Engine>(left, right);
		}
	};
}
