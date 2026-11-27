#pragma once

#include <atl/nt/convolution_mod.hpp>
#include <atl/nt/detail/polynomial_points_native.hpp>
#include <atl/nt/detail/power_series_extensions_native.hpp>
#include <atl/nt/detail/power_series_native.hpp>
#include <atl/nt/ntt_engine_core.hpp>
#include <atl/nt/prime_mod_int_core.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

#include <atl/nt/detail/online_convolution_native_base.hpp>
