#pragma once

/** @file
 * @brief The online_convolution API for prime moduli up to 32 bits with native double-width coefficient multiplication.
 */

#include <atl/nt/detail/online_convolution_native.hpp>
#include <atl/nt/detail/power_series_native.hpp>
#include <atl/nt/ntt_engine_core.hpp>
#include <atl/nt/prime_mod_int_core.hpp>
#include <concepts>
#include <cstddef>

#include <atl/nt/online_convolution_ntt.hpp>
