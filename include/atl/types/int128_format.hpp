#pragma once

/**
 * @file
 * @brief `std::format` support for `atl::int128` and `atl::uint128` on every target.
 *
 * Where the aliases are native integers the standard library formats them
 * itself and this header adds nothing; otherwise it brings in the formatters
 * of the software classes (`<atl/types/detail/software_int128_format.hpp>`).
 */

#include <atl/config.hpp>
#include <atl/types/alias.hpp>
#include <format>

#if !ATL_HAS_INT128
#include <atl/types/detail/software_int128_format.hpp>
#endif
