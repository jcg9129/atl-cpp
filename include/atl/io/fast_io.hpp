#pragma once

/**
 * @file
 * @brief Umbrella for atl's fast competitive I/O (reader + writer).
 *
 *   auto [n, q] = atl::io::read<int, int>();
 *   auto a      = atl::io::read_n<std::uint32_t>(n);
 *   atl::io::writeln(x);
 *
 * Default reader path is fread; define ATL_USE_MMAP for the POSIX mmap path.
 * Default writer table is 2-wide; define ATL_IO_BIG_LUT for the 4-wide one.
 */

#include <atl/io/reader.hpp>
#include <atl/io/writer.hpp>
