/**
 * @file
 * @brief Online (relaxed) convolution: the library classes of `<atl/nt/online_convolution.hpp>` against their contenders.
 *
 * Raced for the fully relaxed product of two streams (`online_convolution`) and for a stream against a fixed kernel
 * (`online_fixed_convolution`), each with the one thing the contender does differently:
 * - library = dyadic rectangles with the spectra of the stationary blocks kept, blocks up to 32 multiplied directly;
 * - by block cutoff = the library schedule with the direct block size swept over 16, 24, 32, 48, 64, 96 and 128;
 * - by uncached blocks = the same, transforming the stationary blocks again for every rectangle;
 * - by Karatsuba blocks = the same, with recursive Karatsuba for the block products instead of transforms;
 * - by direct sums = the defining quadratic sum at every push (no blocks, so no cutoff).
 *
 * Every contender is checked against the offline product of the same inputs before any timing. Each sample pushes all n
 * coefficients through a freshly constructed stream, so it measures the total over a whole stream (including construction and,
 * for a fixed kernel, the kernel transforms), not one push. Lengths n = 2^10, 2^12, 2^16, 2^19 and 500000 modulo 998244353,
 * and 2^16 modulo 1000000007; the quadratic contender runs only up to 2^12.
 *
 * Result: not measured yet (race pending). The page docs/online-convolution.md has the table once a record exists.
 */
#include <atl/experimental/nt/online_convolution_variants.hpp>
#include <atl/nt/detail/power_series.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/online_convolution.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
	namespace ex = atl::experimental;

	/** @brief The quadratic contender is never run above this length. */
	constexpr std::size_t quadratic_limit = std::size_t{1} << 12;

	/** @brief A stream of two inputs per push. */
	template<typename Stream, typename Coefficient>
	concept full_stream = requires(Stream &stream, Coefficient value) {
		{ stream.push(value, value) } -> std::same_as<Coefficient>;
	};

	/** @brief A stream of one input per push, against a fixed kernel. */
	template<typename Stream, typename Coefficient>
	concept fixed_stream = requires(Stream &stream, Coefficient value) {
		{ stream.push(value) } -> std::same_as<Coefficient>;
	};

	/** @brief Push every coefficient of both inputs through a stream and collect the returned products. */
	template<atl::static_prime_mod_int Coefficient, full_stream<Coefficient> Stream>
	auto stream_full(std::vector<Coefficient> const &left, std::vector<Coefficient> const &right, Stream stream) -> std::vector<Coefficient> {
		std::vector<Coefficient> result;
		result.reserve(left.size());
		for (std::size_t i = 0; i < left.size(); ++i) result.push_back(stream.push(left[i], right[i]));
		return result;
	}

	/** @brief Push every coefficient of the input through a fixed-kernel stream and collect the returned products. */
	template<atl::static_prime_mod_int Coefficient, fixed_stream<Coefficient> Stream>
	auto stream_fixed(std::vector<Coefficient> const &left, Stream stream) -> std::vector<Coefficient> {
		std::vector<Coefficient> result;
		result.reserve(left.size());
		for (std::size_t i = 0; i < left.size(); ++i) result.push_back(stream.push(left[i]));
		return result;
	}

	/** @brief Check a contender against the expected result, then time it under the label. */
	template<typename Result, typename Contender>
		requires std::is_invocable_r_v<Result, Contender &>
	void time_contender(std::string const &label, Result const &expected, Contender contender) {
		REQUIRE(contender() == expected);
		BENCHMARK(std::string(label)) { return contender(); };
	}

	/** @brief The fully relaxed classes on random inputs; race pending. */
	template<atl::static_prime_mod_int Coefficient>
	void race_full(std::size_t length) {
		using Vector = std::vector<Coefficient>;
		std::mt19937_64 random(7103);
		Vector left(length), right(length);
		for (auto &value : left) value = Coefficient(random());
		for (auto &value : right) value = Coefficient(random());
		auto const expected = atl::detail::series_product<atl::default_ntt_engine, Coefficient>(left, right, length);
		auto const shape = " | p=" + std::to_string(Coefficient::mod()) + " n=" + std::to_string(length);
		time_contender("online_convolution (library)" + shape, expected, [&] { return stream_full<Coefficient>(left, right, atl::online_convolution<Coefficient>()); });
		if (length <= quadratic_limit)
			time_contender("online_convolution by direct sums" + shape, expected, [&] { return stream_full<Coefficient>(left, right, ex::online_convolution_by_direct_sums<Coefficient>()); });
		for (std::size_t cutoff : {16, 24, 32, 48, 64, 96, 128}) {
			auto const cutoff_shape = shape + " cutoff=" + std::to_string(cutoff);
			time_contender("online_convolution by block cutoff" + cutoff_shape, expected, [&] { return stream_full<Coefficient>(left, right, ex::online_convolution_by_block_cutoff<Coefficient>(cutoff)); });
			time_contender("online_convolution by uncached blocks" + cutoff_shape, expected, [&] { return stream_full<Coefficient>(left, right, ex::online_convolution_by_uncached_blocks<Coefficient>(cutoff)); });
			time_contender("online_convolution by Karatsuba blocks" + cutoff_shape, expected, [&] { return stream_full<Coefficient>(left, right, ex::online_convolution_by_karatsuba_blocks<Coefficient>(cutoff)); });
		}
	}

	/** @brief The fixed-kernel classes on random inputs; race pending. */
	template<atl::static_prime_mod_int Coefficient>
	void race_fixed(std::size_t length) {
		using Vector = std::vector<Coefficient>;
		std::mt19937_64 random(7103);
		Vector left(length), kernel(length);
		for (auto &value : left) value = Coefficient(random());
		for (auto &value : kernel) value = Coefficient(random());
		auto const expected = atl::detail::series_product<atl::default_ntt_engine, Coefficient>(left, kernel, length);
		auto const shape = " | p=" + std::to_string(Coefficient::mod()) + " n=" + std::to_string(length) + " fixed kernel";
		time_contender("online_fixed_convolution (library)" + shape, expected, [&] { return stream_fixed<Coefficient>(left, atl::online_fixed_convolution<Coefficient>(kernel)); });
		if (length <= quadratic_limit)
			time_contender("online_fixed_convolution by direct sums" + shape, expected, [&] { return stream_fixed<Coefficient>(left, ex::online_fixed_convolution_by_direct_sums<Coefficient>(kernel)); });
		for (std::size_t cutoff : {16, 24, 32, 48, 64, 96, 128}) {
			auto const cutoff_shape = shape + " cutoff=" + std::to_string(cutoff);
			time_contender("online_fixed_convolution by block cutoff" + cutoff_shape, expected, [&] { return stream_fixed<Coefficient>(left, ex::online_fixed_convolution_by_block_cutoff<Coefficient>(kernel, cutoff)); });
			time_contender("online_fixed_convolution by uncached blocks" + cutoff_shape, expected, [&] { return stream_fixed<Coefficient>(left, ex::online_fixed_convolution_by_uncached_blocks<Coefficient>(kernel, cutoff)); });
			time_contender("online_fixed_convolution by Karatsuba blocks" + cutoff_shape, expected, [&] { return stream_fixed<Coefficient>(left, ex::online_fixed_convolution_by_karatsuba_blocks<Coefficient>(kernel, cutoff)); });
		}
	}
}

TEST_CASE("online_convolution: library versus contenders (race pending)", "[online_convolution]") {
	for (std::size_t length : {std::size_t{1} << 10, quadratic_limit, std::size_t{1} << 16, std::size_t{1} << 19, std::size_t{500000}}) {
		race_full<atl::mod_int32<998244353>>(length);
		race_fixed<atl::mod_int32<998244353>>(length);
	}
	race_full<atl::mod_int32<1000000007>>(std::size_t{1} << 16);
	race_fixed<atl::mod_int32<1000000007>>(std::size_t{1} << 16);
}
