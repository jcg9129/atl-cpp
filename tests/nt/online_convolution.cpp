#include <atl/experimental/nt/online_convolution_variants.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/online_convolution.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <limits>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

namespace online_tests {
	/** @brief Oracle: the defining Cauchy sum c_k = sum_{i<=k} left[i] right[k-i], sharing no code with the contenders. */
	template<atl::static_prime_mod_int Coefficient>
	Coefficient cauchy_coefficient(std::vector<Coefficient> const &left, std::vector<Coefficient> const &right, std::size_t k) {
		Coefficient result(0);
		for (std::size_t i = 0; i <= k; ++i) result += left[i] * right[k - i];
		return result;
	}

	/** @brief Compare all streaming contenders with the Cauchy-sum oracle. */
	template<atl::static_prime_mod_int Coefficient>
	void family() {
		using Vector = std::vector<Coefficient>;
		std::mt19937_64 random(2718);
		Vector left(1025), right(1025);
		for (auto &value : left) value = Coefficient(random());
		for (auto &value : right) value = Coefficient(random());
		for (std::size_t cutoff : {1, 16, 32, 64, 128}) {
			atl::online_convolution<Coefficient> library_stream;
			atl::experimental::online_convolution_by_block_cutoff<Coefficient> cached(cutoff);
			atl::experimental::online_convolution_by_uncached_blocks<Coefficient> uncached(cutoff);
			atl::experimental::online_convolution_by_direct_sums<Coefficient> direct(cutoff);
			atl::experimental::online_convolution_by_karatsuba_blocks<Coefficient> karatsuba(cutoff);
			atl::online_fixed_convolution<Coefficient> fixed(right);
			atl::experimental::online_fixed_convolution_by_block_cutoff<Coefficient> fixed_cached(right, cutoff);
			atl::experimental::online_fixed_convolution_by_uncached_blocks<Coefficient> fixed_uncached(right, cutoff);
			atl::experimental::online_fixed_convolution_by_direct_sums<Coefficient> fixed_direct(right, cutoff);
			atl::experimental::online_fixed_convolution_by_karatsuba_blocks<Coefficient> fixed_karatsuba(right, cutoff);
			for (std::size_t k = 0; k < left.size(); ++k) {
				Coefficient const expected = cauchy_coefficient(left, right, k);
				REQUIRE(library_stream.push(left[k], right[k]) == expected);
				REQUIRE(cached.push(left[k], right[k]) == expected);
				REQUIRE(uncached.push(left[k], right[k]) == expected);
				REQUIRE(direct.push(left[k], right[k]) == expected);
				REQUIRE(karatsuba.push(left[k], right[k]) == expected);
				REQUIRE(fixed.push(left[k]) == expected);
				REQUIRE(fixed_cached.push(left[k]) == expected);
				REQUIRE(fixed_uncached.push(left[k]) == expected);
				REQUIRE(fixed_direct.push(left[k]) == expected);
				REQUIRE(fixed_karatsuba.push(left[k]) == expected);
				REQUIRE(library_stream.size() == k + 1);
				REQUIRE(cached.size() == k + 1);
				REQUIRE(uncached.size() == k + 1);
				REQUIRE(direct.size() == k + 1);
				REQUIRE(karatsuba.size() == k + 1);
				REQUIRE(fixed.size() == k + 1);
				REQUIRE(fixed_cached.size() == k + 1);
				REQUIRE(fixed_uncached.size() == k + 1);
				REQUIRE(fixed_direct.size() == k + 1);
				REQUIRE(fixed_karatsuba.size() == k + 1);
			}
		}
	}

	/** @brief Feedback depends only on already returned coefficients; interleave independent streams. */
	template<atl::static_prime_mod_int Coefficient>
	void feedback() {
		using Vector = std::vector<Coefficient>;
		Vector kernel{1, 3, 2, 5}, values{1};
		atl::online_fixed_convolution<Coefficient> stream(kernel), empty(Vector{});
		atl::online_convolution<Coefficient> full;
		for (std::size_t degree = 0; degree < 257; ++degree) {
			Coefficient expected(0), square(0);
			for (std::size_t i = 0; i <= degree; ++i) {
				if (degree - i < kernel.size()) expected += values[i] * kernel[degree - i];
				square += values[i] * values[degree - i];
			}
			REQUIRE(stream.push(values[degree]) == expected);
			REQUIRE(full.push(values[degree], values[degree]) == square);
			REQUIRE(empty.push(values[degree]) == Coefficient(0));
			values.push_back(expected + square);
		}
	}

	/** @brief A kernel much shorter than the stream: blocks past the kernel are zero, and block spectra are computed late. */
	template<atl::static_prime_mod_int Coefficient>
	void short_kernel() {
		using Vector = std::vector<Coefficient>;
		std::mt19937_64 random(31);
		Vector kernel(100), stream(600);
		for (auto &value : kernel) value = Coefficient(random());
		for (auto &value : stream) value = Coefficient(random());
		Vector padded = kernel;
		padded.resize(stream.size());
		atl::online_fixed_convolution<Coefficient> library(kernel);
		atl::experimental::online_fixed_convolution_by_block_cutoff<Coefficient> small_blocks(kernel, 4);
		atl::experimental::online_fixed_convolution_by_uncached_blocks<Coefficient> uncached(kernel, 4);
		atl::experimental::online_fixed_convolution_by_karatsuba_blocks<Coefficient> karatsuba(kernel, 4);
		for (std::size_t k = 0; k < stream.size(); ++k) {
			Coefficient const expected = cauchy_coefficient(stream, padded, k);
			REQUIRE(library.push(stream[k]) == expected);
			REQUIRE(small_blocks.push(stream[k]) == expected);
			REQUIRE(uncached.push(stream[k]) == expected);
			REQUIRE(karatsuba.push(stream[k]) == expected);
		}
	}
}

TEST_CASE("online_convolution: dyadic blocks in every field tier", "[online_convolution]") {
	online_tests::family<atl::mod_int32<998244353>>();
	online_tests::family<atl::mod_int32<1000000007>>();
	online_tests::family<atl::mod_int32<7>>();
	online_tests::family<atl::mod_int32<13>>();
	online_tests::family<atl::mod_int32<2>>();
	online_tests::family<atl::mod_int64<18446744073709551557ULL>>();
}

TEST_CASE("online_convolution: feedback, empty and finite kernels", "[online_convolution]") {
	online_tests::short_kernel<atl::mod_int32<998244353>>();
	online_tests::short_kernel<atl::mod_int32<7>>();
	online_tests::feedback<atl::mod_int32<998244353>>();
	online_tests::feedback<atl::mod_int32<2>>();
	using C = atl::mod_int32<7>;
	std::forward_list<C> const kernel{1, 2, 0, 0};
	atl::online_fixed_convolution<C> stream(kernel);
	REQUIRE(stream.size() == 0);
	REQUIRE(stream.push(C(3)) == C(3));
	REQUIRE(stream.push(C(4)) == C(3));
	REQUIRE(stream.push(C(5)) == C(6));
}

TEST_CASE("online_convolution: contracts", "[online_convolution]") {
	using C = atl::mod_int32<7>;
	using V = std::vector<C>;
	for (std::size_t cutoff : {std::size_t{0}, atl::maximum_series_length / 2 + 1}) {
		CHECK(atl_test::aborts([&] { atl::experimental::online_convolution_by_block_cutoff<C> stream(cutoff); }));
		CHECK(atl_test::aborts([&] { atl::experimental::online_fixed_convolution_by_block_cutoff<C> stream(V{}, cutoff); }));
	}
	V oversized(atl::maximum_series_length + 1);
	oversized.back() = C(1);
	CHECK(atl_test::aborts([&] { atl::online_fixed_convolution<C> stream(oversized); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::online_karatsuba<C>(V{}, V{}); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::online_karatsuba<C>(V{1}, V{1, 2}); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::online_karatsuba<C>(V{1, 2, 3}, V{1, 2, 3}); }));
	// A zero stream exercises the real public cap without constructing an oversized private state.
	CHECK(atl_test::aborts([] {
		atl::online_convolution<C> stream;
		for (std::size_t i = 0; i <= atl::maximum_series_length; ++i) (void)stream.push(C(0), C(0));
	}));
	CHECK(atl_test::aborts([] {
		atl::online_fixed_convolution<C> stream(V{});
		for (std::size_t i = 0; i <= atl::maximum_series_length; ++i) (void)stream.push(C(0));
	}));
}

namespace online_tests {
	/** @brief Count block transforms independently of the scheduler. */
	struct counting_engine : atl::ntt_engine_base<counting_engine> {
		/** @brief Forward transform count. */
		inline static std::size_t forwards = 0;
		/** @brief Inverse transform count. */
		inline static std::size_t inverses = 0;

		/** @brief Delegate scalar support. */
		template<atl::static_mod_int Coefficient>
		static constexpr bool supports() { return atl::scalar_ntt::supports<Coefficient>(); }

		/** @brief Count and perform a forward. */
		template<atl::static_mod_int Coefficient>
		static void forward(decltype(Coefficient::mod()) *values, std::uint32_t levels) {
			++forwards;
			atl::scalar_ntt::forward<Coefficient>(values, levels);
		}

		/** @brief Count and perform an inverse. */
		template<atl::static_mod_int Coefficient>
		static void inverse(decltype(Coefficient::mod()) *values, std::uint32_t levels) {
			++inverses;
			atl::scalar_ntt::inverse<Coefficient>(values, levels);
		}
	};
}

TEST_CASE("online_convolution: cached, uncached and fixed transform counts", "[online_convolution]") {
	using C = atl::mod_int32<998244353>;
	using V = std::vector<C>;
	using E = online_tests::counting_engine;
	atl::experimental::online_convolution_by_block_cutoff<C, E> cached(32);
	atl::experimental::online_convolution_by_uncached_blocks<C, E> uncached(32);
	E::forwards = E::inverses = 0;
	atl::experimental::online_fixed_convolution_by_block_cutoff<C, E> fixed(V(192, C(1)), 32);
	REQUIRE(E::forwards == 1);
	REQUIRE(E::inverses == 0);
	for (std::size_t i = 0; i < 127; ++i) {
		REQUIRE(cached.push(C(1), C(1)) == C(i + 1));
		REQUIRE(uncached.push(C(1), C(1)) == C(i + 1));
		REQUIRE(fixed.push(C(1)) == C(i + 1));
	}
	E::forwards = E::inverses = 0;
	REQUIRE(cached.push(C(1), C(1)) == C(128));
	REQUIRE(E::forwards == 2);
	REQUIRE(E::inverses == 1);
	E::forwards = E::inverses = 0;
	REQUIRE(uncached.push(C(1), C(1)) == C(128));
	REQUIRE(E::forwards == 2);
	REQUIRE(E::inverses == 1);
	E::forwards = E::inverses = 0;
	REQUIRE(fixed.push(C(1)) == C(128));
	REQUIRE(E::forwards == 1);
	REQUIRE(E::inverses == 1);
	for (std::size_t i = 128; i < 191; ++i) {
		REQUIRE(cached.push(C(1), C(1)) == C(i + 1));
		REQUIRE(uncached.push(C(1), C(1)) == C(i + 1));
		REQUIRE(fixed.push(C(1)) == C(i + 1));
	}
	E::forwards = E::inverses = 0;
	REQUIRE(cached.push(C(1), C(1)) == C(192));
	REQUIRE(E::forwards == 2);
	REQUIRE(E::inverses == 1);
	E::forwards = E::inverses = 0;
	REQUIRE(uncached.push(C(1), C(1)) == C(192));
	REQUIRE(E::forwards == 4);
	REQUIRE(E::inverses == 1);
	E::forwards = E::inverses = 0;
	REQUIRE(fixed.push(C(1)) == C(192));
	REQUIRE(E::forwards == 2);
	REQUIRE(E::inverses == 1);
}

namespace online_tests {
	/** @brief Reserving up front, part way or below the current size leaves every result and size unchanged. */
	template<atl::static_prime_mod_int Coefficient>
	void reserve_keeps_results(std::size_t count) {
		std::mt19937_64 random(41);
		std::vector<Coefficient> left(count), right(count), kernel(count / 3 + 1);
		for (auto &value : left) value = Coefficient(random());
		for (auto &value : right) value = Coefficient(random());
		for (auto &value : kernel) value = Coefficient(random());
		atl::online_convolution<Coefficient> plain, reserved, late;
		atl::online_fixed_convolution<Coefficient> fixed_plain(kernel), fixed_reserved(kernel), fixed_late(kernel);
		reserved.reserve(count);
		fixed_reserved.reserve(count);
		for (std::size_t k = 0; k < count; ++k) {
			if (k == count / 2) {
				late.reserve(count);
				fixed_late.reserve(count);
				late.reserve(k / 2);
				fixed_late.reserve(0);
				REQUIRE(late.size() == k);
				REQUIRE(fixed_late.size() == k);
			}
			auto const expected = plain.push(left[k], right[k]);
			REQUIRE(reserved.push(left[k], right[k]) == expected);
			REQUIRE(late.push(left[k], right[k]) == expected);
			auto const fixed_expected = fixed_plain.push(left[k]);
			REQUIRE(fixed_reserved.push(left[k]) == fixed_expected);
			REQUIRE(fixed_late.push(left[k]) == fixed_expected);
		}
		REQUIRE(reserved.size() == count);
		REQUIRE(fixed_reserved.size() == count);
	}
}

TEST_CASE("online_convolution: reserve keeps results and size", "[online_convolution]") {
	for (std::size_t count : {0, 1, 2, 63, 64, 65, 300}) {
		CAPTURE(count);
		online_tests::reserve_keeps_results<atl::mod_int32<998244353>>(count);
		online_tests::reserve_keeps_results<atl::mod_int32<7>>(count);
	}
	using C = atl::mod_int32<7>;
	atl::online_convolution<C> stream;
	stream.reserve(atl::maximum_series_length);
	REQUIRE(stream.size() == 0);
	atl::online_fixed_convolution<C> fixed(std::vector<C>{1, 2});
	CHECK(atl_test::aborts([&] { stream.reserve(atl::maximum_series_length + 1); }));
	CHECK(atl_test::aborts([&] { fixed.reserve(atl::maximum_series_length + 1); }));
	CHECK(atl_test::aborts([&] { fixed.reserve((std::numeric_limits<std::size_t>::max)()); }));
}
