#include <atl/nt/convolution.hpp>
#include <atl/nt/fft.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
	using Z = atl::mod_int<atl::uint32, 998244353>;
	std::vector<Z> a{1, 2, 3}, b{4, 5};
	auto const product = atl::convolution(a, b);
	for (std::size_t i = 0; i < product.size(); ++i) std::cout << (i == 0 ? "" : " ") << product[i].value();
	std::cout << '\n';
	std::vector<unsigned> counts{1, 2, 1};
	auto const exact = atl::convolution_uint64(counts, counts);
	std::cout << exact[2] << '\n';
	std::vector<Z> spectrum{1, 2, 3, 4};
	atl::ntt(spectrum);
	atl::inverse_ntt(spectrum);
	std::cout << spectrum[3].value() << '\n';
	std::vector<atl::cplx> signal{{1, 0}, {2, 0}};
	atl::fft(signal);
	std::cout << signal[0].re << '\n';
}
