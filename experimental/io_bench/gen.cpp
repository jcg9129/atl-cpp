#include <charconv>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string_view>
#include <system_error>
#include <utility>

int main(int argc, char **argv) {
	if (argc < 3 || argc > 4) return 1;
	auto parse = [](char const *text, std::uint32_t &value) {
		std::string_view const token(text);
		auto const result = std::from_chars(token.data(), token.data() + token.size(), value);
		return result.ec == std::errc{} && result.ptr == token.data() + token.size();
	};
	std::uint32_t n = 0, q = 0, seed = 42;
	if (!parse(argv[1], n) || n == 0 || !parse(argv[2], q) || (argc == 4 && !parse(argv[3], seed))) return 1;
	std::mt19937 rng(seed);
	std::printf("%u %u\n", unsigned(n), unsigned(q));
	for (std::uint32_t i = 0; i < n; ++i) std::printf("%u%c", unsigned(rng() % 1'000'000'000u), i + 1 < n ? ' ' : '\n');
	for (std::uint32_t i = 0; i < q; ++i) {
		std::uint32_t a = std::uint32_t(rng()) % n, b = std::uint32_t(rng()) % n;
		if (a > b) std::swap(a, b);
		std::printf("%u %u\n", unsigned(a), unsigned(b + 1));
	}
}
