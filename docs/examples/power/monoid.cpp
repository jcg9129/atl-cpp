#include <atl/nt/mod_int.hpp>
#include <atl/nt/power.hpp>
#include <iostream>

struct repetition {
	int copies;

	friend constexpr repetition operator*(repetition a, repetition b) { return {a.copies + b.copies}; }
};

int main() {
	std::cout << atl::power(3ull, 10u) << '\n';
	std::cout << atl::power(atl::mod_int32<7>(3), 4u).value() << '\n';
	std::cout << atl::power(repetition{2}, 3u, repetition{0}).copies << '\n';
}
