#include <atl/strings/lyndon.hpp>
#include <iostream>
#include <string_view>

int main() {
	for (auto boundary : atl::lyndon_factorization(std::string_view("banana"))) std::cout << boundary << '\n';
	std::cout << atl::minimal_rotation_index(std::string_view("cab")) << '\n';
	for (auto repetition : atl::runs(std::string_view("aaaa")))
		std::cout << repetition.period << ' ' << repetition.l << ' ' << repetition.r << '\n';
}
