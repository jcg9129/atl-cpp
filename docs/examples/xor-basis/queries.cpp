#include <atl/bits/xor_basis.hpp>
#include <atl/types/alias.hpp>
#include <array>
#include <iostream>

int main() {
	std::array<atl::uint32, 3> const words{6, 5, 3};
	atl::xor_basis const basis(words);
	std::cout << "span:";
	for (atl::uint32 k = 0; k < basis.representable_count(); ++k)
		std::cout << ' ' << basis.kth_smallest(k);
	std::cout << "\nmaximum: " << basis.max_xor() << '\n';
	atl::stamped_xor_basis<atl::uint32> stamped;
	for (atl::uint32 i = 0; i < words.size(); ++i) stamped.insert(words[i], i);
	std::cout << "suffix maximum: " << stamped.max_xor(2) << '\n';
}
