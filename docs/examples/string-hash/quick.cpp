#include <atl/strings/string_hash.hpp>
#include <iostream>
#include <string_view>

int main() {
	atl::string_hash const text(std::string_view("abcab"), 131);
	std::cout << text.equal(0, 2, 3, 5) << ' ' << text.lcp(0, 3) << '\n';
	std::cout << (text.full_hash() == atl::hash_of(std::string_view("abcab"), 131)) << '\n';
	atl::string_hash online(131);
	online.push_back('a');
	online.push_back('b');
	std::cout << (online.full_hash() == text.substring_hash(3, 5)) << '\n';
}
