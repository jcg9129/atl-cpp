#include <atl/strings/static_trie.hpp>
#include <array>
#include <iostream>
#include <string_view>

int main() {
	using std::string_view;
	auto const symbol = [](char c) { return c - 'a'; };
	atl::static_trie<26> const words(std::array{string_view("b"), string_view("ab"), string_view("abc")}, symbol);
	auto const node = words.find(string_view("ab"), symbol);
	if (!node) return 1;
	auto const interval = words.word_interval(*node);
	std::cout << interval.begin << ' ' << interval.end << '\n';
	for (auto i = interval.begin; i < interval.end; ++i)
		std::cout << words.word_order()[i] << '\n';
}
