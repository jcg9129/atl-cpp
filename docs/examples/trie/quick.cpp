#include <atl/strings/trie.hpp>
#include <iostream>
#include <string_view>

int main() {
	auto const symbol = [](char c) { return c - 'a'; };
	atl::trie<26> words;
	words.insert(std::string_view("tea"), symbol);
	words.insert(std::string_view("ten"), symbol);
	words.insert(std::string_view("ten"), symbol);
	std::cout << words.count(std::string_view("ten"), symbol) << ' '
			  << words.count_with_prefix(std::string_view("te"), symbol) << '\n';
	std::cout << words.erase(std::string_view("ten"), symbol) << ' ' << words.size() << '\n';
}
