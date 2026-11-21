#include <atl/strings/palindromic_tree.hpp>
#include <iostream>
#include <string_view>

int main() {
	atl::palindromic_tree<26> tree;
	auto const lowercase = [](char value) { return value - 'a'; };
	auto const suffix = tree.insert(std::string_view("aba"), lowercase);
	tree.insert(std::string_view("aba"), lowercase);
	auto const counts = tree.occurrence_counts();
	std::cout << tree.palindrome_count() << ' ' << tree.length(suffix) << '\n';
	auto const singleton = tree.walk(tree.odd_root(), 0);
	if (singleton) std::cout << counts[suffix] << ' ' << counts[*singleton] << '\n';
}
