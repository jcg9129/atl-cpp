#include <atl/strings/palindrome_lengths.hpp>
#include <iostream>
#include <string_view>

int main() {
	atl::palindrome_lengths const text(std::string_view("abacaba"));
	std::cout << text.odd_length(3) << '\n';
	std::cout << text.is_palindrome(0, 7) << ' ' << text.is_palindrome(0, 2) << '\n';
	atl::palindrome_lengths const even(std::string_view("abba"));
	std::cout << even.even_length(2) << '\n';
}
