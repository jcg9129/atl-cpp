#include <atl/io/reader.hpp>
#include <atl/io/writer.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdio>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace {
	// The field with two elements supplies a published-concept model without
	// depending on the unpublished modular integer implementations.
	class mod_two {
		atl::uint32 value_ = 0;

	public:
		mod_two() = default;

		template<atl::integral Integer>
		explicit mod_two(Integer value) : value_(atl::uint32(value & Integer(1))) {}

		static atl::uint32 mod() { return 2; }

		atl::uint32 value() const { return value_; }

		bool operator==(mod_two const &) const = default;

		mod_two operator+(mod_two b) const { return mod_two(value_ ^ b.value_); }

		mod_two operator-(mod_two b) const { return *this + b; }

		mod_two operator*(mod_two b) const { return mod_two(value_ & b.value_); }

		mod_two operator/(mod_two b) const { return *this * b.inverse(); }

		mod_two operator-() const { return *this; }

		mod_two &operator+=(mod_two b) { return *this = *this + b; }

		mod_two &operator-=(mod_two b) { return *this = *this - b; }

		mod_two &operator*=(mod_two b) { return *this = *this * b; }

		mod_two &operator/=(mod_two b) { return *this = *this / b; }

		mod_two inverse() const {
			REQUIRE(value_ != 0);
			return *this;
		}

		mod_two power(atl::uint64 exponent) const { return exponent == 0 ? mod_two(1) : *this; }
	};

	template<typename T>
	concept extracts_into = requires(atl::io::reader &in, T &value) { in >> value; };

	template<typename T>
	concept tries_into = requires(atl::io::reader &in, T &value) { in.try_read(value); };

	template<typename... Ts>
	concept member_reads_into = requires(atl::io::reader &in, Ts &...values) { in.read(values...); };

	template<typename... Ts>
	concept free_reads_into = requires(Ts &...values) { atl::io::read(values...); };

	template<typename T>
	concept member_read_n = requires(atl::io::reader &in) { in.template read_n<T>(std::size_t(2)); };

	template<typename T>
	concept free_read_n = requires { atl::io::read_n<T>(std::size_t(2)); };

	template<typename Reader>
	concept exposes_malformed_skip = requires(Reader &in) { in.skip_malformed_token(nullptr); };

	static_assert(atl::io::readable_into<int>);
	static_assert(atl::io::readable_into<bool>);
	static_assert(atl::io::readable_into<char>);
#if defined(__cpp_lib_to_chars)
	static_assert(atl::io::readable_into<float>);
#endif
	static_assert(atl::io::readable_into<atl::int128>);
	static_assert(atl::io::readable_into<atl::uint128>);
	static_assert(atl::modular_integer<mod_two>);
	static_assert(atl::io::readable_into<mod_two>);
	static_assert(atl::io::readable_into<std::string>);
	static_assert(atl::io::readable<int const>);
	static_assert(!atl::io::readable_into<int const>);
	static_assert(extracts_into<int> && !extracts_into<int const>);
	static_assert(tries_into<int> && !tries_into<int const>);
	static_assert(member_reads_into<> && member_reads_into<int, mod_two>);
	static_assert(free_reads_into<> && free_reads_into<int, mod_two>);
	static_assert(!member_reads_into<int const> && !member_reads_into<int, int const>);
	static_assert(!free_reads_into<int const> && !free_reads_into<int, int const>);
	static_assert(member_read_n<int> && free_read_n<int>);
	static_assert(member_read_n<atl::uint128> && free_read_n<atl::uint128>);
	static_assert(!member_read_n<int const> && !free_read_n<int const>);
	static_assert(!member_read_n<int volatile> && !free_read_n<int volatile>);
	static_assert(!member_read_n<int const volatile> && !free_read_n<int const volatile>);
	static_assert(!member_read_n<atl::uint128 const> && !free_read_n<atl::uint128 const>);
	static_assert(!exposes_malformed_skip<atl::io::reader>);
	static_assert(requires(atl::io::reader &in) {
		in.read<int const>();
		in.read_one<int const>();
		atl::io::read<int const>();
	});

	struct const_only_range {
		int values[3] = {2, 4, 6};

		int *begin() = delete;
		int *end() = delete;

		int const *begin() const { return values; }

		int const *end() const { return values + 3; }
	};

	struct mutable_only_range {
		int values[3] = {2, 4, 6};

		int *begin() { return values; }

		int *end() { return values + 3; }
	};

	struct unprintable {};

	// A proxy produced by value is writable as a named mutable reference, but
	// writable<mutable_text> rejects the const reference it normally checks.
	struct mutable_text {
		std::string_view value;

		operator std::string_view() { return value; }
	};

	struct make_mutable_text {
		mutable_text operator()(std::string_view value) const { return {value}; }
	};

	using proxy_range = std::ranges::transform_view<std::ranges::ref_view<std::vector<std::string_view>>, make_mutable_text>;

	template<typename R>
	concept member_writeln = requires(atl::io::writer &out, R const &range) { out.writeln(range); };

	template<typename R>
	concept free_writeln = requires(R const &range) { atl::io::writeln(range); };

	static_assert(std::ranges::input_range<const_only_range const>);
	static_assert(!std::ranges::input_range<const_only_range>);
	static_assert(member_writeln<const_only_range> && free_writeln<const_only_range>);
	static_assert(std::ranges::input_range<mutable_only_range>);
	static_assert(!std::ranges::input_range<mutable_only_range const>);
	static_assert(!member_writeln<mutable_only_range> && !free_writeln<mutable_only_range>);
	static_assert(!member_writeln<std::vector<unprintable>> && !free_writeln<std::vector<unprintable>>);
	static_assert(member_writeln<std::string> && free_writeln<std::string>);
	static_assert(std::ranges::input_range<proxy_range const>);
	static_assert(!atl::io::writable<mutable_text>);
	static_assert(member_writeln<proxy_range> && free_writeln<proxy_range>);
}

TEST_CASE("io: assignable destinations and unqualified vector elements", "[io]") {
	std::FILE *file = std::tmpfile();
	REQUIRE(file != nullptr);
	std::fputs("7 -3 1 0 x -1 2 9 10 11", file);
	std::rewind(file);
	{
		atl::io::reader in(file);
		REQUIRE(in.read());
		REQUIRE(atl::io::read());
		REQUIRE(in.read<int const>() == 7);
		REQUIRE(in.read_one<int const>() == -3);
		bool flag = false;
		in >> flag;
		REQUIRE(flag);
		REQUIRE(in.try_read(flag));
		REQUIRE_FALSE(flag);
		char letter = 0;
		mod_two odd, even;
		REQUIRE(in.read(letter, odd, even));
		REQUIRE(letter == 'x');
		REQUIRE(odd.value() == 1);
		REQUIRE(even.value() == 0);
		REQUIRE(in.read_n<int>(0).empty());
		REQUIRE(in.read_n<int>(4) == std::vector<int>{9, 10, 11, 0});
		int untouched = 42;
		REQUIRE_FALSE(in.try_read(untouched));
		REQUIRE(untouched == 42);
		REQUIRE(in.read());
	}
	std::fclose(file);
}

TEST_CASE("io: writeln iterates const-only ranges", "[io]") {
	std::FILE *file = std::tmpfile();
	REQUIRE(file != nullptr);
	{
		atl::io::writer out(file);
		const_only_range range;
		out.writeln(range);
		std::vector<std::string_view> values{"ef", "gh"};
		auto proxies = values | std::views::transform(make_mutable_text{});
		out.writeln(proxies);
		// Taking the address instantiates the forwarding body without writing to stdout.
		[[maybe_unused]] auto const emit = &atl::io::writeln<proxy_range>;
		out.writeln(std::vector<int>{});
		out.writeln(std::string("ab cd"));
	}
	std::rewind(file);
	char text[64]{};
	REQUIRE(std::fread(text, 1, sizeof text - 1, file) == 19);
	REQUIRE(std::string(text) == "2 4 6\nef gh\n\nab cd\n");
	std::fclose(file);
}
