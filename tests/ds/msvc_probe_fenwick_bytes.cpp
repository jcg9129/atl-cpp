// Temporary MSVC ARM64 probe for ds-2 (removed before merge).
#include <atl/ds/binary_indexed_tree.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

namespace msvc_probe {
	using sizes_t = std::array<std::size_t, 2>;

	template<typename T, typename Op, bool NoUniqueAddress>
	struct copied_op;

	template<typename T, typename Op>
	struct copied_op<T, Op, true> {
		sizes_t sizes_;
		sizes_t strides_{};
		std::vector<T> data_;
		[[no_unique_address]] Op op_;
		T identity_;

		constexpr explicit copied_op(sizes_t sizes, T identity = T{}, Op op = Op{})
			: sizes_(sizes), op_(std::move(op)), identity_(std::move(identity)) {}
	};

	template<typename T, typename Op>
	struct copied_op<T, Op, false> {
		sizes_t sizes_;
		sizes_t strides_{};
		std::vector<T> data_;
		Op op_;
		T identity_;

		constexpr explicit copied_op(sizes_t sizes, T identity = T{}, Op op = Op{})
			: sizes_(sizes), op_(std::move(op)), identity_(std::move(identity)) {}
	};

	template<typename Object>
	void dump(char const *what, Object const &object) {
		unsigned char bytes[sizeof(Object)];
		std::memcpy(bytes, static_cast<void const *>(&object), sizeof(Object));
		std::printf("probe %s bytes:", what);
		for (std::size_t i = 0; i < sizeof(Object); ++i) std::printf("%s%02x", i % 8 == 0 ? " " : "", bytes[i]);
		std::printf("\n");
		std::fflush(stdout);
	}

	template<typename Replica>
	void show(char const *what) {
		Replica r(sizes_t{3, 4});
		auto const base = reinterpret_cast<char const *>(&r);
		std::printf("probe %s: sizes=%zu,%zu offsets sizes=%td strides=%td data=%td op=%td identity=%td sizeof=%zu\n", what,
					r.sizes_[0], r.sizes_[1], reinterpret_cast<char const *>(&r.sizes_) - base,
					reinterpret_cast<char const *>(&r.strides_) - base, reinterpret_cast<char const *>(&r.data_) - base,
					reinterpret_cast<char const *>(&r.op_) - base, reinterpret_cast<char const *>(&r.identity_) - base,
					sizeof(Replica));
		std::fflush(stdout);
		dump(what, r);
	}
}

TEST_CASE("msvc probe fenwick bytes", "[msvc_probe]") {
	msvc_probe::show<msvc_probe::copied_op<int, std::plus<>, true>>("copied nua");
	msvc_probe::show<msvc_probe::copied_op<int, std::plus<>, false>>("copied plain");
	atl::binary_indexed_tree<int, 2> tree(msvc_probe::sizes_t{3, 4});
	std::printf("probe library: sizes=%zu,%zu\n", tree.sizes()[0], tree.sizes()[1]);
	msvc_probe::dump("library", tree);
#if defined(_M_ARM64)
	FAIL("print the probe output on MSVC ARM64");
#else
	SUCCEED();
#endif
}
