// Temporary MSVC ARM64 probe for ds-2 (removed before merge).
#include <array>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <vector>

namespace msvc_probe {
	using sizes_t = std::array<std::size_t, 2>;

#define ATL_PROBE_BODY(OP_DECL, STRIDES_DECL)                                                                          \
	std::array<std::size_t, 2> sizes_;                                                                                 \
	STRIDES_DECL;                                                                                                      \
	std::vector<int> data_;                                                                                            \
	OP_DECL;                                                                                                           \
	int identity_;                                                                                                     \
	std::size_t init_strides(bool forward) {                                                                           \
		std::size_t total = 1;                                                                                         \
		if (forward) {                                                                                                 \
			for (std::size_t k = 0; k < 2; ++k) {                                                                      \
				std::size_t const d = 1 - k;                                                                           \
				strides_[d] = total;                                                                                   \
				total *= sizes_[d];                                                                                    \
			}                                                                                                          \
		} else {                                                                                                       \
			for (std::size_t d = 2; d-- > 0;) {                                                                        \
				strides_[d] = total;                                                                                   \
				if (!(sizes_[d] == 0 || total <= (std::numeric_limits<std::size_t>::max)() / sizes_[d])) std::abort(); \
				total *= sizes_[d];                                                                                    \
			}                                                                                                          \
		}                                                                                                              \
		return total;                                                                                                  \
	}

	struct nua_init {
		ATL_PROBE_BODY([[no_unique_address]] std::plus<> op_, sizes_t strides_{})

		nua_init(std::array<std::size_t, 2> s, bool forward) : sizes_(s), op_(), identity_(0) {
			data_.assign(init_strides(forward), identity_);
		}
	};

	struct plain_init {
		ATL_PROBE_BODY(std::plus<> op_, sizes_t strides_{})

		plain_init(std::array<std::size_t, 2> s, bool forward) : sizes_(s), op_(), identity_(0) {
			data_.assign(init_strides(forward), identity_);
		}
	};

	struct nua_uninit {
		ATL_PROBE_BODY([[no_unique_address]] std::plus<> op_, sizes_t strides_)

		nua_uninit(std::array<std::size_t, 2> s, bool forward) : sizes_(s), strides_{}, op_(), identity_(0) {
			data_.assign(init_strides(forward), identity_);
		}
	};

	struct nua_no_assign {
		ATL_PROBE_BODY([[no_unique_address]] std::plus<> op_, sizes_t strides_{})

		nua_no_assign(std::array<std::size_t, 2> s, bool forward) : sizes_(s), op_(), identity_(0) {
			(void)init_strides(forward);
		}
	};

#undef ATL_PROBE_BODY

	template<typename Replica>
	void show(char const *what) {
		std::array<std::size_t, 2> const s{3, 4};
		for (bool const forward : {false, true}) {
			Replica r(s, forward);
			std::printf("probe %s %s: sizes=%zu,%zu strides=%zu,%zu data=%zu sizeof=%zu\n", what,
						forward ? "forward" : "backward", r.sizes_[0], r.sizes_[1], r.strides_[0], r.strides_[1],
						r.data_.size(), sizeof(Replica));
			std::fflush(stdout);
		}
	}
}

TEST_CASE("msvc probe fenwick layout", "[msvc_probe]") {
	msvc_probe::show<msvc_probe::nua_init>("nua_init");
	msvc_probe::show<msvc_probe::plain_init>("plain_init");
	msvc_probe::show<msvc_probe::nua_uninit>("nua_uninit");
	msvc_probe::show<msvc_probe::nua_no_assign>("nua_no_assign");
	SUCCEED();
}
