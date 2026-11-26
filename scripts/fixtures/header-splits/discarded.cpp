template<bool IsUsed>
void route() {
	if constexpr (IsUsed) {
		unused_path();
	}
}

template<bool IsUsed>
void qualified_route() {
	if constexpr (IsUsed) {
		top_clusters::sum<int>{};
	}
}
