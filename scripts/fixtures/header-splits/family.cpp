template<class T>
struct adaptive_incircle;

template<class T>
struct adaptive_incircle<T *> {};

template<class T>
struct dual_support_view {
	dual_support_view(T);
};
template<class T>
dual_support_view(T) -> dual_support_view<T>;
