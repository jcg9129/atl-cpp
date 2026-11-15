// One-definition-rule guard, translation unit 2 of 2 (see tu1.cpp).
#include <all_headers.hpp>

int odr_probe_tu1();

int odr_probe_tu2() { return 0; }

int main() { return odr_probe_tu1() + odr_probe_tu2(); }
