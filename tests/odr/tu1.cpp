// One-definition-rule guard, translation unit 1 of 2 (see cmake/Headers.cmake).
//
// This TU and tu2.cpp both include every public atl header and are linked into
// one binary. A non-inline function (or other externally-linked definition)
// defined at namespace scope in any header would appear in both objects and
// fail to link with a "multiple definition" error — catching the bug that
// would otherwise only surface in a consumer that includes a header twice.
#include <all_headers.hpp>

int odr_probe_tu1() { return 0; }
