#ifndef PARAMETER_H
#define PARAMETER_H

/* Generic toml++ parser helpers used by the maxwell parser
 * implementation (parameter.cpp + maxwell_parameters.cpp).
 *
 * This header is an *internal* header of the parser: it pulls in
 * toml++ and exposes a namespace of typed lookup helpers built on top
 * of toml::table.  Only the two .cc files that implement and call
 * these helpers should include it.
 *
 * Translation units that consume the parsed result (driver.cpp,
 * maxwell_eqs.cpp, etc.) include the public ABI in
 * maxwell_parameters.h instead, which carries only the POD parameter
 * structs and the C-callable parse_maxwell_parameters() entry point.
 *
 * Keeping toml.hpp out of the consumer translation units avoids
 * GCC 12's if-constexpr-in-generic-lambda bug (toml.hpp:8239 under
 * C++20) and shortens their compile time substantially. */

#ifndef __cplusplus
#  error "parameter.hpp is C++ only; consumers wanting the C ABI should include maxwell_parameters.h"
#endif

#include <cstdint>
#include <string>

#include "toml.hpp"

namespace parameters {

void    reset_error();
int     error_count();
void    increment_error();

/* Integer helpers come in two widths.
 *
 * The 64-bit variants return the toml++ value verbatim — TOML integers
 * are int64_t per the spec, so this is a faithful pass-through. Use
 * these when the destination field is int64_t (e.g. global grid
 * extents).
 *
 * The 32-bit variants additionally bounds-check the toml++ value
 * against [INT_MIN, INT_MAX] before narrowing; on overflow they log
 * an error, increment parser_error, and return INVALID_INTEGER. Use
 * these whenever the destination is `int`. The bounds check is the
 * point of the 32-bit overload — without it, a TOML value above
 * INT_MAX silently sign-flips or wraps under signed two's-complement
 * narrowing, with no diagnostic. */
int64_t get_integer64_value            (const char *section, const char *element,
                                        toml::table &tbl);
int64_t get_positive_integer64_value   (const char *section, const char *element,
                                        toml::table &tbl);
int64_t get_nonnegative_integer64_value(const char *section, const char *element,
                                        toml::table &tbl);

int32_t get_integer32_value            (const char *section, const char *element,
                                        toml::table &tbl);
int32_t get_integer32_value_or_default (const char *section, const char *element,
                                        toml::table &tbl, int32_t dflt);
int32_t get_positive_integer32_value   (const char *section, const char *element,
                                        toml::table &tbl);
int32_t get_nonnegative_integer32_value(const char *section, const char *element,
                                        toml::table &tbl);

double  get_real_value               (const char *section, const char *element,
                                      toml::table &tbl);
double  get_positive_real_value      (const char *section, const char *element,
                                      toml::table &tbl);
double  get_nonnegative_real_value   (const char *section, const char *element,
                                      toml::table &tbl);
bool    get_boolean_value            (const char *section, const char *element,
                                      toml::table &tbl, bool required = true);
std::string get_string_value         (const char *section, const char *element,
                                      toml::table &tbl);

} /* namespace parameters */

#endif /* PARAMETER_H */
