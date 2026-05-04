#include "parameter.hpp"
#include <climits>
#include <iostream>
#include <math.h>

/* The sentinel must fit in BOTH int32_t and int64_t so the 32- and
 * 64-bit error returns share one value.  static_cast<int32_t>(0xdeadbeef)
 * sign-extends to -559038737 in int32_t, and to the same -559038737 when
 * subsequently widened to int64_t.  Any caller that wants to detect a
 * parser failure should check parameters::error_count() rather than
 * comparing return values; the sentinel is purely a defensive default. */
#define INVALID_INTEGER (static_cast<int64_t>(static_cast<int32_t>(0xdeadbeef)))
#define INVALID_DOUBLE  nan("")

/* File-local error counter.  Exposed to callers via parameters::reset_error()
 * and parameters::error_count() without making the variable itself public. */
static int parser_error = 0;

namespace parameters {

void reset_error()     { parser_error = 0; }
int  error_count()     { return parser_error; }
void increment_error() { parser_error++; }

/* Build a dotted TOML path "section.element" so toml++'s at_path() can
 * descend into nested sub-tables — e.g. section = "source.plane_wave",
 * element = "ax" resolves to the node at source.plane_wave.ax. */
static inline std::string path_of(const char *section, const char *element)
{
    return std::string(section) + "." + element;
}

int64_t get_integer64_value(const char *section, const char *element,
                            toml::table &tbl)
{
    auto entry = tbl.at_path(path_of(section, element));
    if (entry.is_integer())
        return static_cast<int64_t>(*entry.as_integer());
    std::cerr << "invalid " << section << "::" << element << " value\n"
              << "Expected an integer, but didn't find an int\n";
    parser_error++;
    return INVALID_INTEGER;
}

int32_t get_integer32_value(const char *section, const char *element,
                            toml::table &tbl)
{
    int64_t res = get_integer64_value(section, element, tbl);
    if (res > INT_MAX || res < INT_MIN)
    {
        std::cerr << "invalid " << section << "::" << element
                  << " value — " << res << " is outside the 32-bit int range "
                  << "[" << INT_MIN << ", " << INT_MAX << "]\n";
        parser_error++;
        return static_cast<int32_t>(INVALID_INTEGER);
    }
    return static_cast<int32_t>(res);
}

double get_real_value(const char *section, const char *element,
                      toml::table &tbl)
{
    auto entry = tbl.at_path(path_of(section, element));
    if (entry.is_floating_point())
        return static_cast<double>(*entry.as_floating_point());
    if (entry.is_integer())
        return static_cast<double>(static_cast<int64_t>(*entry.as_integer()));
    std::cerr << "invalid " << section << "::" << element << " value\n"
              << "Expected a floating point or integer number, but didn't find one\n";
    parser_error++;
    return INVALID_DOUBLE;
}

double get_positive_real_value(const char *section, const char *element,
                               toml::table &tbl)
{
    double res = get_real_value(section, element, tbl);
    if (res <= 0.0)
    {
        std::cerr << "invalid " << section << "::" << element
                  << " value — must be positive\n";
        parser_error++;
        return INVALID_DOUBLE;
    }
    return res;
}

double get_nonnegative_real_value(const char *section, const char *element,
                                  toml::table &tbl)
{
    double res = get_real_value(section, element, tbl);
    if (res < 0.0)
    {
        std::cerr << "invalid " << section << "::" << element
                  << " value — must be non-negative\n";
        parser_error++;
        return INVALID_DOUBLE;
    }
    return res;
}

/* See the 32-bit variants below for the rationale on the err_before
 * guard: if the inner get_integer64_value raised an error (missing
 * key or wrong type), the sentinel return is negative and would
 * otherwise produce a spurious second "must be positive" error. */
int64_t get_positive_integer64_value(const char *section, const char *element,
                                     toml::table &tbl)
{
    const int err_before = parser_error;
    int64_t res = get_integer64_value(section, element, tbl);
    if (parser_error > err_before) return res;
    if (res <= 0)
    {
        std::cerr << "invalid " << section << "::" << element
                  << " value — must be positive\n";
        parser_error++;
        return INVALID_INTEGER;
    }
    return res;
}

int64_t get_nonnegative_integer64_value(const char *section, const char *element,
                                        toml::table &tbl)
{
    const int err_before = parser_error;
    int64_t res = get_integer64_value(section, element, tbl);
    if (parser_error > err_before) return res;
    if (res < 0)
    {
        std::cerr << "invalid " << section << "::" << element
                  << " value — must be non-negative\n";
        parser_error++;
        return INVALID_INTEGER;
    }
    return res;
}

/* The 32-bit positive/nonnegative variants are layered on top of
 * get_integer32_value: the [INT_MIN, INT_MAX] bounds check happens
 * first, so a value above INT_MAX is reported as a range error rather
 * than passing the positivity check and then silently truncating.
 *
 * If get_integer32_value itself raised an error (missing key, wrong
 * type, or out-of-range), the sign/non-negativity check is skipped --
 * the sentinel return is negative, which would otherwise produce a
 * confusing second "must be positive" error on top of the real one. */
int32_t get_positive_integer32_value(const char *section, const char *element,
                                     toml::table &tbl)
{
    const int err_before = parser_error;
    int32_t res = get_integer32_value(section, element, tbl);
    if (parser_error > err_before) return res;
    if (res <= 0)
    {
        std::cerr << "invalid " << section << "::" << element
                  << " value — must be positive\n";
        parser_error++;
        return static_cast<int32_t>(INVALID_INTEGER);
    }
    return res;
}

int32_t get_nonnegative_integer32_value(const char *section, const char *element,
                                        toml::table &tbl)
{
    const int err_before = parser_error;
    int32_t res = get_integer32_value(section, element, tbl);
    if (parser_error > err_before) return res;
    if (res < 0)
    {
        std::cerr << "invalid " << section << "::" << element
                  << " value — must be non-negative\n";
        parser_error++;
        return static_cast<int32_t>(INVALID_INTEGER);
    }
    return res;
}

bool get_boolean_value(const char *section, const char *element,
                       toml::table &tbl, bool required)
{
    auto entry = tbl.at_path(path_of(section, element));
    if (entry.is_boolean())
        return static_cast<bool>(*entry.as_boolean());
    if (required)
    {
        std::cerr << "invalid " << section << "::" << element << " value\n"
                  << "Expected a boolean, but didn't find a boolean\n";
        parser_error++;
    }
    return false;
}

std::string get_string_value(const char *section, const char *element,
                             toml::table &tbl)
{
    auto entry = tbl.at_path(path_of(section, element));
    if (entry.is_string())
        return std::string(*entry.as_string());
    std::cerr << "invalid " << section << "::" << element << " value\n"
              << "Expected a string, but didn't find one\n";
    parser_error++;
    return "";
}

} /* namespace parameters */
