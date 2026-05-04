#include "parameter.h"
#include <iostream>
#include <math.h>

#define INVALID_INTEGER ((int64_t)0xdeadbeef)
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

int64_t get_integer_value(const char *section, const char *element,
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

int64_t get_positive_integer_value(const char *section, const char *element,
                                   toml::table &tbl)
{
    int64_t res = get_integer_value(section, element, tbl);
    if (res <= 0)
    {
        std::cerr << "invalid " << section << "::" << element
                  << " value — must be positive\n";
        parser_error++;
        return INVALID_INTEGER;
    }
    return res;
}

int64_t get_nonnegative_integer_value(const char *section, const char *element,
                                      toml::table &tbl)
{
    int64_t res = get_integer_value(section, element, tbl);
    if (res < 0)
    {
        std::cerr << "invalid " << section << "::" << element
                  << " value — must be non-negative\n";
        parser_error++;
        return INVALID_INTEGER;
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
