#pragma once

#include <stdlib.h>

#define NELEMS(x) (sizeof(x) / sizeof(x[0]))

size_t count_base_10_digits( int number );
