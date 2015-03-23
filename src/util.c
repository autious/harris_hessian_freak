#include "util.h"

size_t count_base_10_digits( int number )
{
    size_t count = 0;

    if( number < 0 ) //space for sign
        count++;

    if( number == 0 )
        count++;

    while( number )
    {
        number /= 10;
        count++;
    } 

    return count;
}
