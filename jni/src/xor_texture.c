#include "xor_texture.h"

void xor_texture_generate( uint8_t* data, int width, int height )
{
    for(int y = 0; y < height; y++) 
    for(int x = 0; x < width; x++) 
    {    
        uint8_t c = x ^ y; 
        data[x*4+y*width*4+0] = c;
        data[x*4+y*width*4+1] = c;
        data[x*4+y*width*4+2] = c;
        data[x*4+y*width*4+3] = 255;
    }
}
