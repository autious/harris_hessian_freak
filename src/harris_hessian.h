#pragma once

void harris_hessian_init( int width, int height );
void harris_hessian_detection( uint8_t *rgba_data );
void harris_hessian_close();
