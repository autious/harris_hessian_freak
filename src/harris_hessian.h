#pragma once

void harris_hessian_init();
void harris_hessian_detection( uint8_t *rgba_data, int width, int height );
void harris_hessian_close();
