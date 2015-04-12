#ifndef FREAK_H
#define FREAK_H

#include <stdint.h>
#include <stdlib.h>

#include <harris_hessian_freak.h>

#include "opencl_config.h"

extern const int FREAK_NB_PAIRS;
extern const int WORD_SIZE;


void freak_buildPattern();
descriptor* freak_compute(const hh_float* src, size_t width, size_t height, keyPoint* keyPoints, int kpCount, int* descriptorCount);
float freak_meanIntensity(const hh_float* src, size_t width, size_t height, const float* integral, const float kp_x, const float kp_y, const uint32_t scale, const uint32_t rot, const uint32_t point);

#endif
