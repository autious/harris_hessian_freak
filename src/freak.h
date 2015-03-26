#pragma once

#include <stdint.h>
#include <stdlib.h>

typedef uint8_t word_t;

struct keyPoint_t {
	int x; // pixel coordinates
	int y;
	float size; // gaussian sigma
}; typedef struct keyPoint_t keyPoint;

struct descriptor_t {
	word_t* data;
	uint32_t x;
	uint32_t y;
}; typedef struct descriptor_t descriptor;

void freak_buildPattern();
descriptor* freak_compute(const double* src, size_t width, size_t height, keyPoint* keyPoints, int kpCount, int* descriptorCount);
double freak_meanIntensity(const double* src, size_t width, size_t height, const double* integral, const float kp_x, const float kp_y, const uint32_t scale, const uint32_t rot, const uint32_t point);
