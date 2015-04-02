#ifndef FREAK_H
#define FREAK_H

extern const int FREAK_NB_PAIRS;
extern const int WORD_SIZE;

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
descriptor* freak_compute(const float* src, size_t width, size_t height, keyPoint* keyPoints, int kpCount, int* descriptorCount);
float freak_meanIntensity(const float* src, size_t width, size_t height, const float* integral, const float kp_x, const float kp_y, const uint32_t scale, const uint32_t rot, const uint32_t point);

#endif
