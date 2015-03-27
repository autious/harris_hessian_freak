#include "freak.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

struct patternPoint {
	float x; // coordinates relative to keypoint center
	float y;
	float sigma; // gauss
}; typedef struct patternPoint patternPoint;

struct descriptionPair {
	uint8_t i; // index of the first point
	uint8_t j; // index of the second point
}; typedef struct descriptionPair descriptionPair;

struct orientationPair {
	uint8_t i;
	uint8_t j;
	int weight_dx;
	int weight_dy;
}; typedef struct orientationPair orientationPair;


// FREAK related constants
static const float FREAK_LOG2 = 0.693147180559945f;
#define FREAK_NB_SCALES 64
static const int FREAK_NB_ORIENTATION = 256;
#define FREAK_NB_POINTS 43
#define FREAK_NB_PAIRS 512
static const int FREAK_SMALLEST_KP_SIZE = 7;
#define FREAK_NB_ORIENTATION_PAIRS 45
static const int FREAK_NB_OCTAVES = 4;
static const float FREAK_PATTERN_SCALE = 22.0;
static const int FREAK_DEF_PAIRS[FREAK_NB_PAIRS] = { // default pairs
	404, 431, 818, 511, 181, 52, 311, 874, 774, 543, 719, 230, 417, 205, 11,
	560, 149, 265, 39, 306, 165, 857, 250, 8, 61, 15, 55, 717, 44, 412,
	592, 134, 761, 695, 660, 782, 625, 487, 549, 516, 271, 665, 762, 392, 178,
	796, 773, 31, 672, 845, 548, 794, 677, 654, 241, 831, 225, 238, 849, 83,
	691, 484, 826, 707, 122, 517, 583, 731, 328, 339, 571, 475, 394, 472, 580,
	381, 137, 93, 380, 327, 619, 729, 808, 218, 213, 459, 141, 806, 341, 95,
	382, 568, 124, 750, 193, 749, 706, 843, 79, 199, 317, 329, 768, 198, 100,
	466, 613, 78, 562, 783, 689, 136, 838, 94, 142, 164, 679, 219, 419, 366,
	418, 423, 77, 89, 523, 259, 683, 312, 555, 20, 470, 684, 123, 458, 453, 833,
	72, 113, 253, 108, 313, 25, 153, 648, 411, 607, 618, 128, 305, 232, 301, 84,
	56, 264, 371, 46, 407, 360, 38, 99, 176, 710, 114, 578, 66, 372, 653,
	129, 359, 424, 159, 821, 10, 323, 393, 5, 340, 891, 9, 790, 47, 0, 175, 346,
	236, 26, 172, 147, 574, 561, 32, 294, 429, 724, 755, 398, 787, 288, 299,
	769, 565, 767, 722, 757, 224, 465, 723, 498, 467, 235, 127, 802, 446, 233,
	544, 482, 800, 318, 16, 532, 801, 441, 554, 173, 60, 530, 713, 469, 30,
	212, 630, 899, 170, 266, 799, 88, 49, 512, 399, 23, 500, 107, 524, 90,
	194, 143, 135, 192, 206, 345, 148, 71, 119, 101, 563, 870, 158, 254, 214,
	276, 464, 332, 725, 188, 385, 24, 476, 40, 231, 620, 171, 258, 67, 109,
	844, 244, 187, 388, 701, 690, 50, 7, 850, 479, 48, 522, 22, 154, 12, 659,
	736, 655, 577, 737, 830, 811, 174, 21, 237, 335, 353, 234, 53, 270, 62,
	182, 45, 177, 245, 812, 673, 355, 556, 612, 166, 204, 54, 248, 365, 226,
	242, 452, 700, 685, 573, 14, 842, 481, 468, 781, 564, 416, 179, 405, 35,
	819, 608, 624, 367, 98, 643, 448, 2, 460, 676, 440, 240, 130, 146, 184,
	185, 430, 65, 807, 377, 82, 121, 708, 239, 310, 138, 596, 730, 575, 477,
	851, 797, 247, 27, 85, 586, 307, 779, 326, 494, 856, 324, 827, 96, 748,
	13, 397, 125, 688, 702, 92, 293, 716, 277, 140, 112, 4, 80, 855, 839, 1,
	413, 347, 584, 493, 289, 696, 19, 751, 379, 76, 73, 115, 6, 590, 183, 734,
	197, 483, 217, 344, 330, 400, 186, 243, 587, 220, 780, 200, 793, 246, 824,
	41, 735, 579, 81, 703, 322, 760, 720, 139, 480, 490, 91, 814, 813, 163,
	152, 488, 763, 263, 425, 410, 576, 120, 319, 668, 150, 160, 302, 491, 515,
	260, 145, 428, 97, 251, 395, 272, 252, 18, 106, 358, 854, 485, 144, 550,
	131, 133, 378, 68, 102, 104, 58, 361, 275, 209, 697, 582, 338, 742, 589,
	325, 408, 229, 28, 304, 191, 189, 110, 126, 486, 211, 547, 533, 70, 215,
	670, 249, 36, 581, 389, 605, 331, 518, 442, 822
};

// FREAK variables
patternPoint* patternLookup;
int patternSizes[FREAK_NB_SCALES];
descriptionPair descriptionPairs[FREAK_NB_PAIRS];
orientationPair orientationPairs[FREAK_NB_ORIENTATION_PAIRS];

void makeIntegral(const float* src, size_t width, size_t height, float* outIntegral) {
	for (size_t y = 0; y < height; ++y) {
		for (size_t x = 0; x < width; ++x) {
			float val = (x % width != 0) ? outIntegral[width * y + x - 1] : 0.0;
			outIntegral[width * y + x] = val + src[width * y + x];
		}
	}

	for (size_t x = 0; x < width; ++x) {
		for (size_t y = 0; y < height; ++y) {
			float val = (y % height != 0) ? outIntegral[width * (y - 1) + x] : 0.0;
			outIntegral[width * y + x] += val;
		}
	}
}
float integralBlur(const float* integral, size_t width, size_t height, int x, int y, float sigma) {
	int farX = (x + sigma < width) ? (int)(x + sigma) : (int)(width - 1);
	int nearX = (x - sigma >= 0) ? (int)(x - sigma) : 0;

	int farY = (y + sigma < height) ? (int)(y + sigma) : (int)(height - 1);
	int nearY = (y - sigma >= 0) ? (int)(y - sigma) : 0;

	float result = integral[width * farY + farX];
	if (nearX > 0 && nearY > 0) {
		return (result - integral[width * nearY + farX] - integral[width * farY + nearX] + integral[width * nearY + nearX]) / ((farX - nearX) * (farY - nearY));
	}
	else if (nearX > 0) {
		return result - integral[width * farY + nearX] / ((farX - nearX) * (farY - nearY));
	}
	else if (nearY > 0) {
		return result - integral[width * nearY + farX] / ((farX - nearX) * (farY - nearY));
	}
	else return result / ((farX - nearX) * (farY - nearY));
}

// Some utility functions for handling bitsets
static const int WORD_SIZE = sizeof(word_t)* 8;

static int bs_index(int b) { return b / WORD_SIZE; }
static int bs_offset(int b) { return b % WORD_SIZE; }

static void bs_set_bit(int b, word_t* words) {
	words[bs_index(b)] |= 1 << (bs_offset(b));
}
/* UNUSED
static void bs_clear_bit(int b, word_t* words) {
	words[bs_index(b)] &= ~(1 << (bs_offset(b)));
}
static uint8_t bs_get_bit(int b, const word_t* words) {
	return words[bs_index(b)] & (1 << (bs_offset(b)));
}
*/



void freak_buildPattern() {
	patternLookup = (patternPoint*)malloc(sizeof(patternPoint)* FREAK_NB_SCALES * FREAK_NB_ORIENTATION * FREAK_NB_POINTS);
	float scaleStep = pow(2.0, (float)(FREAK_NB_OCTAVES) / FREAK_NB_SCALES);
	float scalingFactor, alpha, beta, theta = 0;

	// Pattern definition, radius normalized to 1.0
	const int n[8] = { 6, 6, 6, 6, 6, 6, 6, 1 }; // Number of points on each concentric circle, outer to inner
	const float bigR = 2.0 / 3.0;
	const float smallR = 2.0 / 24.0;
	const float unitSpace = (bigR - smallR) / 21.0; // Spaces between concentric circles
	// Radii of the concentric circles, again outer to inner
	const float radius[8] = { bigR, bigR - 6 * unitSpace, bigR - 11 * unitSpace, bigR - 15 * unitSpace, bigR - 18 * unitSpace, bigR - 20 * unitSpace, smallR, 0.0 };
	// Sigmas of pattern points, proportional to radii (sigma can't be zero so last one is same as second to last)
	const float sigma[8] = { radius[0] / 2.0, radius[1] / 2.0, radius[2] / 2.0, radius[3] / 2.0, radius[4] / 2.0, radius[5] / 2.0, radius[6] / 2.0, radius[6] / 2.0 };

	// Now let's fill the lookup table
	for (size_t scaleIdx = 0; scaleIdx < FREAK_NB_SCALES; ++scaleIdx) {
		patternSizes[scaleIdx] = 0;
		scalingFactor = pow(scaleStep, scaleIdx);

		for (int orientationIdx = 0; orientationIdx < FREAK_NB_ORIENTATION; ++orientationIdx) {
			theta = (float)orientationIdx * 2 * M_PI / (float)FREAK_NB_ORIENTATION;
			int pointIdx = 0;

			for (size_t i = 0; i < 8; ++i) {
				for (int k = 0; k < n[i]; ++k) {
					beta = M_PI / n[i] * (i % 2); // Orientation offset to stagger points on each circle
					alpha = (float)k * 2 * M_PI / (float)n[i] + beta + theta;

					// Finally add stuff to the look-up table
					int INDEX = scaleIdx * FREAK_NB_ORIENTATION * FREAK_NB_POINTS + orientationIdx * FREAK_NB_POINTS + pointIdx;
					//patternPoint* point = &patternLookup[INDEX];
					patternLookup[INDEX].x = (float)(radius[i] * cos(alpha) * scalingFactor * FREAK_PATTERN_SCALE);
					patternLookup[INDEX].y = (float)(radius[i] * sin(alpha) * scalingFactor * FREAK_PATTERN_SCALE);
					patternLookup[INDEX].sigma = (float)(sigma[i] * scalingFactor * FREAK_PATTERN_SCALE);

					// Adapt sizeList
					const int sizeMax = (int)(ceil((radius[i] + sigma[i]) * scalingFactor * FREAK_PATTERN_SCALE) + 1);
					patternSizes[scaleIdx] = (int)fmax(patternSizes[scaleIdx], sizeMax);

					++pointIdx;
				}
			}
		}
	}

	// Build the list of orientation pairs (this is 1:1 copypasta)
	orientationPairs[0].i = 0; orientationPairs[0].j = 3; orientationPairs[1].i = 1; orientationPairs[1].j = 4; orientationPairs[2].i = 2; orientationPairs[2].j = 5;
	orientationPairs[3].i = 0; orientationPairs[3].j = 2; orientationPairs[4].i = 1; orientationPairs[4].j = 3; orientationPairs[5].i = 2; orientationPairs[5].j = 4;
	orientationPairs[6].i = 3; orientationPairs[6].j = 5; orientationPairs[7].i = 4; orientationPairs[7].j = 0; orientationPairs[8].i = 5; orientationPairs[8].j = 1;

	orientationPairs[9].i = 6; orientationPairs[9].j = 9; orientationPairs[10].i = 7; orientationPairs[10].j = 10; orientationPairs[11].i = 8; orientationPairs[11].j = 11;
	orientationPairs[12].i = 6; orientationPairs[12].j = 8; orientationPairs[13].i = 7; orientationPairs[13].j = 9; orientationPairs[14].i = 8; orientationPairs[14].j = 10;
	orientationPairs[15].i = 9; orientationPairs[15].j = 11; orientationPairs[16].i = 10; orientationPairs[16].j = 6; orientationPairs[17].i = 11; orientationPairs[17].j = 7;

	orientationPairs[18].i = 12; orientationPairs[18].j = 15; orientationPairs[19].i = 13; orientationPairs[19].j = 16; orientationPairs[20].i = 14; orientationPairs[20].j = 17;
	orientationPairs[21].i = 12; orientationPairs[21].j = 14; orientationPairs[22].i = 13; orientationPairs[22].j = 15; orientationPairs[23].i = 14; orientationPairs[23].j = 16;
	orientationPairs[24].i = 15; orientationPairs[24].j = 17; orientationPairs[25].i = 16; orientationPairs[25].j = 12; orientationPairs[26].i = 17; orientationPairs[26].j = 13;

	orientationPairs[27].i = 18; orientationPairs[27].j = 21; orientationPairs[28].i = 19; orientationPairs[28].j = 22; orientationPairs[29].i = 20; orientationPairs[29].j = 23;
	orientationPairs[30].i = 18; orientationPairs[30].j = 20; orientationPairs[31].i = 19; orientationPairs[31].j = 21; orientationPairs[32].i = 20; orientationPairs[32].j = 22;
	orientationPairs[33].i = 21; orientationPairs[33].j = 23; orientationPairs[34].i = 22; orientationPairs[34].j = 18; orientationPairs[35].i = 23; orientationPairs[35].j = 19;

	orientationPairs[36].i = 24; orientationPairs[36].j = 27; orientationPairs[37].i = 25; orientationPairs[37].j = 28; orientationPairs[38].i = 26; orientationPairs[38].j = 29;
	orientationPairs[39].i = 30; orientationPairs[39].j = 33; orientationPairs[40].i = 31; orientationPairs[40].j = 34; orientationPairs[41].i = 32; orientationPairs[41].j = 35;
	orientationPairs[42].i = 36; orientationPairs[42].j = 39; orientationPairs[43].i = 37; orientationPairs[43].j = 40; orientationPairs[44].i = 38; orientationPairs[44].j = 41;

	for (unsigned m = FREAK_NB_ORIENTATION_PAIRS; --m;) { // Magic for loop condition :3
		const float dx = patternLookup[orientationPairs[m].i].x - patternLookup[orientationPairs[m].j].x;
		const float dy = patternLookup[orientationPairs[m].i].y - patternLookup[orientationPairs[m].j].y;
		const float norm_sq = dx * dx + dy * dy;
		orientationPairs[m].weight_dx = (int)((dx / norm_sq) * 4096.0 + 0.5);
		orientationPairs[m].weight_dy = (int)((dy / norm_sq) * 4096.0 + 0.5);
	}

	// Build the list of description pairs
	descriptionPair allPairs[903];
	int count = 0;
	for (unsigned int i = 1; i < FREAK_NB_POINTS; ++i) {
		for (unsigned int j = 0; j < i; ++j) {
			descriptionPair* dp = &allPairs[count++]; //= { (uint8_t)i, (uint8_t)j };
			dp->i = (uint8_t)i;
			dp->j = (uint8_t)j;
		}
	}

	// This is where the real funky lookup table into lookup table magic is
	for (size_t i = 0; i < FREAK_NB_PAIRS; ++i) {
		descriptionPairs[i] = allPairs[FREAK_DEF_PAIRS[i]];
	}
}

// Returns an ALLOCATED pointer to <descriptorCount> descriptors. Note that descriptorCount is an out variable
descriptor* freak_compute(const float* src, size_t width, size_t height, keyPoint* keyPoints, int kpCount, int* descriptorCount) {
	float* integral = (float*)malloc(width * height * sizeof(float));
	makeIntegral(src, width, height, integral);
	int* kpScaleIdx = (int*)malloc(kpCount * sizeof(int));
	int* kpScaleIdx_TEMP = (int*)malloc(kpCount * sizeof(int));
	keyPoint* keyPoints_TEMP = (keyPoint*)malloc(kpCount * sizeof(keyPoint));
	const float sizeCst = FREAK_NB_SCALES / (FREAK_LOG2 * FREAK_NB_OCTAVES);
	float pointsValue[FREAK_NB_POINTS];
	int thetaIdx = 0;
	int direction0;
	int direction1;
	descriptor* descriptors;

	int tempCounter = 0;
	for (int k = 0; k < kpCount; ++k) {
		kpScaleIdx_TEMP[k] = (int)fmax((int)(log(keyPoints[k].size / FREAK_SMALLEST_KP_SIZE) * sizeCst + 0.5), 0);
		if (kpScaleIdx_TEMP[k] >= FREAK_NB_SCALES) kpScaleIdx_TEMP[k] = FREAK_NB_SCALES - 1;

		if (keyPoints[k].x <= patternSizes[kpScaleIdx_TEMP[k]] || keyPoints[k].y <= patternSizes[kpScaleIdx_TEMP[k]] ||
			keyPoints[k].x >= (int)width - patternSizes[kpScaleIdx_TEMP[k]] || keyPoints[k].y >= (int)height - patternSizes[kpScaleIdx_TEMP[k]]) {
			keyPoints[k].x = keyPoints[k].y = -1;
			keyPoints[k].size = -1.0f;
			kpScaleIdx_TEMP[k] = -1;
		}
		else {
			kpScaleIdx[tempCounter] = kpScaleIdx_TEMP[k];
			keyPoints_TEMP[tempCounter++] = keyPoints[k];
		}
	}
	kpCount = tempCounter;
	free(keyPoints);

	keyPoints = (keyPoint*)realloc(keyPoints_TEMP, kpCount * sizeof(keyPoint));
	kpScaleIdx = (int*)realloc(kpScaleIdx, kpCount * sizeof(int));

	free(kpScaleIdx_TEMP);
	//keyPoints = keyPoints_TEMP;

	*descriptorCount = kpCount;
	descriptors = (descriptor*)malloc(sizeof(descriptor)* kpCount);
	//for (int i = 0; i < kpCount; ++i) {
	//	bitset_alloc(FREAK_NB_PAIRS, &(*descriptors)[i]);
	//}

	for (int k = 0; k < kpCount; ++k) {
		// Only do orientation-invariant version of FREAK
		for (size_t i = 0; i < FREAK_NB_POINTS; ++i) {
			pointsValue[i] = freak_meanIntensity(src, width, height, integral, (float)keyPoints[k].x, (float)keyPoints[k].y, kpScaleIdx[k], 0, i);
		}

		direction0 = direction1 = 0;
		for (size_t m = 0; m < FREAK_NB_ORIENTATION_PAIRS; ++m) {
			const int delta = (int)(pointsValue[orientationPairs[m].i] - pointsValue[orientationPairs[m].j]);
			direction0 += delta * orientationPairs[m].weight_dx / 2048;
			direction1 += delta * orientationPairs[m].weight_dy / 2048;
		}

		float angle = atan2((float)direction1, (float)direction0) * 180.0 / M_PI;
		thetaIdx = (int)(FREAK_NB_ORIENTATION * angle * (1 / 360.0) + 0.5);

		if (thetaIdx < 0) thetaIdx += FREAK_NB_ORIENTATION;
		if (thetaIdx >= FREAK_NB_ORIENTATION) thetaIdx -= FREAK_NB_ORIENTATION;

		for (size_t i = 0; i < FREAK_NB_POINTS; ++i) {
			pointsValue[i] = freak_meanIntensity(src, width, height, integral, (float)keyPoints[k].x, (float)keyPoints[k].y, kpScaleIdx[k], thetaIdx, i);
		}

		descriptors[k].data = (word_t*)malloc(sizeof(word_t)* FREAK_NB_PAIRS / WORD_SIZE);
		memset(descriptors[k].data, 0, sizeof(word_t)* FREAK_NB_PAIRS / WORD_SIZE);
		descriptors[k].x = keyPoints[k].x;
		descriptors[k].y = keyPoints[k].y;

		//int cnt = 0;
		for (size_t m = FREAK_NB_PAIRS; m--;)  {
			if (pointsValue[descriptionPairs[m].i] > pointsValue[descriptionPairs[m].j]) bs_set_bit(m, descriptors[k].data);
		}
	}

	free(kpScaleIdx);
	free(integral);

	return descriptors;
}

// FREAK paper says Gaussian blur, implementation does box filter with integral image.
float freak_meanIntensity(const float* src, size_t width, size_t height, const float* integral, const float kp_x, const float kp_y, const uint32_t scale, const uint32_t rot, const uint32_t point) {
	//const patternPoint* freakPoint = &patternLookup[scale * FREAK_NB_ORIENTATION * FREAK_NB_POINTS + rot * FREAK_NB_POINTS + point];
	int INDEX = scale * FREAK_NB_ORIENTATION * FREAK_NB_POINTS + rot * FREAK_NB_POINTS + point;

	const float xf = patternLookup[INDEX].x + kp_x;
	const float yf = patternLookup[INDEX].y + kp_y;
	const int x = (int)xf;
	const int y = (int)yf;

	const float radius = patternLookup[INDEX].sigma;

	float ret_val;
	if (radius < 0.5) {
		// Simple linear interpolation
		const int r_x = (int)((xf - x) * 1024);
		const int r_y = (int)((yf - y) * 1024);
		const int r_x_1 = 1024 - r_x;
		const int r_y_1 = 1024 - r_y;
		const float* ptr = &src[y * width + x];
		// Keeping the pointer arithmetic mumbo jumbo for fun
		ret_val = r_x_1 * r_y_1 * (*ptr);
		ptr++;
		ret_val += r_x * r_y_1 * (*ptr);
		ptr += width;
		ret_val += r_x * r_y * (*ptr);
		ptr--;
		ret_val += r_x_1 * r_y * (*ptr);

		return (ret_val + 512) / 1024;
	}

	return integralBlur(integral, width, height, x, y, radius);
}
