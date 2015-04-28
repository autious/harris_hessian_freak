//  The following header comes from the source material from which the 
//  vast majority of our implementation is sourced. The original source can be 
//  found at: https://github.com/kikohs/freak
//  As this source is modified, the original authors hold no responsibility
//  for the functionality or quality of this source program. Furthermore, 
//  this modified version of the source retains the same license as the original
//  stated below.
//
//  Max Danielsson, Thomas Sievert 2015
//
//  Copyright (C) 2011-2012  Signal processing laboratory 2, EPFL,
//  Kirell Benzi (kirell.benzi@epfl.ch),
//  Raphael Ortiz (raphael.ortiz@a3.epfl.ch)
//  Alexandre Alahi (alexandre.alahi@epfl.ch)
//  and Pierre Vandergheynst (pierre.vandergheynst@epfl.ch)
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors "as is" and
//  any express or implied warranties, including, but not limited to, the implied
//  warranties of merchantability and fitness for a particular purpose are disclaimed.
//  In no event shall the Intel Corporation or contributors be liable for any direct,
//  indirect, incidental, special, exemplary, or consequential damages
//  (including, but not limited to, procurement of substitute goods or services;
//  loss of use, data, or profits; or business interruption) however caused
//  and on any theory of liability, whether in contract, strict liability,
//  or tort (including negligence or otherwise) arising in any way out of
//  the use of this software, even if advised of the possibility of such damage.

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
