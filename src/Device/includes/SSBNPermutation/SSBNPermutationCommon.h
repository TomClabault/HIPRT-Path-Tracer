/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_SSBN_PERMUTATION_SSBNPERMUTATIONCOMMON_H
#define DEVICE_INCLUDES_SSBN_PERMUTATION_SSBNPERMUTATIONCOMMON_H

#include "Device/includes/FixIntellisense.h"

HIPRT_DEVICE static void get_blue_noise_texture_offset(unsigned int blue_noise_texture_width,
													   unsigned int blue_noise_texture_height,
													   unsigned int zero_based_frame_number,
													   int& out_offset_x,
													   int& out_offset_y)
{
	constexpr float golden_ratio_2d = 1.3247179572f;
	out_offset_x					= (int)(1.0f / golden_ratio_2d * (float)blue_noise_texture_width * zero_based_frame_number);
	out_offset_y					= (int)(1.0f / (golden_ratio_2d * golden_ratio_2d) * (float)blue_noise_texture_height * zero_based_frame_number);
}

#endif
