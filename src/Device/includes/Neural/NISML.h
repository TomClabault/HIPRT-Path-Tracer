/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEURAL_NIS_ML_H
#define DEVICE_INCLUDES_NEURAL_NIS_ML_H

#include "HostDeviceCommon/Maths/VecTypes.h"

struct NISTrainingSample
{
	float3_t position;
	float3_t outgoing_direction;
	float3_t normal;

	unsigned char cluster_index = 0xFF;

	float cluster_probability			 = 0.0f;
	float conditional_light_probability	 = 0.0f;
	float point_on_light_pdf_solid_angle = 0.0f;
	float contribution_luminance		 = 0.0f;

	float sg_specular_weight = 0.0f;
	float alpha_x			 = 0.0f;
	float alpha_y			 = 0.0f;
};

#endif
