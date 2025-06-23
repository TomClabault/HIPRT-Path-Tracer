/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_GRID_FILL_SURFACE_H
#define DEVICE_INCLUDES_REGIR_GRID_FILL_SURFACE_H

struct ReGIRGridFillSurface
{
	int cell_primitive_index = -1;
	float3 cell_point = make_float3(0.0f, 0.0f, 0.0f);
	float3 cell_normal = make_float3(0.0f, 0.0f, 0.0f);
	float cell_roughness = -1.0f;
	float cell_metallic = -1.0f;
	float cell_specular = -1.0f;
};

#endif
