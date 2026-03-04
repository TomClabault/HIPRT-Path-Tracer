/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_GRID_FILL_SURFACE_H
#define DEVICE_INCLUDES_REGIR_GRID_FILL_SURFACE_H

struct ReGIRGridFillSurface
{
	int cell_primitive_index = -1;
	float3_t cell_point		 = make_float3(0.0f, 0.0f, 0.0f);
	float3_t cell_normal	 = make_float3(0.0f, 0.0f, 0.0f);
	float cell_roughness	 = -1.0f;
	float cell_metallic		 = -1.0f;
	float cell_specular		 = -1.0f;
};

HIPRT_DEVICE float3_t ReGIR_get_cell_world_normal(const HIPRTRenderData& render_data, int hash_grid_cell_index, bool primary_hit)
{
	return render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).world_normals[hash_grid_cell_index].unpack();
}

HIPRT_DEVICE float3_t ReGIR_get_cell_world_point(const HIPRTRenderData& render_data, int hash_grid_cell_index, bool primary_hit)
{
	return render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).world_points[hash_grid_cell_index];
}

HIPRT_DEVICE int ReGIR_get_cell_primitive_index(const HIPRTRenderData& render_data, int hash_grid_cell_index, bool primary_hit)
{
	return render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).hit_primitive[hash_grid_cell_index];
}

HIPRT_DEVICE float ReGIR_get_cell_roughness(const HIPRTRenderData& render_data, int hash_grid_cell_index, bool primary_hit)
{
	// / 255.0f to convert from uchar [0, 255] to float [0, 1]
	return render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).roughness[hash_grid_cell_index] / 255.0f;
}

HIPRT_DEVICE float ReGIR_get_cell_metallic(const HIPRTRenderData& render_data, int hash_grid_cell_index, bool primary_hit)
{
	// / 255.0f to convert from uchar [0, 255] to float [0, 1]
	return render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).metallic[hash_grid_cell_index] / 255.0f;
}

HIPRT_DEVICE float ReGIR_get_cell_specular(const HIPRTRenderData& render_data, int hash_grid_cell_index, bool primary_hit)
{
	// / 255.0f to convert from uchar [0, 255] to float [0, 1]
	return render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).specular[hash_grid_cell_index] / 255.0f;
}

HIPRT_DEVICE ReGIRGridFillSurface ReGIR_get_cell_surface(const HIPRTRenderData& render_data, int hash_grid_cell_index, bool primary_hit)
{
	int cell_primitive_index = ReGIR_get_cell_primitive_index(render_data, hash_grid_cell_index, primary_hit);
	float3_t cell_point		 = ReGIR_get_cell_world_point(render_data, hash_grid_cell_index, primary_hit);
	float3_t cell_normal	 = ReGIR_get_cell_world_normal(render_data, hash_grid_cell_index, primary_hit);
	float cell_roughness	 = ReGIR_get_cell_roughness(render_data, hash_grid_cell_index, primary_hit);
	float cell_metallic		 = ReGIR_get_cell_metallic(render_data, hash_grid_cell_index, primary_hit);
	float cell_specular		 = ReGIR_get_cell_specular(render_data, hash_grid_cell_index, primary_hit);

	ReGIRGridFillSurface surface;
	surface.cell_primitive_index = cell_primitive_index;
	surface.cell_point			 = cell_point;
	surface.cell_normal			 = cell_normal;
	surface.cell_roughness		 = cell_roughness;
	surface.cell_metallic		 = cell_metallic;
	surface.cell_specular		 = cell_specular;

	return surface;
}

#endif
