/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_REGIR_LIGHT_DISTRIBUTIONS_GRID_FILL_H
#define DEVICE_INCLUDES_RESTIR_REGIR_LIGHT_DISTRIBUTIONS_GRID_FILL_H

#define REGIR_NEEDS_LIGHT_SAMPLE_FALLBACK -42

#include "Device/includes/LightSampling/EmissiveMeshAliasTableDevice.h"
#include "Device/includes/LightSampling/PDFTriangles.h"
#include "Device/includes/LightSampling/TriangleEmissiveSampling.h"
#include "Device/includes/ReSTIR/ReGIR/TargetFunction.h"
#include "Device/includes/TriangleLoadUtils.h"

HIPRT_DEVICE LightSamplePointArray<DirectLightSampleCount<ReGIR_GridFillCellDistributionsCanonicalSamplingTechnique>()> grid_fill_cell_light_distributions_canonical_sample(
	const HIPRTRenderData& render_data, const ReGIRGridFillSurface& surface, float3 view_direction, Xorshift32Generator& rng)
{
	RayPayload dummy_ray_payload;
	dummy_ray_payload.material.roughness = surface.cell_roughness;
	dummy_ray_payload.material.metallic = surface.cell_metallic;
	dummy_ray_payload.material.specular = surface.cell_specular;

	return sample_one_point_on_light<ReGIR_GridFillCellDistributionsCanonicalSamplingTechnique>(render_data,
		surface.cell_point, view_direction, surface.cell_normal, surface.cell_normal,
		surface.cell_primitive_index, dummy_ray_payload, rng);
}

HIPRT_DEVICE LightSamplePointInformation sample_one_emissive_triangle_with_cell_light_distribution(const HIPRTRenderData& render_data,
	float3 shading_point, float3 view_direction, float3 shading_normal,
	const DeviceUnpackedEffectiveMaterial& material,
	unsigned int hash_grid_cell_index, bool primary_hit, Xorshift32Generator& rng)
{
	const ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

	CDFDeviceU16 cell_light_distribution = regir_settings.get_cell_light_distributions(hash_grid_cell_index, primary_hit);
	if (cell_light_distribution.cdf_u16 == nullptr)
	{
		// No light distribution available for that cell. This can happen if new cells have been discovered
		// by rays bouncing around but we haven't recomputed light distributions yet

		LightSamplePointInformation fallback_needed;
		fallback_needed.emissive_triangle_global_index = REGIR_NEEDS_LIGHT_SAMPLE_FALLBACK;

		return fallback_needed;
	}
	int index_in_distribution = cell_light_distribution.sample(rng);

	unsigned int emissive_mesh_index = render_data.render_settings.regir_settings.get_cell_distributions_soa(primary_hit).get_emissive_mesh_index(hash_grid_cell_index, index_in_distribution);
	float mesh_PDF = render_data.render_settings.regir_settings.get_cell_distributions_soa(primary_hit).get_PDF(hash_grid_cell_index, index_in_distribution);
	if (mesh_PDF == 0.0f)
		// No valid mesh for this cell, early exit by returning
		// an empty sample
		return LightSamplePointInformation();

	EmissiveMeshAliasTableDevice mesh_alias_table = render_data.buffers.emissive_meshes_data.get_emissive_mesh_alias_table(emissive_mesh_index);

	// Now that we have importance sampled a mesh, we're importance sampling a triangle
	// on that mesh
	float triangle_PDF;
	int emissive_triangle_global_index = mesh_alias_table.sample_one_triangle_power(rng, triangle_PDF);

	LightSamplePointInformation light_sample = sample_point_on_light_and_fill_light_sample_information(render_data,
		shading_point, view_direction, shading_normal,
		material,
		emissive_triangle_global_index, rng);
	if (light_sample.emissive_triangle_global_index == -1)
		// Probably a degenerate triangle
		return LightSamplePointInformation();

	// Area measure PDF already contains the PDF for sampling the point *on the triangle*.
	// We need to add (multiply) the PDF of sampling the triangle itself within the sampled mesh
	light_sample.area_measure_pdf *= mesh_PDF * triangle_PDF;

	sanity_check<true>(render_data, ColorRGB32F(1.0f / light_sample.area_measure_pdf), -1, -1);

	return light_sample;
}

HIPRT_DEVICE float get_cell_distribution_PDF_of_light_sample(const HIPRTRenderData& render_data, unsigned int hash_grid_cell_index, bool primary_hit, float light_area, ColorRGB32F light_emission, unsigned int mesh_index)
{
	const ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

	CDFDeviceU16 cell_light_distribution = regir_settings.get_cell_light_distributions(hash_grid_cell_index, primary_hit);

	float mesh_sampling_PDF = 0.0f;
	// TODO absolutely need to replace that with a perfect hash table (or any fast membership data structure)
	// for performance instead of brute forcing
	for (int i = 0; i < cell_light_distribution.size; i++)
	{
		if (regir_settings.get_cell_distributions_soa(primary_hit).get_emissive_mesh_index(hash_grid_cell_index, i) == mesh_index)
		{
			mesh_sampling_PDF = render_data.render_settings.regir_settings.get_cell_distributions_soa(primary_hit).get_PDF(hash_grid_cell_index, i);

			break;
		}
	}

	float triangle_within_mesh_sampling_PDF = render_data.buffers.emissive_meshes_data.get_power_sampled_triangle_PDF_in_mesh(mesh_index, light_area, light_emission);
	float point_on_triangle_PDF = 1.0f / light_area;

	return mesh_sampling_PDF * triangle_within_mesh_sampling_PDF * point_on_triangle_PDF;
}

HIPRT_DEVICE float get_cell_distribution_PDF_of_light_sample(const HIPRTRenderData& render_data, unsigned int hash_grid_cell_index, bool primary_hit, const LightSamplePointInformation& light_sample, unsigned int mesh_index)
{
	return get_cell_distribution_PDF_of_light_sample(render_data, hash_grid_cell_index, primary_hit, hippt::length(triangle_load_normal_not_normalized(render_data, light_sample.emissive_triangle_global_index) * 0.5f), light_sample.emission, mesh_index);
}

#endif
