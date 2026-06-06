/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_SPMIS_SPMIS_UTILS_H
#define DEVICE_INCLUDES_RESTIR_SPMIS_SPMIS_UTILS_H

#include "Device/includes/HashGridHash.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRSettingsHelper.h"

HIPRT_DEVICE unsigned int ReSTIR_spmis_hash(
	ReSTIRCommonSPMISSettings& spmis_settings, int pixel_x, int pixel_y, float3_t shading_point, float3_t surface_normal, unsigned int& out_checksum)
{
	return screen_space_gbuffer_hash(pixel_x, pixel_y, spmis_settings.tile_size, shading_point, surface_normal, 0.2f, &out_checksum);
}

template <int ReSTIRVariant>
HIPRT_DEVICE void ReSTIR_spmis_insert_pixel_hash(HIPRTRenderData& render_data, int pixel_x, int pixel_y, float3_t shading_point, float3_t surface_normal)
{
	ReSTIRCommonSPMISSettings& spmis_settings = ReSTIRSettingsHelper::get_restir_spmis_settings<ReSTIRVariant>(render_data);

	unsigned int pixel_index		= pixel_x + pixel_y * render_data.render_settings.render_resolution.x;
	unsigned int current_hash_value = spmis_settings.all_pixel_hashes[pixel_index];

	unsigned int total_num_cells = spmis_settings.pixel_hashes_count;

	unsigned int checksum;
	unsigned int hash_cell_index = ReSTIR_spmis_hash(spmis_settings, pixel_x, pixel_y, shading_point, surface_normal, checksum) % total_num_cells;
	if (!HashGrid::resolve_collision<16, true>(spmis_settings.all_pixel_hashes_checksums,
											   render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y,
											   hash_cell_index, checksum))
	{
		spmis_settings.all_pixel_hashes[pixel_index] = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;

		return;
	}

	spmis_settings.all_pixel_hashes[pixel_index] = hash_cell_index;
}

template <int ReSTIRVariant>
HIPRT_DEVICE void ReSTIR_spmis_update_cell_importance(
	HIPRTRenderData& render_data, int pixel_x, int pixel_y, unsigned int cell_index, float pixel_path_importance)
{
	ReSTIRCommonSPMISSettings& spmis_settings = ReSTIRSettingsHelper::get_restir_spmis_settings<ReSTIRVariant>(render_data);

	if (cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX && pixel_path_importance > 0.0f)
	{
		unsigned int pixel_index = pixel_x + pixel_y * render_data.render_settings.render_resolution.x;

		// That's one more valid pixel to reuse from in this cell
		hippt::atomic_fetch_add(&spmis_settings.cell_counters[cell_index], 1u);

		spmis_settings.important_pixel_indices_sorted[pixel_index] = pixel_index;
		spmis_settings.important_pixel_hashes[pixel_index]		   = cell_index;
	}

	if (cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
	{
		// We also update the confidence sum of the cell which is used for the spatial reuse
		// Each reservoir inserted into the cell is 1 confidence
		hippt::atomic_fetch_add(&spmis_settings.cell_confidence_sums[cell_index], 1u);
	}
}

#endif
