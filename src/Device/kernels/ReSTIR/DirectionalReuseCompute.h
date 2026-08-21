/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_DIRECTIONAL_REUSE_COMPUTE_H
#define KERNELS_RESTIR_DIRECTIONAL_REUSE_COMPUTE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/ReSTIR/DI_GI/UtilsSpatial.h"
#include "Device/includes/ReSTIR/NeighborSimilarity.h"

#include "HostDeviceCommon/KernelOptions/ReSTIRCommonOptions.h"
#include "HostDeviceCommon/RenderData.h"

#define NB_RADIUS			  32
#define NB_SAMPLES_PER_RADIUS 64

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char RESTIR_DIRECTIONAL_REUSE_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void)
__launch_bounds__(64) ReSTIR_Directional_Reuse_Compute(unsigned long long int* __restrict__ out_directional_reuse_masks_buffer_ull,
													   unsigned char* __restrict__ out_adaptive_radius_buffer)
#else
template <int ReSTIRVariant>
GLOBAL_KERNEL_SIGNATURE(void)
inline ReSTIR_Directional_Reuse_Compute(HIPRTRenderData render_data,
										int x,
										int y,
										unsigned long long int* __restrict__ out_directional_reuse_masks_buffer_ull,
										unsigned char* __restrict__ out_adaptive_radius_buffer)
#endif
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(RESTIR_DIRECTIONAL_REUSE_RENDER_DATA);

	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
	if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
		return;

	uint32_t center_pixel_index = x + y * render_data.render_settings.render_resolution.x;

	if (!render_data.aux_buffers.pixel_active[center_pixel_index])
		// Pixel isn't active because of adaptive sampling or render resolution scaling
		return;

	Xorshift32Generator random_number_generator(render_data.get_updated_random_seed(center_pixel_index));

	// Clearing previous data
	out_directional_reuse_masks_buffer_ull[center_pixel_index] = 0;
	out_adaptive_radius_buffer[center_pixel_index]			   = 0;

#ifdef __KERNELCC__
	constexpr int RESTIR_VARIANT = ComputingSpatialDirectionalReuseReSTIRVariant;
#else
	constexpr int RESTIR_VARIANT = ReSTIRVariant;
#endif

	ReSTIRCommonSpatialPassSettings spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<RESTIR_VARIANT>(render_data);

	float3_t center_shading_point = render_data.g_buffer.primary_hit_position[center_pixel_index];
	float3_t center_normal		  = ReSTIRSettingsHelper::get_restir_neighbor_similarity_settings<RESTIR_VARIANT>(render_data).reject_using_geometric_normals
										? render_data.g_buffer.geometric_normals[center_pixel_index].unpack()
										: render_data.g_buffer.shading_normals[center_pixel_index].unpack();

	float best_area		  = 0.0f;
	int best_radius_index = 0;
	// Each long long int in there contains, in each bit, whether or not the direction for that radius is reusable or not
	unsigned long long int valid_samples_per_radius[NB_RADIUS] = { 0 };
	for (int radius_index = 0; radius_index < NB_RADIUS; radius_index++)
	{
		float current_radius = spatial_pass_settings.minimum_per_pixel_reuse_radius +
							   (radius_index / (float)NB_RADIUS) * (spatial_pass_settings.reuse_radius - spatial_pass_settings.minimum_per_pixel_reuse_radius);
		float current_radius_circle_area = hippt::M_Pi * current_radius * current_radius;

		// Now sampling a bunch of neighbors *on* that radius, exactly at that radius distance from the center (i.e. *not* within the disk of that radius)
		float area_at_current_radius = 0.0f;
		for (int sample_index = 0; sample_index < NB_SAMPLES_PER_RADIUS; sample_index++)
		{
			if (radius_index > 0)
				if (!(valid_samples_per_radius[radius_index - 1] & (1ull << sample_index)))
					// If this direction wasn't accepted at the previous radius
					continue;

			float theta	   = sample_index / (float)NB_SAMPLES_PER_RADIUS * hippt::M_TWO_PI;
			float x_circle = current_radius * hippt::intrin_cosf(theta);
			float y_circle = current_radius * hippt::intrin_sinf(theta);

			int2_t neighbor_offset_in_disk = make_int2(static_cast<int>(roundf(x_circle)), static_cast<int>(roundf(y_circle)));
			int2_t neighbor_pixel_coords   = make_int2(x, y) + neighbor_offset_in_disk;
			if (neighbor_pixel_coords.x < 0 || neighbor_pixel_coords.x >= render_data.render_settings.render_resolution.x || neighbor_pixel_coords.y < 0 ||
				neighbor_pixel_coords.y >= render_data.render_settings.render_resolution.y)
				// Rejecting the sample if it's outside of the viewport
				continue;

			int neighbor_index = neighbor_pixel_coords.x + neighbor_pixel_coords.y * render_data.render_settings.render_resolution.x;

			// On the CPU, it is the template argument that dictates whether this is for ReSTIR DI or GI
			if (!check_neighbor_similarity_heuristics<RESTIR_VARIANT>(render_data, neighbor_index, center_pixel_index, center_shading_point, center_normal))
				continue;

			valid_samples_per_radius[radius_index] |= (1ull << sample_index);
			area_at_current_radius += current_radius_circle_area * (1.0f / NB_SAMPLES_PER_RADIUS);
		}

		if (best_area < area_at_current_radius)
		{
			best_area		  = area_at_current_radius;
			best_radius_index = radius_index;
		}
	}

	// Computing the actual radius from the best radius index
	float best_radius = spatial_pass_settings.minimum_per_pixel_reuse_radius +
						(best_radius_index / (float)NB_RADIUS) * (spatial_pass_settings.reuse_radius - spatial_pass_settings.minimum_per_pixel_reuse_radius);
	if (best_area == 0.0f)
		best_radius = 0.0f;

	out_adaptive_radius_buffer[center_pixel_index]			   = (unsigned char)best_radius;
	out_directional_reuse_masks_buffer_ull[center_pixel_index] = valid_samples_per_radius[best_radius_index];
}

#endif
