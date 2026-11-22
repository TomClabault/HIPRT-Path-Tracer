/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_SPMIS_BUILD_NEIGHBOR_RECORDS_H
#define KERNELS_RESTIR_SPMIS_BUILD_NEIGHBOR_RECORDS_H

#include "Device/includes/BSDFs/BSDFDispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Material.h"
#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "Device/includes/ReSTIR/PT/SpatialSPMISUnionDomain.h"

#include "HostDeviceCommon/RenderData.h"

/**
 * Neighbor records of the SPMIS union domain reuse (ReSTIRPTSPMISUnionDomainSettings::neighbor_records, layout documented above
 * spmis_neighbor_record_write_normal() in SpatialSPMISUnionDomain.h).
 *
 * One thread per pixel: every pixel with a non-zero spatial reuse input reservoir writes its record at its slot in its SPMIS cell (cell_offsets[cell] +
 * index_in_cell, the same slot as in pixel_indices_sorted). The records let the neighbor selection of every center pixel evaluate the visibility-free
 * shifted integrand of all the non-zero neighbors of its domain in a few FLOPs (spmis_neighbor_record_target()).
 *
 * Must run after the cells of the frame are counted (cell confidence sums, index in cell) and their offsets computed
 */
#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char RESTIR_PT_SPMIS_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_PT_SPMIS_BuildNeighborRecords()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PT_SPMIS_BuildNeighborRecords(HIPRTRenderData render_data, unsigned int pixel_index)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(RESTIR_PT_SPMIS_RENDER_DATA);

	unsigned int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	unsigned int pixel_count = render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y;
	if (pixel_index >= pixel_count)
		return;

	const ReSTIRPTSPMISSettings& spmis_settings = render_data.render_settings.restir_pt_settings.spmis_settings;

	unsigned int cell_index = spmis_settings.all_pixel_hashes[pixel_index];
	if (cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
		// No cell, the pixel is never a neighbor
		return;

	const ReSTIRPTReservoir& reservoir = render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs[pixel_index];
	if (reservoir.UCW <= 0.0f)
		// Only the non-zero reservoirs are selected as neighbors and have a record
		return;

	float3_t rc_vertex_normal = reservoir.sample.rc_vertex_geometric_normal.unpack();
	float3_t shading_point	  = render_data.g_buffer.primary_hit_position[pixel_index];

	// Radiance leaving the reconnection vertex toward the shading point of this pixel and inverse of the solid angle to area conversion of this pixel
	float outgoing_radiance_luminance;
	float inverse_area_to_solid_angle = 1.0f;

	if (reservoir.sample.is_envmap_path())
		outgoing_radiance_luminance = reservoir.sample.rc_vertex_incident_radiance.luminance();
	else
	{
		float3_t to_shading_point	= shading_point - reservoir.sample.rc_vertex;
		float distance_2			= hippt::dot(to_shading_point, to_shading_point);
		float cosine_at_rc_vertex	= hippt::abs(hippt::dot(rc_vertex_normal, to_shading_point)) / hippt::sqrt(distance_2);
		inverse_area_to_solid_angle = cosine_at_rc_vertex > 0.0f ? distance_2 / cosine_at_rc_vertex : 0.0f;

		if (reservoir.sample.di_sample)
			outgoing_radiance_luminance = reservoir.sample.rc_vertex_incident_radiance.luminance();
		else
		{
			// BSDF * cosine at the reconnection vertex toward the shading point of this pixel. Exact for any center pixel that the sample is shifted to if
			// the material of the reconnection vertex is Lambertian (the throughput then does not depend on the outgoing direction)
			Xorshift32Generator random_number_generator(wang_hash(pixel_index + 1u));

			int rc_vertex_material_index							= render_data.buffers.material_indices[reservoir.sample.rc_vertex_primitive_index];
			DeviceUnpackedPrincipledFullMaterial rc_vertex_material = get_intersection_material(
				render_data, rc_vertex_material_index, make_float2(reservoir.sample.rc_vertex_texcoords_u, reservoir.sample.rc_vertex_texcoords_v));

			RayVolumeState ray_volume_state;
			BSDFIncidentLightInfo incident_light_info_at_sample_point = reservoir.sample.incident_light_info_at_sample_point;
			float3_t view_direction									  = hippt::normalize(to_shading_point);
			float3_t rc_vertex_shading_normal						  = reservoir.sample.rc_vertex_shading_normal.unpack();
			BSDFContext rc_vertex_context(view_direction, rc_vertex_shading_normal, rc_vertex_normal, reservoir.sample.rc_vertex_incident_light_direction,
										  incident_light_info_at_sample_point, ray_volume_state, false, rc_vertex_material, 0.0f);

			float bsdf_pdf;
			ColorRGB32F rc_vertex_throughput = bsdf_dispatcher_eval(render_data, rc_vertex_context, bsdf_pdf, random_number_generator) *
											   hippt::abs(hippt::dot(reservoir.sample.rc_vertex_incident_light_direction, rc_vertex_shading_normal));
			outgoing_radiance_luminance = (rc_vertex_throughput * reservoir.sample.rc_vertex_incident_radiance).luminance();
		}
	}

	// The neighbor selection scales this confidence sum to the one of the whole domain. No non-canonical confidence scaling: the union domain reuse
	// doesn't use it
	float cell_confidence_sum = static_cast<float>(spmis_settings.cell_confidence_sums[cell_index]);
	float qk				  = outgoing_radiance_luminance * inverse_area_to_solid_angle;

	unsigned int record_slot = spmis_settings.cell_offsets[cell_index] + spmis_settings.all_pixels_index_in_cell[pixel_index];
	float* record			 = spmis_settings.union_domain.neighbor_records + record_slot * RESTIR_PT_SPMIS_UNION_DOMAIN_NEIGHBOR_RECORD_FLOAT_COUNT;

	record[0] = reservoir.sample.rc_vertex.x;
	record[1] = reservoir.sample.rc_vertex.y;
	record[2] = reservoir.sample.rc_vertex.z;
	// The padding byte of the packed normal marks an envmap path, which has no normal
	record[3] = spmis_neighbor_record_write_normal(reservoir.sample.rc_vertex_geometric_normal, reservoir.sample.is_envmap_path());
	record[4] = 0.0f;
	// Negative for DI samples, see the backfacing light test of spmis_neighbor_record_target()
	record[5] = !reservoir.sample.is_envmap_path() && reservoir.sample.di_sample ? -qk : qk;
	// Same importance as in the CDF of the cell (ReSTIR_PT_SPMIS_BuildCDFs)
	record[6] = reservoir.confidence * reservoir.UCW * reservoir.sample.target_function;
	record[7] = cell_confidence_sum * reservoir.sample.target_function;
}

#endif // #ifndef KERNELS_RESTIR_SPMIS_BUILD_NEIGHBOR_RECORDS_H
