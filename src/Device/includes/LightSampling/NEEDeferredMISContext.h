/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_NEE_DEFERRED_MIS_CONTEXT_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_NEE_DEFERRED_MIS_CONTEXT_H

#include "Device/includes/LightSampling/RIS/RISReservoir.h"
#include "Device/includes/Material.h"
#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"

static constexpr unsigned int NEE_DEFERRED_INVALID_PATH_INDEX = 0xFFFFFFFFu;
#if defined(BSDF_MODEL)
typedef LightProposalStateFor<DirectLightSamplingStrategy, TrianglePointSamplingStrategy, static_cast<BSDFModel>(BSDF_MODEL)> NEEDeferredLightProposalState;
#else
typedef LightProposalStateFor<DirectLightSamplingStrategy, TrianglePointSamplingStrategy, BSDFModel::Principled> NEEDeferredLightProposalState;
#endif

HIPRT_HOST_DEVICE inline NEEDeferredLightProposalState make_nee_deferred_light_proposal_state(const LightProposalInputs& proposal_inputs)
{
#if defined(BSDF_MODEL)
	return make_light_proposal_state<DirectLightSamplingStrategy, TrianglePointSamplingStrategy, static_cast<BSDFModel>(BSDF_MODEL)>(proposal_inputs);
#else
	return make_light_proposal_state<DirectLightSamplingStrategy, TrianglePointSamplingStrategy, BSDFModel::Principled>(proposal_inputs);
#endif
}

HIPRT_DEVICE static DeviceUnpackedPrincipledFullMaterial load_deferred_material_reference(const HIPRTRenderData& render_data,
																						  int primitive_index,
																						  float2_t texcoords,
																						  unsigned int primary_gbuffer_path_index)
{
	if (primary_gbuffer_path_index != NEE_DEFERRED_INVALID_PATH_INDEX)
		return render_data.g_buffer.materials[primary_gbuffer_path_index].unpack();

	int material_index = render_data.buffers.material_indices[primitive_index];
	return get_intersection_material(render_data, material_index, texcoords);
}

template <typename MaterialType, typename NEEContext>
HIPRT_DEVICE static void load_deferred_material_reference(const HIPRTRenderData& render_data,
														  const NEEContext& nee_deferred_MIS_context,
														  const ResolvedMaterialUserControlsCache& resolved_user_controls,
														  MaterialType& out_material)
{
	if (nee_deferred_MIS_context.last_primary_gbuffer_path_index != NEE_DEFERRED_INVALID_PATH_INDEX)
	{
		load_effective_material(render_data.g_buffer.materials[nee_deferred_MIS_context.last_primary_gbuffer_path_index], out_material);
		return;
	}

	int material_index = render_data.buffers.material_indices[nee_deferred_MIS_context.last_primitive_index];
	load_effective_material(render_data, material_index, nee_deferred_MIS_context.last_texcoords, resolved_user_controls, out_material);
}

template <int NEEEstimator, int PathIntegrator = PathSamplingStrategy>
struct NEEDeferredMISContextSpecialized
{
	BSDFIncidentLightInfo last_bsdf_incident_light_info;

	HIPRT_DEVICE void fill_last_hit_information(HitInfo& closest_hit_info,
												const float3_t& view_direction,
												const RayVolumeState& volume_state,
												int primitive_index,
												float2_t texcoords,
												unsigned int primary_gbuffer_path_index,
												const ColorRGB32F& ray_throughput)
	{
	}

	HIPRT_DEVICE void fill_last_bsdf_information(ColorRGB32F bsdf_cos_theta, float bsdf_pdf) {}

	HIPRT_DEVICE void fill_ris_reservoir(const RISReservoir& reservoir) {}

	HIPRT_DEVICE void set_last_light_proposal_state(const NEEDeferredLightProposalState& proposal_state, bool can_do_light_sampling) {}
};

template <int PathIntegrator>
struct NEEDeferredMISContextSpecialized<LSS_BSDF, PathIntegrator>
{
	float3_t last_shading_point;

	// Ray throughput before multiplication with last_bsdf_throughput
	ColorRGB32F last_ray_throughput;

	// BSDF * cos_theta
	ColorRGB32F last_bsdf_x_cos_theta;
	float last_bsdf_sample_pdf;

	HIPRT_DEVICE void fill_last_hit_information(HitInfo& closest_hit_info,
												const float3_t& view_direction,
												const RayVolumeState& volume_state,
												int primitive_index,
												float2_t texcoords,
												unsigned int primary_gbuffer_path_index,
												const ColorRGB32F& ray_throughput)
	{
		last_shading_point	= closest_hit_info.inter_point;
		last_ray_throughput = ray_throughput;
	}

	HIPRT_DEVICE void fill_last_bsdf_information(ColorRGB32F bsdf_cos_theta, float bsdf_pdf)
	{
		last_bsdf_x_cos_theta = bsdf_cos_theta;
		last_bsdf_sample_pdf  = bsdf_pdf;
	}

	HIPRT_DEVICE void set_last_light_proposal_state(const NEEDeferredLightProposalState& proposal_state, bool can_do_light_sampling) {}
};

template <int PathIntegrator>
struct NEEDeferredMISContextSpecialized<LSS_MIS_LIGHT_BSDF, PathIntegrator>
{
	float3_t last_view_direction;
	float3_t last_shading_point;
	float3_t last_shading_normal;
	int last_primitive_index = -1;
	float2_t last_texcoords;
	unsigned int last_primary_gbuffer_path_index = NEE_DEFERRED_INVALID_PATH_INDEX;

	// Ray throughput before multiplication with last_bsdf_throughput
	ColorRGB32F last_ray_throughput;

	// BSDF * cos_theta
	ColorRGB32F last_bsdf_x_cos_theta;
	float last_bsdf_sample_pdf;
	BSDFIncidentLightInfo last_bsdf_incident_light_info;
	NEEDeferredLightProposalState last_light_proposal_state;
	bool last_can_do_light_sampling;

	HIPRT_DEVICE void fill_last_hit_information(HitInfo& closest_hit_info,
												const float3_t& view_direction,
												const RayVolumeState& volume_state,
												int primitive_index,
												float2_t texcoords,
												unsigned int primary_gbuffer_path_index,
												const ColorRGB32F& ray_throughput)
	{
		last_view_direction				= view_direction;
		last_shading_point				= closest_hit_info.inter_point;
		last_shading_normal				= closest_hit_info.shading_normal;
		last_primitive_index			= primitive_index;
		last_texcoords					= texcoords;
		last_primary_gbuffer_path_index = primary_gbuffer_path_index;
		last_ray_throughput				= ray_throughput;
	}

	HIPRT_DEVICE void fill_last_bsdf_information(ColorRGB32F bsdf_cos_theta, float bsdf_pdf)
	{
		last_bsdf_x_cos_theta = bsdf_cos_theta;
		last_bsdf_sample_pdf  = bsdf_pdf;
	}

	HIPRT_DEVICE void fill_ris_reservoir(const RISReservoir& reservoir) {}

	HIPRT_DEVICE void set_last_light_proposal_state(const NEEDeferredLightProposalState& proposal_state, bool can_do_light_sampling)
	{
		last_light_proposal_state  = proposal_state;
		last_can_do_light_sampling = can_do_light_sampling;
	}

	HIPRT_DEVICE const NEEDeferredLightProposalState& get_last_light_proposal_state() const
	{
		return last_light_proposal_state;
	}

	HIPRT_DEVICE DeviceUnpackedPrincipledFullMaterial get_last_material(const HIPRTRenderData& render_data) const
	{
		return load_deferred_material_reference(render_data, last_primitive_index, last_texcoords, last_primary_gbuffer_path_index);
	}
};

template <int PathIntegrator>
struct NEEDeferredMISContextSpecialized<LSS_LEARNING_TO_CLUSTER_MIS, PathIntegrator> : NEEDeferredMISContextSpecialized<LSS_MIS_LIGHT_BSDF, PathIntegrator>
{
	// The previous surface's mesh selects the learned cut, independently of the light hit by the BSDF ray.
	HIPRT_DEVICE void fill_last_hit_information(HitInfo& closest_hit_info,
												const float3_t& view_direction,
												const RayVolumeState& volume_state,
												int primitive_index,
												float2_t texcoords,
												unsigned int primary_gbuffer_path_index,
												const ColorRGB32F& ray_throughput)
	{
		NEEDeferredMISContextSpecialized<LSS_MIS_LIGHT_BSDF, PathIntegrator>::fill_last_hit_information(
			closest_hit_info, view_direction, volume_state, primitive_index, texcoords, primary_gbuffer_path_index, ray_throughput);
	}
};

template <int PathIntegrator>
struct NEEDeferredMISContextSpecialized<LSS_RIS_BSDF_AND_LIGHT, PathIntegrator>
{
	float3_t last_view_direction;
	float3_t last_shading_point;
	float3_t last_shading_normal;
	float3_t last_geometric_normal;
	RayVolumeState last_volume_state;

	int last_primitive_index;
	float2_t last_texcoords;
	unsigned int last_primary_gbuffer_path_index = NEE_DEFERRED_INVALID_PATH_INDEX;

	// Ray throughput before multiplication with last_bsdf_throughput
	ColorRGB32F last_ray_throughput;

	// BSDF * cos_theta
	ColorRGB32F last_bsdf_x_cos_theta;
	float last_bsdf_sample_pdf;

	RISReservoir ris_reservoir;
	NEEDeferredLightProposalState last_light_proposal_state;
	bool last_can_do_light_sampling;

	HIPRT_DEVICE void fill_last_hit_information(HitInfo& closest_hit_info,
												const float3_t& view_direction,
												const RayVolumeState& volume_state,
												int primitive_index,
												float2_t texcoords,
												unsigned int primary_gbuffer_path_index,
												const ColorRGB32F& ray_throughput)
	{
		last_view_direction				= view_direction;
		last_shading_point				= closest_hit_info.inter_point;
		last_shading_normal				= closest_hit_info.shading_normal;
		last_geometric_normal			= closest_hit_info.geometric_normal;
		last_volume_state				= volume_state;
		last_primitive_index			= primitive_index;
		last_texcoords					= texcoords;
		last_primary_gbuffer_path_index = primary_gbuffer_path_index;
		last_ray_throughput				= ray_throughput;
	}

	HIPRT_DEVICE void fill_last_bsdf_information(ColorRGB32F bsdf_cos_theta, float bsdf_pdf)
	{
		last_bsdf_x_cos_theta = bsdf_cos_theta;
		last_bsdf_sample_pdf  = bsdf_pdf;
	}

	HIPRT_DEVICE void fill_ris_reservoir(const RISReservoir& reservoir)
	{
		ris_reservoir = reservoir;
	}

	HIPRT_DEVICE DeviceUnpackedPrincipledFullMaterial get_last_material(const HIPRTRenderData& render_data) const
	{
		return load_deferred_material_reference(render_data, last_primitive_index, last_texcoords, last_primary_gbuffer_path_index);
	}

	HIPRT_DEVICE void set_last_light_proposal_state(const NEEDeferredLightProposalState& proposal_state, bool can_do_light_sampling)
	{
		last_light_proposal_state  = proposal_state;
		last_can_do_light_sampling = can_do_light_sampling;
	}

	HIPRT_DEVICE const NEEDeferredLightProposalState& get_last_light_proposal_state() const
	{
		return last_light_proposal_state;
	}
};

template <>
struct NEEDeferredMISContextSpecialized<LSS_RIS_BSDF_AND_LIGHT, PATH_SAMPLING_RESTIR_PT>
{
	float3_t last_view_direction;
	float3_t last_shading_point;
	float3_t last_shading_normal;

	int last_primitive_index;
	float2_t last_texcoords;
	unsigned int last_primary_gbuffer_path_index = NEE_DEFERRED_INVALID_PATH_INDEX;

	// Ray throughput before multiplication with last_bsdf_throughput
	ColorRGB32F last_ray_throughput;

	// BSDF * cos_theta
	ColorRGB32F last_bsdf_x_cos_theta;
	float last_bsdf_sample_pdf;
	BSDFIncidentLightInfo last_bsdf_incident_light_info;
	NEEDeferredLightProposalState last_light_proposal_state;
	bool last_can_do_light_sampling;

	HIPRT_DEVICE void fill_last_hit_information(HitInfo& closest_hit_info,
												const float3_t& view_direction,
												const RayVolumeState& volume_state,
												int primitive_index,
												float2_t texcoords,
												unsigned int primary_gbuffer_path_index,
												const ColorRGB32F& ray_throughput)
	{
		last_view_direction				= view_direction;
		last_shading_point				= closest_hit_info.inter_point;
		last_shading_normal				= closest_hit_info.shading_normal;
		last_primitive_index			= primitive_index;
		last_texcoords					= texcoords;
		last_primary_gbuffer_path_index = primary_gbuffer_path_index;
		last_ray_throughput				= ray_throughput;
	}

	HIPRT_DEVICE void fill_last_bsdf_information(ColorRGB32F bsdf_cos_theta, float bsdf_pdf)
	{
		last_bsdf_x_cos_theta = bsdf_cos_theta;
		last_bsdf_sample_pdf  = bsdf_pdf;
	}

	HIPRT_DEVICE void fill_ris_reservoir(const RISReservoir& reservoir) {}

	HIPRT_DEVICE DeviceUnpackedPrincipledFullMaterial get_last_material(const HIPRTRenderData& render_data) const
	{
		if (last_primary_gbuffer_path_index != NEE_DEFERRED_INVALID_PATH_INDEX)
			return render_data.g_buffer.materials[last_primary_gbuffer_path_index].unpack();

		int last_material_index = render_data.buffers.material_indices[last_primitive_index];

		return get_intersection_material(render_data, last_material_index, last_texcoords);
	}

	HIPRT_DEVICE void set_last_light_proposal_state(const NEEDeferredLightProposalState& proposal_state, bool can_do_light_sampling)
	{
		last_light_proposal_state  = proposal_state;
		last_can_do_light_sampling = can_do_light_sampling;
	}

	HIPRT_DEVICE const NEEDeferredLightProposalState& get_last_light_proposal_state() const
	{
		return last_light_proposal_state;
	}
};

using NEEDeferredMISContext = NEEDeferredMISContextSpecialized<DirectLightNEEEstimator>;

#endif // #ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_NEE_DEFERRED_MIS_CONTEXT_H
