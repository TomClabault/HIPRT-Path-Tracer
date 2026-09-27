/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_WAVEFRONT_SHADING_H
#define DEVICE_WAVEFRONT_SHADING_H

#include "Device/includes/Wavefront/WavefrontCommon.h"
#include "HostDeviceCommon/Material/MaterialTraits.h"
#include "HostDeviceCommon/Material/PrincipledLobeClassification.h"

HIPRT_DEVICE void wavefront_initialize_secondary_hit_material(RayPayloadCommon& ray_payload,
															  const SurfaceTransportMetadata& surface_transport_metadata,
															  Xorshift32Generator& random_number_generator)
{
	if (surface_transport_metadata.dispersion_scale > 0.0f && surface_transport_metadata.specular_transmission > 0.0f &&
		ray_payload.volume_state.sampled_wavelength == 0.0f)
		// If we hit a dispersive material, we sample the wavelength that will be used
		// for computing the wavelength dependent IORs used for dispersion
		//
		// We're also not re-doing the sampling if a wavelength has already been sampled for that path
		//
		// Negating the wavelength to indicate that the throughput filter of the wavelength
		// hasn't been applied yet (applied in principled_glass_eval())
		ray_payload.volume_state.sampled_wavelength = -sample_wavelength_uniformly(random_number_generator);
}

template <bool initialize_primary_path>
HIPRT_DEVICE RayPayloadCommon wavefront_make_ray_payload()
{
	if constexpr (initialize_primary_path)
		return RayPayloadCommon();
	else
		return RayPayloadCommon(NoInitTag{});
}

template <typename MaterialType>
HIPRT_DEVICE void wavefront_copy_compact_principled_material(const DeviceUnpackedPrincipledFullMaterial& source, MaterialType& destination)
{
	destination.emission = source.get_emission();
	if (source.is_emissive())
		destination.emission_flags |= EffectiveMaterialEmissionIsEmissive;
	destination.base_color = source.base_color;

	if constexpr (MaterialTraits<MaterialType>::family == KernelMaterialSpecializationDiffuse)
	{
#if PrincipledBSDFDiffuseLobe == PRINCIPLED_DIFFUSE_LOBE_OREN_NAYAR
		destination.oren_nayar_sigma = source.oren_nayar_sigma;
#endif
	}
	else if constexpr (MaterialTraits<MaterialType>::family == KernelMaterialSpecializationGlass)
	{
		destination.roughness					 = source.roughness;
		destination.anisotropy					 = source.anisotropy;
		destination.anisotropy_rotation			 = source.anisotropy_rotation;
		destination.ior							 = source.ior;
		destination.do_glass_energy_compensation = source.do_glass_energy_compensation;
	}
	else if constexpr (MaterialTraits<MaterialType>::family == KernelMaterialSpecializationSingleMetallic)
	{
		destination.roughness						= source.roughness;
		destination.anisotropy						= source.anisotropy;
		destination.anisotropy_rotation				= source.anisotropy_rotation;
		destination.metallic_F82					= source.metallic_F82;
		destination.metallic_F90					= source.metallic_F90;
		destination.metallic_F90_falloff_exponent	= source.metallic_F90_falloff_exponent;
		destination.do_metallic_energy_compensation = source.do_metallic_energy_compensation;
	}
	else if constexpr (MaterialTraits<MaterialType>::family == KernelMaterialSpecializationSpecularDiffuse)
	{
#if PrincipledBSDFDiffuseLobe == PRINCIPLED_DIFFUSE_LOBE_OREN_NAYAR
		destination.oren_nayar_sigma = source.oren_nayar_sigma;
#endif
		destination.roughness						= source.roughness;
		destination.anisotropy						= source.anisotropy;
		destination.anisotropy_rotation				= source.anisotropy_rotation;
		destination.ior								= source.ior;
		destination.specular						= source.specular;
		destination.specular_tint					= source.specular_tint;
		destination.specular_color					= source.specular_color;
		destination.specular_darkening				= source.specular_darkening;
		destination.do_specular_energy_compensation = source.do_specular_energy_compensation;
	}
}

template <typename MaterialType>
HIPRT_DEVICE bool wavefront_compute_next_indirect_bounce(HIPRTRenderData& render_data,
														 RayPayloadT<MaterialType>& ray_payload,
														 MaterialType& bsdf_material,
														 float dispersion_scale,
														 unsigned int path_index,
														 HitInfo& closest_hit_info,
														 float3_t view_direction,
														 hiprtRay& out_ray,
														 Xorshift32Generator& random_number_generator,
														 BSDFIncidentLightInfo& incident_light_info,
														 NEEDeferredMISContext& nee_deferred_MIS_context)
{
	return path_tracing_compute_next_indirect_bounce(render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, out_ray,
													 random_number_generator, incident_light_info, nee_deferred_MIS_context, dispersion_scale, path_index);
}

template <bool initialize_primary_path, typename MaterialType>
HIPRT_DEVICE void wavefront_load_current_bsdf_material(HIPRTRenderData& render_data,
													   unsigned int pixel_index,
													   const HitInfo& closest_hit_info,
													   const ResolvedMaterialUserControlsCache& resolved_user_controls,
													   MaterialType& out_material)
{
	if constexpr (initialize_primary_path)
		load_effective_material(render_data.g_buffer.materials[pixel_index], out_material);
	else
	{
		int material_index = render_data.buffers.material_indices[closest_hit_info.primitive_index];
		load_effective_material(render_data, material_index, closest_hit_info.texcoords, resolved_user_controls, out_material);
	}
}

template <typename MaterialType>
HIPRT_DEVICE bool wavefront_shade_hit_with_material(HIPRTRenderData& render_data,
													RayPayloadT<MaterialType>& ray_payload,
													MaterialType& bsdf_material,
													const NEEDeferredLightProposalState& proposal_state,
													HitInfo& closest_hit_info,
													hiprtRay& ray,
													int x,
													int y,
													NEEDeferredMISContext& nee_deferred_MIS_context,
													float dispersion_scale,
													unsigned int path_index,
													Xorshift32Generator& random_number_generator,
													BSDFIncidentLightInfo& sampled_light_info)
{
	if (ray_payload.bounce > 0 || render_data.render_settings.enable_direct_lighting)
	{
		ray_payload.ray_color += estimate_direct_lighting(render_data, ray_payload, bsdf_material, ray_payload.throughput, closest_hit_info, -ray.direction, x,
														  y, nee_deferred_MIS_context, proposal_state, random_number_generator);

		sanity_check<true>(render_data, ray_payload.ray_color, x, y);
	}
	nee_deferred_MIS_context.set_last_light_proposal_state(proposal_state, bsdf_material.can_do_light_sampling());

	return wavefront_compute_next_indirect_bounce(render_data, ray_payload, bsdf_material, dispersion_scale, path_index, closest_hit_info, -ray.direction, ray,
												  random_number_generator, sampled_light_info, nee_deferred_MIS_context);
}

template <bool initialize_primary_path, typename MaterialType>
HIPRT_DEVICE bool wavefront_load_and_shade_hit(HIPRTRenderData& render_data,
											   RayPayloadCommon& ray_payload,
											   unsigned int pixel_index,
											   HitInfo& closest_hit_info,
											   hiprtRay& ray,
											   int x,
											   int y,
											   const ResolvedMaterialUserControlsCache& resolved_user_controls,
											   NEEDeferredMISContext& nee_deferred_MIS_context,
											   float dispersion_scale,
											   unsigned int path_index,
											   Xorshift32Generator& random_number_generator,
											   BSDFIncidentLightInfo& sampled_light_info)
{
	RayPayloadT<MaterialType> typed_ray_payload(NoInitTag{});
	static_cast<RayPayloadCommon&>(typed_ray_payload) = ray_payload;
	wavefront_load_current_bsdf_material<initialize_primary_path>(render_data, pixel_index, closest_hit_info, resolved_user_controls,
																  typed_ray_payload.material);
	NEEDeferredLightProposalState proposal_state{};
	if constexpr (initialize_primary_path)
		proposal_state = load_light_proposal_state<DirectLightSamplingStrategy, TrianglePointSamplingStrategy, static_cast<BSDFModel>(BSDF_MODEL)>(
			render_data.g_buffer.materials[pixel_index]);
	else
	{
		int material_index = render_data.buffers.material_indices[closest_hit_info.primitive_index];
		proposal_state	   = load_light_proposal_state<DirectLightSamplingStrategy, TrianglePointSamplingStrategy, static_cast<BSDFModel>(BSDF_MODEL)>(
			render_data, material_index, closest_hit_info.texcoords, typed_ray_payload.material.base_color.luminance(), resolved_user_controls);
	}

	bool valid_indirect_bounce =
		wavefront_shade_hit_with_material(render_data, typed_ray_payload, typed_ray_payload.material, proposal_state, closest_hit_info, ray, x, y,
										  nee_deferred_MIS_context, dispersion_scale, path_index, random_number_generator, sampled_light_info);
	static_cast<RayPayloadCommon&>(ray_payload) = static_cast<const RayPayloadCommon&>(typed_ray_payload);
	return valid_indirect_bounce;
}

template <bool initialize_primary_path>
HIPRT_DEVICE void wavefront_shade_path(HIPRTRenderData& render_data, unsigned int bounce_count, unsigned int pixel_index)
{
	using KernelMaterial =
		typename EffectiveMaterialFor<static_cast<BSDFModel>(BSDF_MODEL), static_cast<KernelMaterialSpecialization>(KERNEL_MATERIAL_SPECIALIZATION)>::Type;

	RayPayloadCommon ray_payload = wavefront_make_ray_payload<initialize_primary_path>();
	hiprtRay ray;
	HitInfo closest_hit_info;
	bool intersection_found;
	ResolvedMaterialUserControlsCache resolved_user_controls;
	PrincipledMaterialClassificationInputs classification_inputs;
	SurfaceTransportMetadata surface_transport_metadata;
	int current_material_index = -1;

	if constexpr (!initialize_primary_path)
		wavefront_load_secondary_material_state(render_data, pixel_index, ray_payload, closest_hit_info, intersection_found);

	Xorshift32Generator random_number_generator(initialize_primary_path ? render_data.get_updated_random_seed(pixel_index)
																		: render_data.wavefront_data.path_rng_states[pixel_index]);
	if constexpr (initialize_primary_path)
		wavefront_initialize_path(render_data, pixel_index, ray_payload, ray, closest_hit_info, intersection_found);

	if constexpr (!initialize_primary_path)
	{
		wavefront_load_secondary_shading_state(render_data, pixel_index, ray_payload, ray, closest_hit_info);

		{
			NEEDeferredMISContext previous_nee_deferred_MIS_context;
			wavefront_load_nee_deferred_mis_context(render_data, pixel_index, previous_nee_deferred_MIS_context);

			EffectiveMaterialEmission current_hit_emission;
			current_hit_emission.emission		= ColorRGB32F(0.0f);
			current_hit_emission.emission_flags = 0u;
			if (intersection_found)
			{
				int material_index = render_data.buffers.material_indices[closest_hit_info.primitive_index];
				load_effective_emission(render_data, material_index, closest_hit_info.texcoords, current_hit_emission);
			}

#if DirectLightNEEEstimator == LSS_RIS_BSDF_AND_LIGHT
			using PreviousBSDFMaterial = typename EffectiveMaterialFor<static_cast<BSDFModel>(BSDF_MODEL), KernelMaterialSpecializationAll>::Type;

			ResolvedMaterialUserControlsCache previous_resolved_user_controls;
			previous_resolved_user_controls.validity_mask = 0;
			// Stored light candidates are evaluated at the previous vertex even when the continuation ray misses or hits nonemissive geometry.
			if (previous_nee_deferred_MIS_context.last_primary_gbuffer_path_index == NEE_DEFERRED_INVALID_PATH_INDEX)
				previous_resolved_user_controls = wavefront_load_resolved_material_user_controls(render_data, pixel_index, previous_resolved_user_controls);

			// This kernel is specialized for the current hit, while the deferred context belongs to the previous vertex and can use another family.
			ray_payload.ray_color +=
				do_deferred_NEE_MIS<PreviousBSDFMaterial>(render_data, intersection_found, ray_payload, closest_hit_info, current_hit_emission,
														  previous_nee_deferred_MIS_context, previous_resolved_user_controls, random_number_generator);
#else
			ray_payload.ray_color += do_deferred_NEE_MIS(render_data, intersection_found, ray_payload, closest_hit_info, current_hit_emission,
														 previous_nee_deferred_MIS_context, random_number_generator);
#endif
		}

		if (!intersection_found)
		{
			ray_payload.ray_color += path_tracing_miss_gather_envmap(render_data, ray_payload, ray.direction, pixel_index);
			ray_payload.next_ray_state = RayState::MISSED;
			int x					   = static_cast<int>(pixel_index % render_data.render_settings.render_resolution.x);
			int y					   = static_cast<int>(pixel_index / render_data.render_settings.render_resolution.x);

			wavefront_finalize_path(render_data, pixel_index, x, y, ray_payload, random_number_generator);
			return;
		}

		current_material_index = render_data.buffers.material_indices[closest_hit_info.primitive_index];
		classification_inputs  = load_material_classification_inputs(render_data, current_material_index, closest_hit_info.texcoords, resolved_user_controls);
		wavefront_store_resolved_material_user_controls(render_data, pixel_index, resolved_user_controls);
		surface_transport_metadata = load_surface_transport_metadata(render_data, current_material_index, classification_inputs);
		wavefront_initialize_secondary_hit_material(ray_payload, surface_transport_metadata, random_number_generator);
	}
	else if (intersection_found)
	{
		classification_inputs	   = load_material_classification_inputs(render_data.g_buffer.materials[pixel_index], resolved_user_controls);
		surface_transport_metadata = load_surface_transport_metadata(render_data.g_buffer.materials[pixel_index], classification_inputs);
		ray_payload.volume_state.reconstruct_first_hit(render_data.buffers.material_indices, closest_hit_info.primitive_index,
													   surface_transport_metadata.dielectric_priority, surface_transport_metadata.dispersion_scale,
													   surface_transport_metadata.specular_transmission, random_number_generator);
	}

	int x = static_cast<int>(pixel_index % render_data.render_settings.render_resolution.x);
	int y = static_cast<int>(pixel_index / render_data.render_settings.render_resolution.x);
	NEEDeferredMISContext nee_deferred_MIS_context;

	if constexpr (initialize_primary_path)
	{
		if (!intersection_found)
		{
			ray_payload.ray_color += path_tracing_miss_gather_envmap(render_data, ray_payload, ray.direction, pixel_index);
			ray_payload.next_ray_state = RayState::MISSED;

			wavefront_finalize_path(render_data, pixel_index, x, y, ray_payload, random_number_generator);
			return;
		}
	}

	if (ray_payload.bounce == 0)
		store_denoiser_AOVs(render_data, pixel_index, closest_hit_info.shading_normal, render_data.g_buffer.materials[pixel_index].get_base_color());
	else
	{
		bool ReGIR_primary_hit = render_data.render_settings.regir_settings.compute_is_primary_hit(ray_payload);
		ReGIRMaterialInputs regir_material_inputs;
		if constexpr (initialize_primary_path)
			regir_material_inputs = load_regir_material_inputs(render_data.g_buffer.materials[pixel_index]);
		else
			regir_material_inputs = load_regir_material_inputs(render_data, current_material_index, resolved_user_controls, classification_inputs);

		// Storing data for ReGIR representative points
		ReGIR_update_representative_data(render_data, closest_hit_info.inter_point, closest_hit_info.geometric_normal, render_data.current_camera,
										 closest_hit_info.primitive_index, ReGIR_primary_hit, regir_material_inputs);
	}

	BSDFIncidentLightInfo sampled_light_info = BSDFIncidentLightInfo::NO_INFO; // This variable is never used, this is just for debugging on the CPU
																			   // so that we know what the BSDF sampled
	bool valid_indirect_bounce;

	valid_indirect_bounce = wavefront_load_and_shade_hit<initialize_primary_path, KernelMaterial>(
		render_data, ray_payload, pixel_index, closest_hit_info, ray, x, y, resolved_user_controls, nee_deferred_MIS_context,
		surface_transport_metadata.dispersion_scale, pixel_index, random_number_generator, sampled_light_info);

	if (!valid_indirect_bounce)
	{
		ray_payload.bounce++;
#if DirectLightNEEEstimatorHasBSDFSampling
		wavefront_queue_path_for_tracing(render_data, pixel_index, ray_payload, ray, closest_hit_info, nee_deferred_MIS_context, random_number_generator, true);
#else  // #if DirectLightNEEEstimatorHasBSDFSampling
		wavefront_finalize_path(render_data, pixel_index, x, y, ray_payload, random_number_generator);
#endif // #if DirectLightNEEEstimatorHasBSDFSampling
		return;
	}

	if (ray_payload.bounce >= static_cast<int>(bounce_count))
	{
		// The original megakernel increments the loop counter before the final deferred NEE pass.
		ray_payload.bounce++;
#if DirectLightNEEEstimatorHasBSDFSampling
		wavefront_queue_path_for_tracing(render_data, pixel_index, ray_payload, ray, closest_hit_info, nee_deferred_MIS_context, random_number_generator, true);
#else  // #if DirectLightNEEEstimatorHasBSDFSampling
		wavefront_finalize_path(render_data, pixel_index, x, y, ray_payload, random_number_generator);
#endif // #if DirectLightNEEEstimatorHasBSDFSampling
		return;
	}

	ray_payload.bounce++;
	wavefront_queue_path_for_tracing(render_data, pixel_index, ray_payload, ray, closest_hit_info, nee_deferred_MIS_context, random_number_generator, false);
}

#endif // #ifndef DEVICE_WAVEFRONT_SHADING_H
