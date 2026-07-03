/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_DI_INITIAL_CANDIDATES_H
#define KERNELS_RESTIR_DI_INITIAL_CANDIDATES_H

#include "Device/includes/BSDFs/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/PDFTriangles.h"
#include "Device/includes/LightSampling/TriangleEmissiveSampling.h"
#include "Device/includes/LightSampling/TriangleEmissiveSamplingReGIR.h"
#include "Device/includes/ReSTIR/DI/TargetFunction.h"
#include "Device/includes/ReSTIR/DI/Utils.h"
#include "Device/includes/ReSTIR/GI/Utils.h"
#include "Device/includes/TriangleLoadUtils.h"

#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRDIOptions.h"
#include "HostDeviceCommon/Maths/Math.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE ReSTIRDISampleArray<DirectLightSampleCount<DirectLightSamplingStrategy>()> sample_light_candidate_array(
	const HIPRTRenderData& render_data,
	float envmap_candidate_probability,
	const float3_t& view_direction,
	const HitInfo& closest_hit_info,
	RayPayload& ray_payload,
	Xorshift32Generator& random_number_generator)
{
	ReSTIRDISampleArray<DirectLightSampleCount<DirectLightSamplingStrategy>()> di_samples;

	float3_t evaluated_point = closest_hit_info.inter_point;

	if (random_number_generator() > envmap_candidate_probability)
	{
		LightSamplePointArray<DirectLightSampleCount<DirectLightSamplingStrategy>()> light_samples =
			sample_one_point_on_light(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
									  closest_hit_info.geometric_normal, closest_hit_info.primitive_index, ray_payload, random_number_generator);

		for (int i = 0; i < DirectLightSampleCount<DirectLightSamplingStrategy>(); i++)
		{
			LightSamplePointInformation& light_sample_info = light_samples[i];

			di_samples[i].emissive_triangle_global_index = light_sample_info.emissive_triangle_global_index;
			di_samples[i].point_on_light_source			 = light_sample_info.point_on_light;
			di_samples[i].pdf							 = light_sample_info.area_measure_pdf;

			// Accounting for the probability of sampling a light, not the envmap
			// (which has probability 'envmap_candidate_probability')
			di_samples[i].pdf *= (1.0f - envmap_candidate_probability);
			di_samples[i].emission = light_sample_info.emission;
		}
	}
	else
	{
		// Envmap samples

		// For simplicity of implementation in the MIS weights, we're sampling the envmap as many
		// times as we would sample lights
		for (int i = 0; i < DirectLightSampleCount<DirectLightSamplingStrategy>(); i++)
		{
			float3_t envmap_sampled_direction;
			di_samples[i].emission = envmap_sample(render_data.world_settings, envmap_sampled_direction, di_samples[i].pdf, random_number_generator);
			// Taking into account the fact that we only have a 1 in 'envmap_candidate_probability' chance to sample
			// the envmap
			di_samples[i].pdf *= envmap_candidate_probability;
			di_samples[i].emissive_triangle_global_index = -1;
			// Storing in envmap space
			di_samples[i].point_on_light_source = matrix_X_vec(render_data.world_settings.world_to_envmap_matrix, envmap_sampled_direction);
			di_samples[i].flags |= ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE;
		}
	}

	return di_samples;
}

HIPRT_DEVICE void sample_light_candidates(const HIPRTRenderData& render_data,
										  const HitInfo& closest_hit_info,
										  RayPayload& ray_payload,
										  ReSTIRDIReservoir& reservoir,
										  int nb_light_candidates,
										  int nb_bsdf_candidates,
										  float envmap_candidate_probability,
										  const float3_t& view_direction,
										  Xorshift32Generator& random_number_generator,
										  const int2_t& pixel_coords)
{
	for (int i = 0; i < nb_light_candidates; i++)
	{
		ReSTIRDISampleArray<DirectLightSampleCount<DirectLightSamplingStrategy>()> di_samples =
			sample_light_candidate_array(render_data, envmap_candidate_probability, view_direction, closest_hit_info, ray_payload, random_number_generator);

		for (int sample_index = 0; sample_index < DirectLightSampleCount<DirectLightSamplingStrategy>(); sample_index++)
		{
			ReSTIRDIInitialSample& light_sample = di_samples[sample_index];
			if (light_sample.emissive_triangle_global_index == -1 && !light_sample.is_envmap_sample())
				continue; // Invalid sample

			float distance_to_light;
			float3_t to_light_direction;
			if (light_sample.is_envmap_sample())
			{
				to_light_direction = matrix_X_vec(render_data.world_settings.envmap_to_world_matrix, light_sample.point_on_light_source);
				distance_to_light  = 1.0e35f;
			}
			else
			{
				to_light_direction = light_sample.point_on_light_source - closest_hit_info.inter_point;
				to_light_direction = to_light_direction / (distance_to_light = hippt::length(to_light_direction)); // Normalization
			}

			float candidate_weight				 = 0.0f;
			float cosine_term_at_evaluated_point = hippt::dot(closest_hit_info.shading_normal, to_light_direction);
			if (cosine_term_at_evaluated_point > 0.0f && light_sample.pdf > 0.0f)
			{
				float bsdf_pdf_solid_angle;
				BSDFIncidentLightInfo incident_light_info;
				BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, to_light_direction,
										 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness);
				ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf_solid_angle, random_number_generator);

				// Filling a surface to give to 'ReSTIR_DI_evaluate_target_function'
				ReSTIRSurface surface;
				surface.geometric_normal = closest_hit_info.geometric_normal;
				surface.primitive_index	 = closest_hit_info.primitive_index;
				surface.material		 = ray_payload.material;
				surface.ray_volume_state = ray_payload.volume_state;
				surface.shading_normal	 = closest_hit_info.shading_normal;
				surface.shading_point	 = closest_hit_info.inter_point;
				surface.view_direction	 = view_direction;

				float target_function =
					ReSTIR_DI_evaluate_target_function<false>(render_data, light_sample.to_reservoir_sample(), surface, random_number_generator);
				float light_pdf_solid_angle;
				if (light_sample.is_envmap_sample())
					// For envmap sample, the PDF is already in solid angle
					light_pdf_solid_angle = light_sample.pdf;
				else
				{
					float3_t light_normal = triangle_load_normal_not_normalized(render_data, light_sample.emissive_triangle_global_index);
					float normal_length	  = hippt::length(light_normal);
					float light_area	  = normal_length * 0.5f;
					light_normal /= normal_length;

					// Converting from area measure to solid angle measure so that we use the balance heuristic we the same measure PDFs
					// (same measure for the BSDF PDF and the light PDF)
					//
					// Removing the envmap proba to avoid double counting it below in
					light_pdf_solid_angle = area_to_solid_angle_pdf(light_sample.pdf / (1.0f - envmap_candidate_probability), distance_to_light,
																	compute_cosine_term_at_light_source(light_normal, -to_light_direction));
					light_pdf_solid_angle *= (1.0f - envmap_candidate_probability);
				}

				float mis_weight = balance_heuristic(light_pdf_solid_angle, nb_light_candidates * DirectLightIntegrationFactor<DirectLightSamplingStrategy>(),
													 bsdf_pdf_solid_angle, nb_bsdf_candidates);
				candidate_weight = mis_weight * target_function / light_sample.pdf;
				sanity_check<true>(render_data, ColorRGB32F(candidate_weight), 0, 0);

				light_sample.target_function = target_function;
			}

#if ReSTIR_DI_InitialTargetFunctionVisibility == KERNEL_OPTION_TRUE
			if (!render_data.render_settings.do_render_low_resolution() && light_sample.target_function > 0.0f)
			{
				// Only doing visiblity if we're render at low resolution
				// (meaning we're moving the camera) for better movement framerates
				// Also, only testing visibility if we got a valid sample

				hiprtRay shadow_ray;
				shadow_ray.origin	 = closest_hit_info.inter_point;
				shadow_ray.direction = to_light_direction;

				bool visible = !evaluate_shadow_ray_occluded(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index,
															 /* bounce. Always 0 for ReSTIR DI*/ 0, random_number_generator);
				if (!visible)
				{
					// Sample occluded, it is not going to be resampled anyways because it is
					// going to have a 0 contribution so we just take it into account in the
					// reservoir (because even if it has zero-contribution, this is still a resampled sample)
					reservoir.M++;

					// And we go onto the next sample
					continue;
				}

				// We are now sure that if the sample survived, it is unoccluded
				light_sample.flags |= RESTIR_DI_FLAGS_UNOCCLUDED;
			}
#endif

			reservoir.add_one_candidate(light_sample, candidate_weight, random_number_generator);
			reservoir.sanity_check(make_int2(-1, -1));
		}
	}
}

HIPRT_DEVICE void sample_bsdf_candidates(const HIPRTRenderData& render_data,
										 const HitInfo& closest_hit_info,
										 RayPayload& ray_payload,
										 ReSTIRDIReservoir& reservoir,
										 int nb_light_candidates,
										 int nb_bsdf_candidates,
										 float envmap_candidate_probability,
										 const float3_t& view_direction,
										 Xorshift32Generator& random_number_generator)
{
	// Sampling the BSDF candidates
	for (int i = 0; i < nb_bsdf_candidates; i++)
	{
		float bsdf_sample_pdf_solid_angle = 0.0f;
		float3_t bsdf_sampled_direction;

		BSDFIncidentLightInfo sampled_lobe_info;
		BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, make_float3(0.0f, 0.0f, 0.0f),
								 sampled_lobe_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness);
		ColorRGB32F bsdf_color =
			bsdf_dispatcher_sample(render_data, bsdf_context, bsdf_sampled_direction, bsdf_sample_pdf_solid_angle, random_number_generator);

		if (bsdf_sample_pdf_solid_angle > 0.0f)
		{
			hiprtRay bsdf_ray;
			bsdf_ray.origin	   = closest_hit_info.inter_point;
			bsdf_ray.direction = bsdf_sampled_direction;

			BSDFLightSampleRayHitInfo shadow_light_ray_hit_info;
			bool hit_found = evaluate_bsdf_light_sample_ray(render_data, bsdf_ray, 1.0e35f, shadow_light_ray_hit_info, closest_hit_info.primitive_index,
															random_number_generator);
			if (hit_found && !shadow_light_ray_hit_info.hit_emission.is_black() &&
				compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -bsdf_sampled_direction) > 0.0f)
			{
				// If we intersected an emissive material, compute the weight.
				// Otherwise, the weight is 0 because of the emision being 0 so we just don't compute it

				// Filling a surface to give to 'ReSTIR_DI_evaluate_target_function'
				ReSTIRSurface surface;
				surface.geometric_normal = closest_hit_info.geometric_normal;
				surface.primitive_index	 = closest_hit_info.primitive_index;
				surface.material		 = ray_payload.material;
				surface.ray_volume_state = ray_payload.volume_state;
				surface.shading_normal	 = closest_hit_info.shading_normal;
				surface.shading_point	 = closest_hit_info.inter_point;
				surface.view_direction	 = view_direction;

				ReSTIRDIInitialSample bsdf_RIS_sample;
				bsdf_RIS_sample.emissive_triangle_global_index = shadow_light_ray_hit_info.hit_prim_index;
				bsdf_RIS_sample.point_on_light_source		   = bsdf_ray.origin + bsdf_ray.direction * shadow_light_ray_hit_info.hit_distance;
				bsdf_RIS_sample.flags |= ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED;
				bsdf_RIS_sample.flags |= ReSTIRDIInitialSample::flags_from_BSDF_incident_light_info(sampled_lobe_info);
				bsdf_RIS_sample.target_function =
					ReSTIR_DI_evaluate_target_function<false>(render_data, bsdf_RIS_sample.to_reservoir_sample(), surface, random_number_generator);

				float light_pdf_solid_angle = 0.0f;
				bool refraction_sampled		= hippt::dot(bsdf_sampled_direction, closest_hit_info.shading_normal) < 0.0f;
				if (!refraction_sampled)
				{
					// TODO we should just allow refraction light samples instead of this

					// Only computing the light PDF if we're not refracting
					//
					// Why?
					//
					// Because right now, we allow sampling BSDF refractions. This means that we can sample a light
					// that is inside an object with a *BSDF sample*. However, a *light sample* to the same light cannot
					// be sampled because there's is going to be the surface of the object we're currently on in-between.
					// Basically, we are not allowing light sample refractions and so they should have a MIS weight of 0 which
					// is what we're doing here: the pdf of a *light sample* that refracts through a surface is 0.
					//
					// If not doing that, we're going to have bad MIS weights that don't sum up to 1
					// (because the BSDF sample, that should have weight 1 [or to be precise: 1 / nb_bsdf_samples]
					// will have weight 1 / (1 + nb_light_samples) [or to be precise: 1 / (nb_bsdf_samples + nb_light_samples)]
					// and this is going to cause darkening as the number of light samples grows)

					light_pdf_solid_angle =
						pdf_of_emissive_triangle_hit_solid_angle(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
																 ray_payload.material, shadow_light_ray_hit_info, bsdf_sampled_direction);
				}

				// Our light sampler is only chosen with probability '1.0f - envmap_candidate_probability'
				// so we multiply that here to take that into account
				light_pdf_solid_angle *= (1.0f - envmap_candidate_probability);

				float mis_weight = balance_heuristic(bsdf_sample_pdf_solid_angle, nb_bsdf_candidates, light_pdf_solid_angle,
													 nb_light_candidates * DirectLightIntegrationFactor<DirectLightSamplingStrategy>());

				float bsdf_sample_pdf_area_measure = bsdf_sample_pdf_solid_angle;
				bsdf_sample_pdf_area_measure /= (shadow_light_ray_hit_info.hit_distance * shadow_light_ray_hit_info.hit_distance);
				bsdf_sample_pdf_area_measure *= compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -bsdf_sampled_direction);

				float candidate_weight = mis_weight * bsdf_RIS_sample.target_function / bsdf_sample_pdf_area_measure;

				reservoir.add_one_candidate(bsdf_RIS_sample, candidate_weight, random_number_generator);
				reservoir.sanity_check(make_int2(-1, -1));
			}
			else if (!hit_found && render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP)
			{
				// Envmap hit, this becomes an envmap sample

				// Not allowing refraction envmap samples here
				// TODO fixthis, we should allow them
				if (hippt::dot(closest_hit_info.shading_normal, bsdf_sampled_direction) > 0.0f)
				{
					float envmap_pdf;
					ColorRGB32F envmap_radiance = envmap_eval(render_data, bsdf_sampled_direction, envmap_pdf);

					// Filling a surface to give to 'ReSTIR_DI_evaluate_target_function'
					ReSTIRSurface surface;
					surface.geometric_normal = closest_hit_info.geometric_normal;
					surface.primitive_index	 = closest_hit_info.primitive_index;
					surface.material		 = ray_payload.material;
					surface.ray_volume_state = ray_payload.volume_state;
					surface.shading_normal	 = closest_hit_info.shading_normal;
					surface.shading_point	 = closest_hit_info.inter_point;
					surface.view_direction	 = view_direction;

					ReSTIRDIInitialSample bsdf_RIS_sample;
					bsdf_RIS_sample.emissive_triangle_global_index = -1;
					// Storing in envmap space
					bsdf_RIS_sample.point_on_light_source = matrix_X_vec(render_data.world_settings.world_to_envmap_matrix, bsdf_sampled_direction);
					bsdf_RIS_sample.flags |= ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED;
					bsdf_RIS_sample.flags |= ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE;
					bsdf_RIS_sample.flags |= ReSTIRDIInitialSample::flags_from_BSDF_incident_light_info(sampled_lobe_info);
					bsdf_RIS_sample.target_function =
						ReSTIR_DI_evaluate_target_function<false>(render_data, bsdf_RIS_sample.to_reservoir_sample(), surface, random_number_generator);

					// Not taking the light sampling PDF into account in the balance heuristic because a envmap hit
					// (not a light surface hit) can never be sampled by a light-surface sampler and so the PDF
					// of the current envmap sample is always 0 for a light sampler.

					// We're evaluating the probability of choosing that BSDF-sample direction with the envmap sampler.
					// Because our envmap sampler is chosen only with probability 'envmap_candidate_probability', we multiply
					// that here to account for that
					envmap_pdf *= envmap_candidate_probability;
					float mis_weight	   = balance_heuristic(bsdf_sample_pdf_solid_angle, nb_bsdf_candidates, envmap_pdf, nb_light_candidates);
					float candidate_weight = mis_weight * bsdf_RIS_sample.target_function / bsdf_sample_pdf_solid_angle;

					reservoir.add_one_candidate(bsdf_RIS_sample, candidate_weight, random_number_generator);
					reservoir.sanity_check(make_int2(-1, -1));
				}
			}
		}
	}
}

HIPRT_DEVICE ReSTIRDIReservoir sample_initial_candidates(const HIPRTRenderData& render_data,
														 const int2_t& pixel_coords,
														 RayPayload& ray_payload,
														 const HitInfo closest_hit_info,
														 const float3_t& view_direction,
														 Xorshift32Generator& random_number_generator)
{
	ReSTIRDIReservoir reservoir;

	// If we're rendering at low resolution, only doing 1 candidate of each
	// for better interactive framerates
	int initial_nb_light_cand = render_data.render_settings.restir_di_settings.initial_candidates.number_of_initial_light_candidates;
	int initial_nb_bsdf_cand  = render_data.render_settings.restir_di_settings.initial_candidates.number_of_initial_bsdf_candidates;
#if DirectLightSamplingStrategy == LSS_BASE_REGIR
	// With ReGIR, initial BSDF candidates are controlled by the ReGIR sampling, not by
	// ReSTIR DI
	initial_nb_bsdf_cand = 0;
#endif

	int nb_light_candidates			   = render_data.render_settings.do_render_low_resolution() ? hippt::min(1, initial_nb_light_cand) : initial_nb_light_cand;
	int nb_bsdf_candidates			   = render_data.render_settings.do_render_low_resolution() ? hippt::min(1, initial_nb_bsdf_cand) : initial_nb_bsdf_cand;
	float envmap_candidate_probability = 0.0f;
	if (render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP && EnvmapSamplingStrategy != ESS_NO_SAMPLING)
	{
		if (render_data.buffers.emissive_triangles_count == 0)
			// Only the envmap to sample
			envmap_candidate_probability = 1.0f;
		else
			envmap_candidate_probability = render_data.render_settings.restir_di_settings.initial_candidates.envmap_candidate_probability;
	}

	// Sampling candidates with weighted reservoir sampling

	sample_light_candidates(render_data, closest_hit_info, ray_payload, reservoir, nb_light_candidates, nb_bsdf_candidates, envmap_candidate_probability,
							view_direction, random_number_generator, pixel_coords);
	sample_bsdf_candidates(render_data, closest_hit_info, ray_payload, reservoir, nb_light_candidates, nb_bsdf_candidates, envmap_candidate_probability,
						   view_direction, random_number_generator);

	reservoir.end();
	reservoir.sanity_check(pixel_coords);
	// There's no need to keep M > 1 here, if you have 4 light candidates and 1 BSDF candidates, that's 5 samples.
	// But if you divide everyone by 5, everything stays correct. That allows manipulating the M-cap without having
	// to take the number of initial candidates into account
	reservoir.confidence = 1;

	return reservoir;
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_DI_InitialCandidates(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_DI_InitialCandidates(HIPRTRenderData render_data, int x, int y)
#endif
{
	if (render_data.buffers.emissive_triangles_count == 0 && render_data.world_settings.ambient_light_type != AmbientLightType::ENVMAP)
		// No initial candidates to sample since no lights
		return;

#ifdef __KERNELCC__
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
	if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
		return;

	uint32_t pixel_index				   = (x + y * render_data.render_settings.render_resolution.x);
	DevicePackedEffectiveMaterial material = render_data.g_buffer.materials[pixel_index];

	Xorshift32Generator random_number_generator(render_data.get_updated_random_seed(pixel_index));

	if (!render_data.aux_buffers.pixel_active[pixel_index] || render_data.g_buffer.first_hit_prim_index[pixel_index] == -1)
		// Pixel inactive because of adaptive sampling, returning
		// Or also we don't have a primary hit
		return;

	HitInfo hit_info;
	hit_info.geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();
	hit_info.shading_normal	  = render_data.g_buffer.shading_normals[pixel_index].unpack();
	hit_info.inter_point	  = render_data.g_buffer.primary_hit_position[pixel_index];
	hit_info.primitive_index  = render_data.g_buffer.first_hit_prim_index[pixel_index];

	RayPayload ray_payload;
	ray_payload.material = material.unpack();
	// Because this is the camera hit (and assuming the camera isn't inside volumes for now),
	// the ray volume state after the camera hit is just an empty interior stack but with
	// the material index that we hit pushed onto the stack. That's it. Because it is that
	// simple, we don't have the ray volume state in the GBuffer but rather we can
	// reconstruct the ray volume state on the fly
	ray_payload.volume_state.reconstruct_first_hit(ray_payload.material, render_data.buffers.material_indices,
												   render_data.g_buffer.first_hit_prim_index[pixel_index], random_number_generator);

	float3_t view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);
	// Producing and storing the reservoir
	ReSTIRDIReservoir initial_candidates_reservoir =
		sample_initial_candidates(render_data, make_int2(x, y), ray_payload, hit_info, view_direction, random_number_generator);

#if ReSTIR_DI_DoVisibilityReuse == KERNEL_OPTION_TRUE
	ReSTIR_DI_visibility_test_kill_reservoir(render_data, initial_candidates_reservoir, hit_info.inter_point, hit_info.primitive_index,
											 random_number_generator);
#endif

	render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs[pixel_index] = initial_candidates_reservoir;
	// render_data.store_updated_random_seed(pixel_index, random_number_generator.m_state.seed);
}

#endif
