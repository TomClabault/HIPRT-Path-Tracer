#ifndef DEVICE_PRINCIPLED_SPECIALIZED_H
#define DEVICE_PRINCIPLED_SPECIALIZED_H

#include "Device/includes/BSDFs/Principled.h"

template <typename MaterialType>
HIPRT_DEVICE static ColorRGB32F principled_specialized_metallic_eval(const HIPRTRenderData& render_data,
																	 BSDFContextT<MaterialType>& bsdf_context,
																	 float& pdf,
																	 Xorshift32Generator& rng)
{
	float3_t tangent, bitangent;
	build_rotated_ONB(bsdf_context.shading_normal, tangent, bitangent, bsdf_context.material.anisotropy_rotation * hippt::M_Pi);

	float3_t local_view_direction	  = world_to_local_frame(tangent, bitangent, bsdf_context.shading_normal, bsdf_context.view_direction);
	float3_t local_to_light_direction = world_to_local_frame(tangent, bitangent, bsdf_context.shading_normal, bsdf_context.to_light_direction);
	float3_t local_half_vector		  = hippt::normalize(local_view_direction + local_to_light_direction);
	float incident_medium_ior		  = principled_get_incident_medium_ior(render_data, bsdf_context.volume_state);

	float metallic_weight;
	PrincipledLobeUserWeights user_weights;
	user_weights.metallic		  = 1.0f;
	PrincipledLobeWeights weights = compute_principled_lobe_weights(user_weights, !bsdf_context.volume_state.inside_material);
	metallic_weight				  = weights.metallic_first;

	ColorRGB32F contribution = principled_metallic_eval(render_data, bsdf_context, bsdf_context.material.roughness, bsdf_context.material.anisotropy,
														incident_medium_ior, local_view_direction, local_to_light_direction, local_half_vector, pdf, rng);

	return contribution * metallic_weight;
}

template <typename MaterialType>
HIPRT_DEVICE static float principled_specialized_metallic_pdf(const HIPRTRenderData& render_data, BSDFContextT<MaterialType>& bsdf_context)
{
	float3_t tangent, bitangent;
	build_rotated_ONB(bsdf_context.shading_normal, tangent, bitangent, bsdf_context.material.anisotropy_rotation * hippt::M_Pi);

	float3_t local_view_direction	  = world_to_local_frame(tangent, bitangent, bsdf_context.shading_normal, bsdf_context.view_direction);
	float3_t local_to_light_direction = world_to_local_frame(tangent, bitangent, bsdf_context.shading_normal, bsdf_context.to_light_direction);
	float3_t local_half_vector		  = hippt::normalize(local_view_direction + local_to_light_direction);

	float pdf = principled_metallic_pdf(render_data, bsdf_context, bsdf_context.material.roughness, bsdf_context.material.anisotropy, local_view_direction,
										local_to_light_direction, local_half_vector);

	return pdf;
}

template <typename MaterialType>
HIPRT_DEVICE static ColorRGB32F principled_specialized_glass_eval(const HIPRTRenderData& render_data,
																  BSDFContextT<MaterialType>& bsdf_context,
																  float& pdf,
																  Xorshift32Generator& rng)
{
	float3_t tangent, bitangent;
	build_rotated_ONB(bsdf_context.shading_normal, tangent, bitangent, bsdf_context.material.anisotropy_rotation * hippt::M_Pi);

	float3_t local_view_direction	  = world_to_local_frame(tangent, bitangent, bsdf_context.shading_normal, bsdf_context.view_direction);
	float3_t local_to_light_direction = world_to_local_frame(tangent, bitangent, bsdf_context.shading_normal, bsdf_context.to_light_direction);

	return principled_glass_eval(render_data, bsdf_context, local_view_direction, local_to_light_direction, pdf, rng);
}

template <typename MaterialType>
HIPRT_DEVICE static float principled_specialized_glass_pdf(const HIPRTRenderData& render_data, BSDFContextT<MaterialType>& bsdf_context)
{
	float3_t tangent, bitangent;
	build_rotated_ONB(bsdf_context.shading_normal, tangent, bitangent, bsdf_context.material.anisotropy_rotation * hippt::M_Pi);

	float3_t local_view_direction	  = world_to_local_frame(tangent, bitangent, bsdf_context.shading_normal, bsdf_context.view_direction);
	float3_t local_to_light_direction = world_to_local_frame(tangent, bitangent, bsdf_context.shading_normal, bsdf_context.to_light_direction);

	return principled_glass_pdf(render_data, bsdf_context, local_view_direction, local_to_light_direction);
}

template <typename MaterialType>
HIPRT_DEVICE static ColorRGB32F principled_specialized_specular_diffuse_eval(const HIPRTRenderData& render_data,
																			 BSDFContextT<MaterialType>& bsdf_context,
																			 float& pdf,
																			 Xorshift32Generator& rng)
{
	pdf = 0.0f;
	float3_t tangent, bitangent;
	build_ONB(bsdf_context.shading_normal, tangent, bitangent);

	float3_t local_view_direction	  = world_to_local_frame(tangent, bitangent, bsdf_context.shading_normal, bsdf_context.view_direction);
	float3_t local_to_light_direction = world_to_local_frame(tangent, bitangent, bsdf_context.shading_normal, bsdf_context.to_light_direction);
	float3_t local_half_vector		  = hippt::normalize(local_view_direction + local_to_light_direction);

	float3_t rotated_tangent, rotated_bitangent;
	build_rotated_ONB(bsdf_context.shading_normal, rotated_tangent, rotated_bitangent, bsdf_context.material.anisotropy_rotation * hippt::M_Pi);

	float3_t local_view_direction_rotated = world_to_local_frame(rotated_tangent, rotated_bitangent, bsdf_context.shading_normal, bsdf_context.view_direction);
	float3_t local_to_light_direction_rotated =
		world_to_local_frame(rotated_tangent, rotated_bitangent, bsdf_context.shading_normal, bsdf_context.to_light_direction);
	float3_t local_half_vector_rotated = hippt::normalize(local_view_direction_rotated + local_to_light_direction_rotated);

	float incident_medium_ior = principled_get_incident_medium_ior(render_data, bsdf_context.volume_state);

	float specular_weight;
	float diffuse_weight;
	principled_specular_diffuse_lobe_weights(bsdf_context.material, !bsdf_context.volume_state.inside_material, specular_weight, diffuse_weight);

	float specular_probability, diffuse_probability;
	principled_specular_diffuse_sampling_probabilities(render_data, bsdf_context.material, local_view_direction.z, bsdf_context.volume_state, specular_weight,
													   diffuse_weight, specular_probability, diffuse_probability);

	ColorRGB32F layers_throughput(1.0f);

	return internal_eval_glossy_base(render_data, bsdf_context, local_view_direction, local_to_light_direction, local_half_vector, local_view_direction_rotated,
									 local_to_light_direction_rotated, local_half_vector_rotated, bsdf_context.shading_normal, incident_medium_ior,
									 diffuse_weight, specular_weight, false, diffuse_probability, specular_probability, layers_throughput, pdf, rng);
}

template <typename MaterialType>
HIPRT_DEVICE static float principled_specialized_specular_diffuse_pdf(const HIPRTRenderData& render_data, BSDFContextT<MaterialType>& bsdf_context)
{
	float3_t tangent, bitangent;
	build_ONB(bsdf_context.shading_normal, tangent, bitangent);

	float3_t local_view_direction	  = world_to_local_frame(tangent, bitangent, bsdf_context.shading_normal, bsdf_context.view_direction);
	float3_t local_to_light_direction = world_to_local_frame(tangent, bitangent, bsdf_context.shading_normal, bsdf_context.to_light_direction);
	float3_t local_half_vector		  = hippt::normalize(local_view_direction + local_to_light_direction);

	float3_t rotated_tangent, rotated_bitangent;
	build_rotated_ONB(bsdf_context.shading_normal, rotated_tangent, rotated_bitangent, bsdf_context.material.anisotropy_rotation * hippt::M_Pi);

	float3_t local_view_direction_rotated = world_to_local_frame(rotated_tangent, rotated_bitangent, bsdf_context.shading_normal, bsdf_context.view_direction);
	float3_t local_to_light_direction_rotated =
		world_to_local_frame(rotated_tangent, rotated_bitangent, bsdf_context.shading_normal, bsdf_context.to_light_direction);
	float3_t local_half_vector_rotated = hippt::normalize(local_view_direction_rotated + local_to_light_direction_rotated);

	float incident_medium_ior = principled_get_incident_medium_ior(render_data, bsdf_context.volume_state);

	float specular_weight;
	float diffuse_weight;
	principled_specular_diffuse_lobe_weights(bsdf_context.material, !bsdf_context.volume_state.inside_material, specular_weight, diffuse_weight);

	float specular_probability, diffuse_probability;
	principled_specular_diffuse_sampling_probabilities(render_data, bsdf_context.material, local_view_direction.z, bsdf_context.volume_state, specular_weight,
													   diffuse_weight, specular_probability, diffuse_probability);

	return internal_pdf_glossy_base(render_data, bsdf_context, local_view_direction, local_to_light_direction, local_half_vector, local_view_direction_rotated,
									local_to_light_direction_rotated, local_half_vector_rotated, bsdf_context.shading_normal, incident_medium_ior,
									diffuse_weight, specular_weight, false, diffuse_probability, specular_probability);
}

template <typename MaterialType>
HIPRT_DEVICE static ColorRGB32F principled_specialized_diffuse_eval(const BSDFContextT<MaterialType>& bsdf_context, float& pdf)
{
#if PrincipledBSDFDiffuseLobe == PRINCIPLED_DIFFUSE_LOBE_LAMBERTIAN
	return lambertian_brdf_eval(bsdf_context.material, hippt::dot(bsdf_context.to_light_direction, bsdf_context.shading_normal), pdf);
#elif PrincipledBSDFDiffuseLobe == PRINCIPLED_DIFFUSE_LOBE_OREN_NAYAR
	return oren_nayar_brdf_eval(bsdf_context.material, bsdf_context.view_direction, bsdf_context.shading_normal, bsdf_context.to_light_direction, pdf);
#endif
}

template <typename MaterialType>
HIPRT_DEVICE static float principled_specialized_diffuse_pdf(const BSDFContextT<MaterialType>& bsdf_context)
{
#if PrincipledBSDFDiffuseLobe == PRINCIPLED_DIFFUSE_LOBE_LAMBERTIAN
	return lambertian_brdf_pdf(hippt::dot(bsdf_context.to_light_direction, bsdf_context.shading_normal));
#elif PrincipledBSDFDiffuseLobe == PRINCIPLED_DIFFUSE_LOBE_OREN_NAYAR
	return oren_nayar_brdf_pdf(bsdf_context.to_light_direction);
#endif
}

template <bool sampleDirectionOnly, typename MaterialType>
HIPRT_DEVICE static ColorRGB32F principled_specialized_family_sample(const HIPRTRenderData& render_data,
																	 BSDFContextT<MaterialType>& bsdf_context,
																	 float3_t& output_direction,
																	 float& pdf,
																	 Xorshift32Generator& random_number_generator)
{
	pdf = 0.0f;

	constexpr KernelMaterialSpecialization family = MaterialTraits<MaterialType>::family;

	if constexpr (family == KernelMaterialSpecializationDiffuse)
	{
		// A single-lobe material does not need an RNG draw for lobe selection.

		if (bsdf_context.update_ray_volume_state)
			bsdf_context.volume_state.interior_stack.pop(false);

		bsdf_context.incident_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_DIFFUSE_LOBE;
		output_direction				 = principled_diffuse_sample(bsdf_context.shading_normal, random_number_generator);

		if (hippt::dot(output_direction, bsdf_context.geometric_normal) < 0.0f)
			return ColorRGB32F(0.0f);

		bsdf_context.to_light_direction = output_direction;
		if constexpr (sampleDirectionOnly)
		{
			pdf = 0.0f;
			return ColorRGB32F(0.0f);
		}

		return principled_specialized_diffuse_eval(bsdf_context, pdf);
	}
	else
	{
		float3_t tangent, bitangent;
		build_rotated_ONB(bsdf_context.shading_normal, tangent, bitangent, bsdf_context.material.anisotropy_rotation * hippt::M_Pi);

		float3_t local_view_direction = world_to_local_frame(tangent, bitangent, bsdf_context.shading_normal, bsdf_context.view_direction);

		if constexpr (family == KernelMaterialSpecializationGlass)
		{
			// Specialized glass sampling skips the top-level lobe-selection draw.
			output_direction = local_to_world_frame(tangent, bitangent, bsdf_context.shading_normal,
													principled_glass_sample(render_data, bsdf_context, local_view_direction, random_number_generator));
			if constexpr (sampleDirectionOnly)
			{
				pdf = 0.0f;
				return ColorRGB32F(0.0f);
			}
			bsdf_context.to_light_direction = output_direction;
			return principled_specialized_glass_eval(render_data, bsdf_context, pdf, random_number_generator);
		}
		else if constexpr (family == KernelMaterialSpecializationSingleMetallic)
		{
			// Specialized metallic sampling skips the top-level lobe-selection draw.
			if (bsdf_context.update_ray_volume_state)
				bsdf_context.volume_state.interior_stack.pop(false);
			bsdf_context.incident_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_FIRST_METAL_LOBE;
			output_direction =
				local_to_world_frame(tangent, bitangent, bsdf_context.shading_normal,
									 principled_metallic_sample(render_data, bsdf_context, bsdf_context.material.roughness, bsdf_context.material.anisotropy,
																local_view_direction, random_number_generator));
		}
		else if constexpr (family == KernelMaterialSpecializationSpecularDiffuse)
		{
			float specular_weight;
			float diffuse_weight;
			principled_specular_diffuse_lobe_weights(bsdf_context.material, !bsdf_context.volume_state.inside_material, specular_weight, diffuse_weight);
			float specular_probability, diffuse_probability;
			principled_specular_diffuse_sampling_probabilities(render_data, bsdf_context.material, local_view_direction.z, bsdf_context.volume_state,
															   specular_weight, diffuse_weight, specular_probability, diffuse_probability);

			float lobe_choice = random_number_generator();
			if (bsdf_context.update_ray_volume_state)
				bsdf_context.volume_state.interior_stack.pop(false);

			if (lobe_choice < specular_probability)
			{
				bsdf_context.incident_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_SPECULAR_LOBE;
				output_direction =
					local_to_world_frame(tangent, bitangent, bsdf_context.shading_normal,
										 principled_specular_sample(render_data, bsdf_context, bsdf_context.material.roughness,
																	bsdf_context.material.anisotropy, local_view_direction, random_number_generator));
			}
			else if (lobe_choice < specular_probability + diffuse_probability)
			{
				bsdf_context.incident_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_DIFFUSE_LOBE;
				output_direction				 = principled_diffuse_sample(bsdf_context.shading_normal, random_number_generator);
			}
			else
				return ColorRGB32F(0.0f);
		}
		else
			return ColorRGB32F(0.0f);

		if (hippt::dot(output_direction, bsdf_context.geometric_normal) < 0.0f)
			return ColorRGB32F(0.0f);

		if constexpr (sampleDirectionOnly)
		{
			pdf = 0.0f;
			return ColorRGB32F(0.0f);
		}

		bsdf_context.to_light_direction = output_direction;
		if constexpr (family == KernelMaterialSpecializationSingleMetallic)
			return principled_specialized_metallic_eval(render_data, bsdf_context, pdf, random_number_generator);
		else if constexpr (family == KernelMaterialSpecializationSpecularDiffuse)
			return principled_specialized_specular_diffuse_eval(render_data, bsdf_context, pdf, random_number_generator);
		else
			return ColorRGB32F(0.0f);
	}
}

#endif // #ifndef DEVICE_PRINCIPLED_SPECIALIZED_H
