/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_MATERIAL_H
#define DEVICE_MATERIAL_H

#include "Device/includes/Texture.h"

#include "Device/includes/HitInfo.h"
#include "HostDeviceCommon/Material/MaterialPacked.h"
#include "HostDeviceCommon/Material/LightProposalState.h"
#include "HostDeviceCommon/Material/MaterialUtils.h"
#include "HostDeviceCommon/Material/PrincipledLobeClassification.h"
#include "HostDeviceCommon/RenderData.h"

#include <type_traits>

#ifndef __KERNELCC__
#include "Image/Image.h"
#endif

template <typename T>
HIPRT_DEVICE static T read_material_texture(const HIPRTRenderData& render_data, const float2_t& texcoords, int texture_index, bool is_srgb);
HIPRT_DEVICE static float2_t get_metallic_roughness(const HIPRTRenderData& render_data,
													const float2_t& texcoords,
													int metallic_texture_index,
													int roughness_texture_index,
													int metallic_roughness_texture_index);
HIPRT_DEVICE static ColorRGB32F get_base_color(const HIPRTRenderData& render_data, float& out_alpha, const float2_t& texcoords, int base_color_texture_index);

HIPRT_DEVICE static float get_hit_base_color_alpha(const HIPRTRenderData& render_data, unsigned short int base_color_texture_index, int prim_id, float2_t uv)
{
	if (base_color_texture_index == MaterialConstants::NO_TEXTURE)
		// Quick exit if no texture
		return 1.0f;

	float2_t texcoords = uv_interpolate(render_data.buffers.triangles_indices, prim_id, render_data.buffers.texcoords, uv);

	// Getting the alpha for transparency check to see if we need to pass the ray through or not
	float alpha;
	get_base_color(render_data, alpha, texcoords, base_color_texture_index);

	return alpha;
}

HIPRT_DEVICE static float get_hit_base_color_alpha(const HIPRTRenderData& render_data, const DevicePackedTexturedMaterial& material, hiprtHit hit)
{
	return get_hit_base_color_alpha(render_data, material.get_base_color_texture_index(), hit.primID, hit.uv);
}

HIPRT_DEVICE static float get_hit_base_color_alpha(const HIPRTRenderData& render_data, int prim_id, float2_t uv)
{
	int material_index							= render_data.buffers.material_indices[prim_id];
	unsigned short int base_color_texture_index = render_data.buffers.materials_buffer_soa.get_base_color_texture_index(material_index);

	return get_hit_base_color_alpha(render_data, base_color_texture_index, prim_id, uv);
}

HIPRT_DEVICE static float get_hit_base_color_alpha(const HIPRTRenderData& render_data, hiprtHit hit)
{
	int material_index							= render_data.buffers.material_indices[hit.primID];
	unsigned short int base_color_texture_index = render_data.buffers.materials_buffer_soa.get_base_color_texture_index(material_index);

	return get_hit_base_color_alpha(render_data, base_color_texture_index, hit.primID, hit.uv);
}

HIPRT_DEVICE static unsigned char get_intersection_dielectric_priority(const HIPRTRenderData& render_data, int material_index)
{
#if BSDF_MODEL == BSDF_LAMBERTIAN || BSDF_MODEL == BSDF_OREN_NAYAR
	return 0;
#else
	return render_data.buffers.materials_buffer_soa.get_dielectric_priority(material_index);
#endif // #if BSDF_MODEL == BSDF_LAMBERTIAN || BSDF_MODEL == BSDF_OREN_NAYAR
}

struct SurfaceTransportMetadata
{
	float dispersion_scale;
	float specular_transmission;
	unsigned char dielectric_priority;
};

HIPRT_DEVICE static SurfaceTransportMetadata load_surface_transport_metadata(const DevicePackedEffectiveMaterial& packed_material,
																			 const PrincipledMaterialClassificationInputs& classification_inputs)
{
	SurfaceTransportMetadata metadata;
	metadata.dispersion_scale	   = classification_inputs.dispersion_scale;
	metadata.specular_transmission = classification_inputs.lobe_user_weights.specular_transmission;
#if BSDF_MODEL == BSDF_LAMBERTIAN || BSDF_MODEL == BSDF_OREN_NAYAR
	metadata.dielectric_priority = 0;
#else
	metadata.dielectric_priority = packed_material.get_dielectric_priority();
#endif // #if BSDF_MODEL == BSDF_LAMBERTIAN || BSDF_MODEL == BSDF_OREN_NAYAR
	return metadata;
}

HIPRT_DEVICE static SurfaceTransportMetadata load_surface_transport_metadata(const HIPRTRenderData& render_data,
																			 int material_index,
																			 const PrincipledMaterialClassificationInputs& classification_inputs)
{
	SurfaceTransportMetadata metadata;
	metadata.dispersion_scale	   = classification_inputs.dispersion_scale;
	metadata.specular_transmission = classification_inputs.lobe_user_weights.specular_transmission;
	metadata.dielectric_priority   = get_intersection_dielectric_priority(render_data, material_index);
	return metadata;
}

HIPRT_DEVICE static ReGIRMaterialInputs load_regir_material_inputs(const DevicePackedEffectiveMaterial& packed_material)
{
	ReGIRMaterialInputs inputs;
	inputs.roughness = packed_material.get_roughness();
	inputs.metallic	 = packed_material.get_metallic();
	inputs.specular	 = packed_material.get_specular();
	return inputs;
}

HIPRT_DEVICE static ReGIRMaterialInputs load_regir_material_inputs(const HIPRTRenderData& render_data,
																   int material_index,
																   const ResolvedMaterialUserControlsCache& resolved_user_controls,
																   const PrincipledMaterialClassificationInputs& classification_inputs)
{
	const DevicePackedTexturedMaterialSoA& materials_buffer_soa = render_data.buffers.materials_buffer_soa;
	ReGIRMaterialInputs inputs;
	inputs.roughness = materials_buffer_soa.get_roughness(material_index);
	if (resolved_user_controls.validity_mask & ResolvedMaterialUserControlRoughness)
		inputs.roughness = resolved_user_controls.roughness;
	inputs.metallic = classification_inputs.lobe_user_weights.metallic;
	inputs.specular = classification_inputs.lobe_user_weights.specular;

	float coat			  = classification_inputs.lobe_user_weights.coat;
	float coat_roughening = materials_buffer_soa.get_coat_roughening(material_index);
	if (coat > 0.0f && coat_roughening > 0.0f)
	{
		float base_roughness		   = inputs.roughness;
		float coat_roughness		   = materials_buffer_soa.get_coat_roughness(material_index);
		float target_base_roughness	   = hippt::pow_1_4(hippt::min(1.0f, hippt::pow_4(base_roughness) + 2.0f * hippt::pow_4(coat_roughness)));
		float roughened_base_roughness = hippt::lerp(base_roughness, target_base_roughness, coat);
		inputs.roughness			   = hippt::lerp(base_roughness, roughened_base_roughness, coat_roughening);
	}

	return inputs;
}

HIPRT_DEVICE static DeviceUnpackedPrincipledFullMaterial get_intersection_material(const HIPRTRenderData& render_data, int material_index, float2_t texcoords)
{
	DeviceUnpackedTexturedMaterial material = render_data.buffers.materials_buffer_soa.read_partial_material(material_index).unpack();

	float trash_alpha;
	if (render_data.bsdfs_data.white_furnace_mode)
		material.base_color = ColorRGB32F(1.0f);
	else
	{
#if UseMaterialTextures == KERNEL_OPTION_TRUE || UseMaterialBaseColorTextureOverride == KERNEL_OPTION_TRUE
		if (material.base_color_texture_index != MaterialConstants::NO_TEXTURE)
			material.base_color = get_base_color(render_data, trash_alpha, texcoords, material.base_color_texture_index);
#endif
	}

	// Reading some parameters from the textures
#if UseMaterialTextures == KERNEL_OPTION_TRUE
	float2_t roughness_metallic = get_metallic_roughness(render_data, texcoords, material.metallic_texture_index, material.roughness_texture_index,
														 material.roughness_metallic_texture_index);
	if (material.roughness_metallic_texture_index != MaterialConstants::NO_TEXTURE)
	{
		// Merged roughness metallic texture

		material.roughness = roughness_metallic.x;
		material.metallic  = roughness_metallic.y;
	}
	else
	{
		// Separate roughness / metallic texture

		if (material.roughness_texture_index != MaterialConstants::NO_TEXTURE)
			// If we have a roughness texture, it will already have been read above by get_metallic_roughness
			material.roughness = roughness_metallic.x;

		if (material.metallic_texture_index != MaterialConstants::NO_TEXTURE)
			// If we have a roughness texture, it will already have been read above by get_metallic_roughness
			material.metallic = roughness_metallic.y;
	}

	float anisotropy = read_material_texture<float>(render_data, texcoords, material.anisotropic_texture_index, false);
	if (material.anisotropic_texture_index != MaterialConstants::NO_TEXTURE)
		material.anisotropy = anisotropy;

	float specular = read_material_texture<float>(render_data, texcoords, material.specular_texture_index, false);
	if (material.specular_texture_index != MaterialConstants::NO_TEXTURE)
		material.specular = specular;

	float coat = read_material_texture<float>(render_data, texcoords, material.coat_texture_index, false);
	if (material.coat_texture_index != MaterialConstants::NO_TEXTURE)
		material.coat = coat;

	float sheen = read_material_texture<float>(render_data, texcoords, material.sheen_texture_index, false);
	if (material.sheen_texture_index != MaterialConstants::NO_TEXTURE)
		material.sheen = sheen;

	float specular_transmission = read_material_texture<float>(render_data, texcoords, material.specular_transmission_texture_index, false);
	if (material.specular_transmission_texture_index != MaterialConstants::NO_TEXTURE)
		material.specular_transmission = specular_transmission;
#endif // #if UseMaterialTextures == KERNEL_OPTION_TRUE

	ColorRGB32F emission;
	if (material.emission_texture_index == MaterialConstants::NO_TEXTURE || material.emission_texture_index == MaterialConstants::CONSTANT_EMISSIVE_TEXTURE)
		emission = material.get_raw_emission();
	else
		emission = read_material_texture<ColorRGB32F>(render_data, texcoords, material.emission_texture_index, false);

	DeviceUnpackedPrincipledFullMaterial unpacked_effective_material(material);
	unpacked_effective_material.base_color = material.base_color;

	unpacked_effective_material.set_raw_emission(emission);
	unpacked_effective_material.set_emission_strength(material.get_emission_strength());
	// Roughening of the base roughness and second metallic roughness based
	// on the coat roughness. This should be precomputed instead of being done here
	//
	// Reference: [OpenPBR Surface 2024 Specification] https://academysoftwarefoundation.github.io/OpenPBR/#model/coat/roughening
	float coat_roughening = unpacked_effective_material.coat_roughening;
	if (material.coat > 0.0f && coat_roughening > 0.0f)
	{
		float base_roughness = material.roughness;
		float coat_roughness = unpacked_effective_material.coat_roughness;

		// Roughening of the base roughness of the material based on the coat roughness
		float target_base_roughness			  = hippt::pow_1_4(hippt::min(1.0f, hippt::pow_4(base_roughness) + 2.0f * hippt::pow_4(coat_roughness)));
		float roughened_base_roughness		  = hippt::lerp(base_roughness, target_base_roughness, material.coat);
		unpacked_effective_material.roughness = hippt::lerp(base_roughness, roughened_base_roughness, coat_roughening);

		if (unpacked_effective_material.second_roughness_weight > 0.0f)
		{
			// Roughening of the second metallic roughness based on the coat roughness

			float second_roughness				   = unpacked_effective_material.second_roughness;
			float target_second_metal_roughness	   = hippt::pow_1_4(hippt::min(1.0f, hippt::pow_4(second_roughness) + 2.0f * hippt::pow_4(coat_roughness)));
			float roughened_second_metal_roughness = hippt::lerp(second_roughness, target_second_metal_roughness, material.coat);
			unpacked_effective_material.second_roughness = hippt::lerp(second_roughness, roughened_second_metal_roughness, coat_roughening);
		}
	}

	return unpacked_effective_material;
}

/**
 * The float2_t returned is (roughness, metallic)
 */
HIPRT_DEVICE static float2_t get_metallic_roughness(const HIPRTRenderData& render_data,
													const float2_t& texcoords,
													int metallic_texture_index,
													int roughness_texture_index,
													int metallic_roughness_texture_index)
{
	float2_t out;

	if (metallic_roughness_texture_index != MaterialConstants::NO_TEXTURE)
	{
		ColorRGB32F rgb = sample_texture_rgb_8bits(render_data.buffers.material_textures, texcoords, metallic_roughness_texture_index, false);

		// Not converting to linear here because material properties (roughness and metallic) here are assumed to be linear already
		out.x = rgb.g;
		out.y = rgb.b;
	}
	else
	{
		out.x = read_material_texture<float>(render_data, texcoords, roughness_texture_index, false);
		out.y = read_material_texture<float>(render_data, texcoords, metallic_texture_index, false);
	}

	return out;
}

HIPRT_DEVICE static ColorRGB32F get_base_color(const HIPRTRenderData& render_data, float& out_alpha, const float2_t& texcoords, int base_color_texture_index)
{
	out_alpha		  = 1.0f;
	ColorRGBA32F rgba = read_material_texture<ColorRGBA32F>(render_data, texcoords, base_color_texture_index, true);
	if (base_color_texture_index != MaterialConstants::NO_TEXTURE)
	{
		ColorRGB32F base_color = ColorRGB32F(rgba.r, rgba.g, rgba.b);
		out_alpha			   = rgba.a;

		return base_color;
	}

	return ColorRGB32F();
}

template <typename T>
HIPRT_DEVICE static T read_data(const ColorRGBA32F& rgba)
{
}

template <>
HIPRT_DEVICE ColorRGBA32F read_data<ColorRGBA32F>(const ColorRGBA32F& rgba)
{
	return rgba;
}

template <>
HIPRT_DEVICE ColorRGB32F read_data<ColorRGB32F>(const ColorRGBA32F& rgba)
{
	return ColorRGB32F(rgba.r, rgba.g, rgba.b);
}

template <>
HIPRT_DEVICE float read_data<float>(const ColorRGBA32F& rgba)
{
	return rgba.r;
}

template <typename T>
HIPRT_DEVICE static T read_material_texture(const HIPRTRenderData& render_data, const float2_t& texcoords, int texture_index, bool is_srgb)
{
	if (texture_index == MaterialConstants::NO_TEXTURE || texture_index == MaterialConstants::CONSTANT_EMISSIVE_TEXTURE)
		return T();

	ColorRGBA32F rgba = sample_texture_rgba(render_data.buffers.material_textures, texcoords, texture_index, is_srgb);
	return read_data<T>(rgba);
}

HIPRT_DEVICE static PrincipledMaterialClassificationInputs load_material_classification_inputs(const DevicePackedEffectiveMaterial& packed_material,
																							   ResolvedMaterialUserControlsCache& out_resolved_user_controls)
{
	PrincipledMaterialClassificationInputs inputs;
	inputs.lobe_user_weights.coat					 = packed_material.get_coat();
	inputs.lobe_user_weights.sheen					 = packed_material.get_sheen();
	inputs.lobe_user_weights.metallic				 = packed_material.get_metallic();
	inputs.lobe_user_weights.second_roughness_weight = packed_material.get_second_roughness_weight();
	inputs.lobe_user_weights.retro_reflection		 = packed_material.get_retro_reflection();
	inputs.lobe_user_weights.specular				 = packed_material.get_specular();
	inputs.lobe_user_weights.specular_transmission	 = packed_material.get_specular_transmission();
	inputs.lobe_user_weights.diffuse_transmission	 = packed_material.get_diffuse_transmission();
	inputs.thin_film_strength						 = packed_material.get_thin_film();
	inputs.dispersion_scale							 = packed_material.get_dispersion_scale();
	inputs.thin_walled								 = packed_material.get_thin_walled();
	inputs.enforce_strong_energy_conservation		 = packed_material.get_enforce_strong_energy_conservation();

	out_resolved_user_controls.roughness			 = packed_material.get_roughness();
	out_resolved_user_controls.metallic				 = inputs.lobe_user_weights.metallic;
	out_resolved_user_controls.specular				 = inputs.lobe_user_weights.specular;
	out_resolved_user_controls.coat					 = inputs.lobe_user_weights.coat;
	out_resolved_user_controls.sheen				 = inputs.lobe_user_weights.sheen;
	out_resolved_user_controls.specular_transmission = inputs.lobe_user_weights.specular_transmission;
	out_resolved_user_controls.validity_mask		 = ResolvedMaterialUserControlRoughness | ResolvedMaterialUserControlMetallic |
											   ResolvedMaterialUserControlSpecular | ResolvedMaterialUserControlCoat | ResolvedMaterialUserControlSheen |
											   ResolvedMaterialUserControlSpecularTransmission;

	return inputs;
}

HIPRT_DEVICE static PrincipledMaterialClassificationInputs load_material_classification_inputs(const HIPRTRenderData& render_data,
																							   int material_index,
																							   float2_t texcoords,
																							   ResolvedMaterialUserControlsCache& out_resolved_user_controls)
{
	const DevicePackedTexturedMaterialSoA& materials_buffer_soa = render_data.buffers.materials_buffer_soa;
	PrincipledMaterialClassificationInputs inputs;
	inputs.lobe_user_weights.coat					 = materials_buffer_soa.get_coat(material_index);
	inputs.lobe_user_weights.sheen					 = materials_buffer_soa.get_sheen(material_index);
	inputs.lobe_user_weights.metallic				 = materials_buffer_soa.get_metallic(material_index);
	inputs.lobe_user_weights.second_roughness_weight = materials_buffer_soa.get_second_roughness_weight(material_index);
	inputs.lobe_user_weights.retro_reflection		 = materials_buffer_soa.get_retro_reflection(material_index);
	inputs.lobe_user_weights.specular				 = materials_buffer_soa.get_specular(material_index);
	inputs.lobe_user_weights.specular_transmission	 = materials_buffer_soa.get_specular_transmission(material_index);
	inputs.lobe_user_weights.diffuse_transmission	 = materials_buffer_soa.get_diffuse_transmission(material_index);
	inputs.thin_film_strength						 = materials_buffer_soa.get_thin_film(material_index);
	inputs.dispersion_scale							 = materials_buffer_soa.get_dispersion_scale(material_index);
	inputs.thin_walled								 = materials_buffer_soa.get_thin_walled(material_index);
	inputs.enforce_strong_energy_conservation		 = materials_buffer_soa.get_enforce_strong_energy_conservation(material_index);

	out_resolved_user_controls = ResolvedMaterialUserControlsCache();

#if UseMaterialTextures == KERNEL_OPTION_TRUE
	unsigned short int roughness_metallic_texture_index = materials_buffer_soa.get_roughness_metallic_texture_index(material_index);
	unsigned short int roughness_texture_index			= materials_buffer_soa.get_roughness_texture_index(material_index);
	unsigned short int metallic_texture_index			= materials_buffer_soa.get_metallic_texture_index(material_index);
	float2_t roughness_metallic =
		get_metallic_roughness(render_data, texcoords, metallic_texture_index, roughness_texture_index, roughness_metallic_texture_index);

	if (roughness_metallic_texture_index != MaterialConstants::NO_TEXTURE)
	{
		out_resolved_user_controls.roughness = roughness_metallic.x;
		out_resolved_user_controls.metallic	 = roughness_metallic.y;
		out_resolved_user_controls.validity_mask |= ResolvedMaterialUserControlRoughness | ResolvedMaterialUserControlMetallic;
	}
	else
	{
		if (roughness_texture_index != MaterialConstants::NO_TEXTURE)
		{
			out_resolved_user_controls.roughness = roughness_metallic.x;
			out_resolved_user_controls.validity_mask |= ResolvedMaterialUserControlRoughness;
		}

		if (metallic_texture_index != MaterialConstants::NO_TEXTURE)
		{
			out_resolved_user_controls.metallic = roughness_metallic.y;
			out_resolved_user_controls.validity_mask |= ResolvedMaterialUserControlMetallic;
		}
	}

	if (out_resolved_user_controls.validity_mask & ResolvedMaterialUserControlMetallic)
		inputs.lobe_user_weights.metallic = out_resolved_user_controls.metallic;

	unsigned short int specular_texture_index = materials_buffer_soa.get_specular_texture_index(material_index);
	if (specular_texture_index != MaterialConstants::NO_TEXTURE)
	{
		out_resolved_user_controls.specular = read_material_texture<float>(render_data, texcoords, specular_texture_index, false);
		out_resolved_user_controls.validity_mask |= ResolvedMaterialUserControlSpecular;
		inputs.lobe_user_weights.specular = out_resolved_user_controls.specular;
	}

	unsigned short int coat_texture_index = materials_buffer_soa.get_coat_texture_index(material_index);
	if (coat_texture_index != MaterialConstants::NO_TEXTURE)
	{
		out_resolved_user_controls.coat = read_material_texture<float>(render_data, texcoords, coat_texture_index, false);
		out_resolved_user_controls.validity_mask |= ResolvedMaterialUserControlCoat;
		inputs.lobe_user_weights.coat = out_resolved_user_controls.coat;
	}

	unsigned short int sheen_texture_index = materials_buffer_soa.get_sheen_texture_index(material_index);
	if (sheen_texture_index != MaterialConstants::NO_TEXTURE)
	{
		out_resolved_user_controls.sheen = read_material_texture<float>(render_data, texcoords, sheen_texture_index, false);
		out_resolved_user_controls.validity_mask |= ResolvedMaterialUserControlSheen;
		inputs.lobe_user_weights.sheen = out_resolved_user_controls.sheen;
	}

	unsigned short int specular_transmission_texture_index = materials_buffer_soa.get_specular_transmission_texture_index(material_index);
	if (specular_transmission_texture_index != MaterialConstants::NO_TEXTURE)
	{
		out_resolved_user_controls.specular_transmission = read_material_texture<float>(render_data, texcoords, specular_transmission_texture_index, false);
		out_resolved_user_controls.validity_mask |= ResolvedMaterialUserControlSpecularTransmission;
		inputs.lobe_user_weights.specular_transmission = out_resolved_user_controls.specular_transmission;
	}
#endif // #if UseMaterialTextures == KERNEL_OPTION_TRUE

	return inputs;
}

template <bool initialize_material_defaults>
HIPRT_DEVICE static void get_intersection_material_into_impl(const HIPRTRenderData& render_data,
															 int material_index,
															 float2_t texcoords,
															 DeviceUnpackedPrincipledFullMaterial& out_material,
															 const ResolvedMaterialUserControlsCache* resolved_user_controls = nullptr)
{
	const DevicePackedTexturedMaterialSoA& materials_buffer_soa = render_data.buffers.materials_buffer_soa;
	if constexpr (initialize_material_defaults)
		materials_buffer_soa.read_partial_effective_material(material_index, out_material);
	else
		materials_buffer_soa.read_partial_effective_material_noinit(material_index, out_material);

	if (render_data.bsdfs_data.white_furnace_mode)
		out_material.base_color = ColorRGB32F(1.0f);
	else
	{
#if UseMaterialTextures == KERNEL_OPTION_TRUE || UseMaterialBaseColorTextureOverride == KERNEL_OPTION_TRUE
		unsigned int base_color_texture_index = materials_buffer_soa.get_base_color_texture_index(material_index);
		if (base_color_texture_index != MaterialConstants::NO_TEXTURE)
		{
			float trash_alpha;
			out_material.base_color = get_base_color(render_data, trash_alpha, texcoords, base_color_texture_index);
		}
#endif // #if UseMaterialTextures == KERNEL_OPTION_TRUE || UseMaterialBaseColorTextureOverride == KERNEL_OPTION_TRUE
	}

	// Reading some parameters from the textures
#if UseMaterialTextures == KERNEL_OPTION_TRUE
	{
		if (resolved_user_controls != nullptr)
		{
			if (resolved_user_controls->validity_mask & ResolvedMaterialUserControlRoughness)
				out_material.roughness = resolved_user_controls->roughness;
			if (resolved_user_controls->validity_mask & ResolvedMaterialUserControlMetallic)
				out_material.metallic = resolved_user_controls->metallic;
		}
		else
		{
			unsigned int roughness_metallic_texture_index = materials_buffer_soa.get_roughness_metallic_texture_index(material_index);
			unsigned int roughness_texture_index		  = materials_buffer_soa.get_roughness_texture_index(material_index);
			unsigned int metallic_texture_index			  = materials_buffer_soa.get_metallic_texture_index(material_index);

			float2_t roughness_metallic =
				get_metallic_roughness(render_data, texcoords, metallic_texture_index, roughness_texture_index, roughness_metallic_texture_index);
			if (roughness_metallic_texture_index != MaterialConstants::NO_TEXTURE)
			{
				// Merged roughness metallic texture
				out_material.roughness = roughness_metallic.x;
				out_material.metallic  = roughness_metallic.y;
			}
			else
			{
				// Separate roughness / metallic texture
				if (roughness_texture_index != MaterialConstants::NO_TEXTURE)
					out_material.roughness = roughness_metallic.x;

				if (metallic_texture_index != MaterialConstants::NO_TEXTURE)
					out_material.metallic = roughness_metallic.y;
			}
		}
	}

	{
		unsigned int anisotropic_texture_index = materials_buffer_soa.get_anisotropic_texture_index(material_index);
		if (anisotropic_texture_index != MaterialConstants::NO_TEXTURE)
			out_material.anisotropy = read_material_texture<float>(render_data, texcoords, anisotropic_texture_index, false);
	}

	{
		if (resolved_user_controls != nullptr && (resolved_user_controls->validity_mask & ResolvedMaterialUserControlSpecular))
			out_material.specular = resolved_user_controls->specular;
		else
		{
			unsigned int specular_texture_index = materials_buffer_soa.get_specular_texture_index(material_index);
			if (specular_texture_index != MaterialConstants::NO_TEXTURE)
				out_material.specular = read_material_texture<float>(render_data, texcoords, specular_texture_index, false);
		}
	}

	{
		if (resolved_user_controls != nullptr && (resolved_user_controls->validity_mask & ResolvedMaterialUserControlCoat))
			out_material.coat = resolved_user_controls->coat;
		else
		{
			unsigned int coat_texture_index = materials_buffer_soa.get_coat_texture_index(material_index);
			if (coat_texture_index != MaterialConstants::NO_TEXTURE)
				out_material.coat = read_material_texture<float>(render_data, texcoords, coat_texture_index, false);
		}
	}

	{
		if (resolved_user_controls != nullptr && (resolved_user_controls->validity_mask & ResolvedMaterialUserControlSheen))
			out_material.sheen = resolved_user_controls->sheen;
		else
		{
			unsigned int sheen_texture_index = materials_buffer_soa.get_sheen_texture_index(material_index);
			if (sheen_texture_index != MaterialConstants::NO_TEXTURE)
				out_material.sheen = read_material_texture<float>(render_data, texcoords, sheen_texture_index, false);
		}
	}

	{
		if (resolved_user_controls != nullptr && (resolved_user_controls->validity_mask & ResolvedMaterialUserControlSpecularTransmission))
			out_material.specular_transmission = resolved_user_controls->specular_transmission;
		else
		{
			unsigned int specular_transmission_texture_index = materials_buffer_soa.get_specular_transmission_texture_index(material_index);
			if (specular_transmission_texture_index != MaterialConstants::NO_TEXTURE)
				out_material.specular_transmission = read_material_texture<float>(render_data, texcoords, specular_transmission_texture_index, false);
		}
	}
#endif // #if UseMaterialTextures == KERNEL_OPTION_TRUE

	{
		unsigned int emission_texture_index = materials_buffer_soa.get_emission_texture_index(material_index);
		if (emission_texture_index != MaterialConstants::NO_TEXTURE && emission_texture_index != MaterialConstants::CONSTANT_EMISSIVE_TEXTURE)
			out_material.set_raw_emission(read_material_texture<ColorRGB32F>(render_data, texcoords, emission_texture_index, false));
	}

	// Roughening of the base roughness and second metallic roughness based
	// on the coat roughness. This should be precomputed instead of being done here
	//
	// Reference: [OpenPBR Surface 2024 Specification] https://academysoftwarefoundation.github.io/OpenPBR/#model/coat/roughening
	float coat_roughening = out_material.coat_roughening;
	if (out_material.coat > 0.0f && coat_roughening > 0.0f)
	{
		float base_roughness = out_material.roughness;
		float coat_roughness = out_material.coat_roughness;

		// Roughening of the base roughness of the material based on the coat roughness
		float target_base_roughness	   = hippt::pow_1_4(hippt::min(1.0f, hippt::pow_4(base_roughness) + 2.0f * hippt::pow_4(coat_roughness)));
		float roughened_base_roughness = hippt::lerp(base_roughness, target_base_roughness, out_material.coat);
		out_material.roughness		   = hippt::lerp(base_roughness, roughened_base_roughness, coat_roughening);

		if (out_material.second_roughness_weight > 0.0f)
		{
			// Roughening of the second metallic roughness based on the coat roughness
			float second_roughness				   = out_material.second_roughness;
			float target_second_metal_roughness	   = hippt::pow_1_4(hippt::min(1.0f, hippt::pow_4(second_roughness) + 2.0f * hippt::pow_4(coat_roughness)));
			float roughened_second_metal_roughness = hippt::lerp(second_roughness, target_second_metal_roughness, out_material.coat);
			out_material.second_roughness		   = hippt::lerp(second_roughness, roughened_second_metal_roughness, coat_roughening);
		}
	}
}

HIPRT_DEVICE static void get_intersection_material_into(const HIPRTRenderData& render_data,
														int material_index,
														float2_t texcoords,
														DeviceUnpackedPrincipledFullMaterial& out_material)
{
	get_intersection_material_into_impl<true>(render_data, material_index, texcoords, out_material);
}

HIPRT_DEVICE static void get_intersection_material_into_noinit(const HIPRTRenderData& render_data,
															   int material_index,
															   float2_t texcoords,
															   DeviceUnpackedPrincipledFullMaterial& out_material)
{
	get_intersection_material_into_impl<false>(render_data, material_index, texcoords, out_material);
}

HIPRT_DEVICE static void load_effective_emission(const DevicePackedEffectiveMaterial& packed_material, EffectiveMaterialEmission& out_emission)
{
	out_emission.emission		= packed_material.get_raw_emission() * packed_material.get_emission_strength();
	out_emission.emission_flags = packed_material.is_emissive() ? EffectiveMaterialEmissionIsEmissive : 0u;
}

HIPRT_DEVICE static void load_effective_emission(const HIPRTRenderData& render_data,
												 int material_index,
												 float2_t texcoords,
												 EffectiveMaterialEmission& out_emission)
{
	const DevicePackedTexturedMaterialSoA& materials_buffer_soa = render_data.buffers.materials_buffer_soa;

	ColorRGB32F raw_emission = materials_buffer_soa.get_emission(material_index);
	float emission_strength	 = materials_buffer_soa.get_emission_strength(material_index);

	unsigned short int emission_texture_index = materials_buffer_soa.get_emission_texture_index(material_index);
	bool uses_emission_texture =
		emission_texture_index != MaterialConstants::NO_TEXTURE && emission_texture_index != MaterialConstants::CONSTANT_EMISSIVE_TEXTURE;

	if (uses_emission_texture)
		raw_emission = read_material_texture<ColorRGB32F>(render_data, texcoords, emission_texture_index, false);

	bool emissive = !hippt::is_zero(raw_emission.r) || !hippt::is_zero(raw_emission.g) || !hippt::is_zero(raw_emission.b) || uses_emission_texture;

	out_emission.emission		= raw_emission * emission_strength;
	out_emission.emission_flags = emissive ? EffectiveMaterialEmissionIsEmissive : 0u;
}

template <typename MaterialType>
HIPRT_DEVICE static void load_effective_material(const DevicePackedEffectiveMaterial& packed_material, MaterialType& out_material)
{
	if constexpr (std::is_same_v<MaterialType, DeviceUnpackedPrincipledFullMaterial>)
		out_material = packed_material.unpack();
	else
	{
		load_effective_emission(packed_material, out_material);

		out_material.base_color = packed_material.get_base_color();

		if constexpr (std::is_same_v<MaterialType, DeviceOrenNayarMaterial>)
			out_material.oren_nayar_sigma = packed_material.get_oren_nayar_sigma();
#if PrincipledBSDFDiffuseLobe == PRINCIPLED_DIFFUSE_LOBE_OREN_NAYAR
		else if constexpr (std::is_same_v<MaterialType, DevicePrincipledDiffuseMaterial>)
			out_material.oren_nayar_sigma = packed_material.get_oren_nayar_sigma();
#endif
		else if constexpr (std::is_same_v<MaterialType, DevicePrincipledGlassMaterial>)
		{
			out_material.roughness					  = packed_material.get_roughness();
			out_material.anisotropy					  = packed_material.get_anisotropy();
			out_material.anisotropy_rotation		  = packed_material.get_anisotropy_rotation();
			out_material.ior						  = packed_material.get_ior();
			out_material.do_glass_energy_compensation = packed_material.get_do_glass_energy_compensation();
		}
		else if constexpr (std::is_same_v<MaterialType, DevicePrincipledSingleMetallicMaterial>)
		{
			out_material.roughness						 = packed_material.get_roughness();
			out_material.anisotropy						 = packed_material.get_anisotropy();
			out_material.anisotropy_rotation			 = packed_material.get_anisotropy_rotation();
			out_material.metallic_F82					 = packed_material.get_metallic_F82();
			out_material.metallic_F90					 = packed_material.get_metallic_F90();
			out_material.metallic_F90_falloff_exponent	 = packed_material.get_metallic_F90_falloff_exponent();
			out_material.do_metallic_energy_compensation = packed_material.get_do_metallic_energy_compensation();
		}
		else if constexpr (std::is_same_v<MaterialType, DevicePrincipledSpecularDiffuseMaterial>)
		{
#if PrincipledBSDFDiffuseLobe == PRINCIPLED_DIFFUSE_LOBE_OREN_NAYAR
			out_material.oren_nayar_sigma = packed_material.get_oren_nayar_sigma();
#endif
			out_material.roughness						 = packed_material.get_roughness();
			out_material.anisotropy						 = packed_material.get_anisotropy();
			out_material.anisotropy_rotation			 = packed_material.get_anisotropy_rotation();
			out_material.ior							 = packed_material.get_ior();
			out_material.specular						 = packed_material.get_specular();
			out_material.specular_tint					 = packed_material.get_specular_tint();
			out_material.specular_color					 = packed_material.get_specular_color();
			out_material.specular_darkening				 = packed_material.get_specular_darkening();
			out_material.do_specular_energy_compensation = packed_material.get_do_specular_energy_compensation();
		}
	}
}

template <typename MaterialType>
HIPRT_DEVICE static void load_effective_material(const HIPRTRenderData& render_data,
												 int material_index,
												 float2_t texcoords,
												 const ResolvedMaterialUserControlsCache& resolved_user_controls,
												 MaterialType& out_material)
{
	if constexpr (std::is_same_v<MaterialType, DeviceUnpackedPrincipledFullMaterial>)
		get_intersection_material_into_impl<true>(render_data, material_index, texcoords, out_material, &resolved_user_controls);
	else
	{
		const DevicePackedTexturedMaterialSoA& materials_buffer_soa = render_data.buffers.materials_buffer_soa;
		load_effective_emission(render_data, material_index, texcoords, out_material);

		if (render_data.bsdfs_data.white_furnace_mode)
			out_material.base_color = ColorRGB32F(1.0f);
		else
		{
#if UseMaterialTextures == KERNEL_OPTION_TRUE || UseMaterialBaseColorTextureOverride == KERNEL_OPTION_TRUE
			unsigned short int base_color_texture_index = materials_buffer_soa.get_base_color_texture_index(material_index);
			if (base_color_texture_index != MaterialConstants::NO_TEXTURE)
			{
				float trash_alpha;
				out_material.base_color = get_base_color(render_data, trash_alpha, texcoords, base_color_texture_index);
			}
			else
#endif // #if UseMaterialTextures == KERNEL_OPTION_TRUE || UseMaterialBaseColorTextureOverride == KERNEL_OPTION_TRUE
				out_material.base_color = materials_buffer_soa.get_base_color(material_index);
		}

		if constexpr (std::is_same_v<MaterialType, DeviceOrenNayarMaterial>)
			out_material.oren_nayar_sigma = materials_buffer_soa.get_oren_nayar_sigma(material_index);
#if PrincipledBSDFDiffuseLobe == PRINCIPLED_DIFFUSE_LOBE_OREN_NAYAR
		else if constexpr (std::is_same_v<MaterialType, DevicePrincipledDiffuseMaterial>)
			out_material.oren_nayar_sigma = materials_buffer_soa.get_oren_nayar_sigma(material_index);
#endif
		else if constexpr (std::is_same_v<MaterialType, DevicePrincipledGlassMaterial>)
		{
			out_material.roughness	= resolved_user_controls.validity_mask & ResolvedMaterialUserControlRoughness
										  ? resolved_user_controls.roughness
										  : materials_buffer_soa.get_roughness(material_index);
			out_material.anisotropy = materials_buffer_soa.get_anisotropy(material_index);
#if UseMaterialTextures == KERNEL_OPTION_TRUE
			unsigned short int anisotropic_texture_index = materials_buffer_soa.get_anisotropic_texture_index(material_index);
			if (anisotropic_texture_index != MaterialConstants::NO_TEXTURE)
				out_material.anisotropy = read_material_texture<float>(render_data, texcoords, anisotropic_texture_index, false);
#endif // #if UseMaterialTextures == KERNEL_OPTION_TRUE
			out_material.anisotropy_rotation		  = materials_buffer_soa.get_anisotropy_rotation(material_index);
			out_material.ior						  = materials_buffer_soa.get_ior(material_index);
			out_material.do_glass_energy_compensation = materials_buffer_soa.get_do_glass_energy_compensation(material_index);
		}
		else if constexpr (std::is_same_v<MaterialType, DevicePrincipledSingleMetallicMaterial>)
		{
			out_material.roughness	= resolved_user_controls.validity_mask & ResolvedMaterialUserControlRoughness
										  ? resolved_user_controls.roughness
										  : materials_buffer_soa.get_roughness(material_index);
			out_material.anisotropy = materials_buffer_soa.get_anisotropy(material_index);
#if UseMaterialTextures == KERNEL_OPTION_TRUE
			unsigned short int anisotropic_texture_index = materials_buffer_soa.get_anisotropic_texture_index(material_index);
			if (anisotropic_texture_index != MaterialConstants::NO_TEXTURE)
				out_material.anisotropy = read_material_texture<float>(render_data, texcoords, anisotropic_texture_index, false);
#endif // #if UseMaterialTextures == KERNEL_OPTION_TRUE
			out_material.anisotropy_rotation			 = materials_buffer_soa.get_anisotropy_rotation(material_index);
			out_material.metallic_F82					 = materials_buffer_soa.get_metallic_F82(material_index);
			out_material.metallic_F90					 = materials_buffer_soa.get_metallic_F90(material_index);
			out_material.metallic_F90_falloff_exponent	 = materials_buffer_soa.get_metallic_F90_falloff_exponent(material_index);
			out_material.do_metallic_energy_compensation = materials_buffer_soa.get_do_metallic_energy_compensation(material_index);
		}
		else if constexpr (std::is_same_v<MaterialType, DevicePrincipledSpecularDiffuseMaterial>)
		{
#if PrincipledBSDFDiffuseLobe == PRINCIPLED_DIFFUSE_LOBE_OREN_NAYAR
			out_material.oren_nayar_sigma = materials_buffer_soa.get_oren_nayar_sigma(material_index);
#endif
			out_material.roughness	= resolved_user_controls.validity_mask & ResolvedMaterialUserControlRoughness
										  ? resolved_user_controls.roughness
										  : materials_buffer_soa.get_roughness(material_index);
			out_material.anisotropy = materials_buffer_soa.get_anisotropy(material_index);
#if UseMaterialTextures == KERNEL_OPTION_TRUE
			unsigned short int anisotropic_texture_index = materials_buffer_soa.get_anisotropic_texture_index(material_index);
			if (anisotropic_texture_index != MaterialConstants::NO_TEXTURE)
				out_material.anisotropy = read_material_texture<float>(render_data, texcoords, anisotropic_texture_index, false);
#endif // #if UseMaterialTextures == KERNEL_OPTION_TRUE
			out_material.anisotropy_rotation			 = materials_buffer_soa.get_anisotropy_rotation(material_index);
			out_material.ior							 = materials_buffer_soa.get_ior(material_index);
			out_material.specular						 = resolved_user_controls.validity_mask & ResolvedMaterialUserControlSpecular
															   ? resolved_user_controls.specular
															   : materials_buffer_soa.get_specular(material_index);
			out_material.specular_tint					 = materials_buffer_soa.get_specular_tint(material_index);
			out_material.specular_color					 = materials_buffer_soa.get_specular_color(material_index);
			out_material.specular_darkening				 = materials_buffer_soa.get_specular_darkening(material_index);
			out_material.do_specular_energy_compensation = materials_buffer_soa.get_do_specular_energy_compensation(material_index);
		}
	}
}

HIPRT_DEVICE static LightProposalInputs load_light_proposal_inputs(const DevicePackedEffectiveMaterial& packed_material)
{
	LightProposalInputs proposal_inputs{};
	proposal_inputs.base_color_luminance  = packed_material.get_base_color().luminance();
	proposal_inputs.roughness			  = packed_material.get_roughness();
	proposal_inputs.ior					  = packed_material.get_ior();
	proposal_inputs.metallic			  = packed_material.get_metallic();
	proposal_inputs.specular			  = packed_material.get_specular();
	proposal_inputs.coat				  = packed_material.get_coat();
	proposal_inputs.coat_roughness		  = packed_material.get_coat_roughness();
	proposal_inputs.coat_ior			  = packed_material.get_coat_ior();
	proposal_inputs.specular_transmission = packed_material.get_specular_transmission();
	proposal_inputs.diffuse_transmission  = packed_material.get_diffuse_transmission();
	proposal_inputs.anisotropy			  = packed_material.get_anisotropy();
	proposal_inputs.coat_anisotropy		  = packed_material.get_coat_anisotropy();
	return proposal_inputs;
}

HIPRT_DEVICE static LightProposalInputs load_light_proposal_inputs(const HIPRTRenderData& render_data,
																   int material_index,
																   float2_t texcoords,
																   float base_color_luminance,
																   const ResolvedMaterialUserControlsCache& resolved_user_controls)
{
	const DevicePackedTexturedMaterialSoA& materials_buffer_soa = render_data.buffers.materials_buffer_soa;
	LightProposalInputs proposal_inputs{};
	proposal_inputs.base_color_luminance  = base_color_luminance;
	proposal_inputs.roughness			  = materials_buffer_soa.get_roughness(material_index);
	proposal_inputs.ior					  = materials_buffer_soa.get_ior(material_index);
	proposal_inputs.metallic			  = materials_buffer_soa.get_metallic(material_index);
	proposal_inputs.specular			  = materials_buffer_soa.get_specular(material_index);
	proposal_inputs.coat				  = materials_buffer_soa.get_coat(material_index);
	proposal_inputs.coat_roughness		  = materials_buffer_soa.get_coat_roughness(material_index);
	proposal_inputs.coat_ior			  = materials_buffer_soa.get_coat_ior(material_index);
	proposal_inputs.specular_transmission = materials_buffer_soa.get_specular_transmission(material_index);
	proposal_inputs.diffuse_transmission  = materials_buffer_soa.get_diffuse_transmission(material_index);
	proposal_inputs.anisotropy			  = materials_buffer_soa.get_anisotropy(material_index);
	proposal_inputs.coat_anisotropy		  = materials_buffer_soa.get_coat_anisotropy(material_index);

#if UseMaterialTextures == KERNEL_OPTION_TRUE
	if (resolved_user_controls.validity_mask & ResolvedMaterialUserControlRoughness)
		proposal_inputs.roughness = resolved_user_controls.roughness;
	if (resolved_user_controls.validity_mask & ResolvedMaterialUserControlMetallic)
		proposal_inputs.metallic = resolved_user_controls.metallic;

	unsigned int roughness_metallic_texture_index = materials_buffer_soa.get_roughness_metallic_texture_index(material_index);
	unsigned int roughness_texture_index		  = materials_buffer_soa.get_roughness_texture_index(material_index);
	unsigned int metallic_texture_index			  = materials_buffer_soa.get_metallic_texture_index(material_index);
	bool roughness_needs_texture				  = !(resolved_user_controls.validity_mask & ResolvedMaterialUserControlRoughness);
	bool metallic_needs_texture					  = !(resolved_user_controls.validity_mask & ResolvedMaterialUserControlMetallic);
	if ((roughness_needs_texture || metallic_needs_texture) &&
		(roughness_metallic_texture_index != MaterialConstants::NO_TEXTURE || roughness_texture_index != MaterialConstants::NO_TEXTURE ||
		 metallic_texture_index != MaterialConstants::NO_TEXTURE))
	{
		float2_t roughness_metallic =
			get_metallic_roughness(render_data, texcoords, metallic_texture_index, roughness_texture_index, roughness_metallic_texture_index);
		if (roughness_metallic_texture_index != MaterialConstants::NO_TEXTURE)
		{
			if (roughness_needs_texture)
				proposal_inputs.roughness = roughness_metallic.x;
			if (metallic_needs_texture)
				proposal_inputs.metallic = roughness_metallic.y;
		}
		else
		{
			if (roughness_needs_texture && roughness_texture_index != MaterialConstants::NO_TEXTURE)
				proposal_inputs.roughness = roughness_metallic.x;
			if (metallic_needs_texture && metallic_texture_index != MaterialConstants::NO_TEXTURE)
				proposal_inputs.metallic = roughness_metallic.y;
		}
	}

	if (resolved_user_controls.validity_mask & ResolvedMaterialUserControlSpecular)
		proposal_inputs.specular = resolved_user_controls.specular;
	else
	{
		unsigned int specular_texture_index = materials_buffer_soa.get_specular_texture_index(material_index);
		if (specular_texture_index != MaterialConstants::NO_TEXTURE)
			proposal_inputs.specular = read_material_texture<float>(render_data, texcoords, specular_texture_index, false);
	}

	if (resolved_user_controls.validity_mask & ResolvedMaterialUserControlCoat)
		proposal_inputs.coat = resolved_user_controls.coat;
	else
	{
		unsigned int coat_texture_index = materials_buffer_soa.get_coat_texture_index(material_index);
		if (coat_texture_index != MaterialConstants::NO_TEXTURE)
			proposal_inputs.coat = read_material_texture<float>(render_data, texcoords, coat_texture_index, false);
	}

	unsigned int anisotropic_texture_index = materials_buffer_soa.get_anisotropic_texture_index(material_index);
	if (anisotropic_texture_index != MaterialConstants::NO_TEXTURE)
		proposal_inputs.anisotropy = read_material_texture<float>(render_data, texcoords, anisotropic_texture_index, false);

	unsigned int specular_transmission_texture_index = materials_buffer_soa.get_specular_transmission_texture_index(material_index);
	if (specular_transmission_texture_index != MaterialConstants::NO_TEXTURE)
		proposal_inputs.specular_transmission = read_material_texture<float>(render_data, texcoords, specular_transmission_texture_index, false);
#endif // #if UseMaterialTextures == KERNEL_OPTION_TRUE

	float coat_roughening = materials_buffer_soa.get_coat_roughening(material_index);
	if (proposal_inputs.coat > 0.0f && coat_roughening > 0.0f)
	{
		float base_roughness		   = proposal_inputs.roughness;
		float target_base_roughness	   = hippt::pow_1_4(hippt::min(1.0f, hippt::pow_4(base_roughness) + 2.0f * hippt::pow_4(proposal_inputs.coat_roughness)));
		float roughened_base_roughness = hippt::lerp(base_roughness, target_base_roughness, proposal_inputs.coat);
		proposal_inputs.roughness	   = hippt::lerp(base_roughness, roughened_base_roughness, coat_roughening);
	}

	return proposal_inputs;
}

template <int light_sampling_strategy, int triangle_point_sampling_strategy, BSDFModel model>
HIPRT_DEVICE static LightProposalStateFor<light_sampling_strategy, triangle_point_sampling_strategy, model> load_light_proposal_state(
	const DevicePackedEffectiveMaterial& packed_material)
{
	LightProposalInputs proposal_inputs = load_light_proposal_inputs(packed_material);
	return make_light_proposal_state<light_sampling_strategy, triangle_point_sampling_strategy, model>(proposal_inputs);
}

template <int light_sampling_strategy, int triangle_point_sampling_strategy, BSDFModel model>
HIPRT_DEVICE static LightProposalStateFor<light_sampling_strategy, triangle_point_sampling_strategy, model> load_light_proposal_state(
	const HIPRTRenderData& render_data,
	int material_index,
	float2_t texcoords,
	float base_color_luminance,
	const ResolvedMaterialUserControlsCache& resolved_user_controls)
{
	LightProposalInputs proposal_inputs = load_light_proposal_inputs(render_data, material_index, texcoords, base_color_luminance, resolved_user_controls);
	return make_light_proposal_state<light_sampling_strategy, triangle_point_sampling_strategy, model>(proposal_inputs);
}

#endif // #ifndef DEVICE_MATERIAL_H
