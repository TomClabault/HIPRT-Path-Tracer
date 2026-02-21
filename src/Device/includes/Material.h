/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_MATERIAL_H
#define DEVICE_MATERIAL_H

#include "Device/includes/Texture.h"

#include "Device/includes/HitInfo.h"
#include "HostDeviceCommon/Material/MaterialUtils.h"
#include "HostDeviceCommon/RenderData.h"

#ifndef __KERNELCC__
#include "Image/Image.h"
#endif

template <typename T>
HIPRT_DEVICE static T read_material_texture(const HIPRTRenderData& render_data, bool is_srgb, const float2_t& texcoords, int texture_index);
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

HIPRT_DEVICE static DeviceUnpackedEffectiveMaterial get_intersection_material(const HIPRTRenderData& render_data, int material_index, float2_t texcoords)
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

	float anisotropy = read_material_texture<float>(render_data, false, texcoords, material.anisotropic_texture_index);
	if (material.anisotropic_texture_index != MaterialConstants::NO_TEXTURE)
		material.anisotropy = anisotropy;

	float specular = read_material_texture<float>(render_data, false, texcoords, material.specular_texture_index);
	if (material.specular_texture_index != MaterialConstants::NO_TEXTURE)
		material.specular = specular;

	float coat = read_material_texture<float>(render_data, false, texcoords, material.coat_texture_index);
	if (material.coat_texture_index != MaterialConstants::NO_TEXTURE)
		material.coat = coat;

	float sheen = read_material_texture<float>(render_data, false, texcoords, material.sheen_texture_index);
	if (material.sheen_texture_index != MaterialConstants::NO_TEXTURE)
		material.sheen = sheen;

	float specular_transmission = read_material_texture<float>(render_data, false, texcoords, material.specular_transmission_texture_index);
	if (material.specular_transmission_texture_index != MaterialConstants::NO_TEXTURE)
		material.specular_transmission = specular_transmission;
#endif

	ColorRGB32F emission = read_material_texture<ColorRGB32F>(render_data, false, texcoords, material.emission_texture_index);
	if (material.emission_texture_index == MaterialConstants::NO_TEXTURE || material.emission_texture_index == MaterialConstants::CONSTANT_EMISSIVE_TEXTURE)
		emission = material.emission;

	DeviceUnpackedEffectiveMaterial unpacked_effective_material(material);
	unpacked_effective_material.base_color = material.base_color;

	unpacked_effective_material.emissive_texture_used = material.emission_texture_index != MaterialConstants::NO_TEXTURE;
	unpacked_effective_material.emission			  = emission;
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
		ColorRGB32F rgb = sample_texture_rgb_8bits(render_data.buffers.material_textures, metallic_roughness_texture_index, false, texcoords);

		// Not converting to linear here because material properties (roughness and metallic) here are assumed to be linear already
		out.x = rgb.g;
		out.y = rgb.b;
	}
	else
	{
		out.x = read_material_texture<float>(render_data, false, texcoords, roughness_texture_index);
		out.y = read_material_texture<float>(render_data, false, texcoords, metallic_texture_index);
	}

	return out;
}

HIPRT_DEVICE static ColorRGB32F get_base_color(const HIPRTRenderData& render_data, float& out_alpha, const float2_t& texcoords, int base_color_texture_index)
{
	out_alpha		  = 1.0f;
	ColorRGBA32F rgba = read_material_texture<ColorRGBA32F>(render_data, true, texcoords, base_color_texture_index);
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
HIPRT_DEVICE static T read_material_texture(const HIPRTRenderData& render_data, bool is_srgb, const float2_t& texcoords, int texture_index)
{
	if (texture_index == MaterialConstants::NO_TEXTURE || texture_index == MaterialConstants::CONSTANT_EMISSIVE_TEXTURE)
		return T();

	ColorRGBA32F rgba = sample_texture_rgba(render_data.buffers.material_textures, texture_index, is_srgb, texcoords);
	return read_data<T>(rgba);
}

#endif
