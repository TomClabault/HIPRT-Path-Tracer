/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_MATERIAL_TRAITS_H
#define HOST_DEVICE_COMMON_MATERIAL_MATERIAL_TRAITS_H

#include "HostDeviceCommon/Material/MaterialUnpacked.h"

template <typename MaterialType>
struct MaterialTraits;

template <>
struct MaterialTraits<DeviceUnpackedPrincipledFullMaterial>
{
	static constexpr bool is_principled		   = true;
	static constexpr bool has_diffuse		   = true;
	static constexpr bool has_glass			   = true;
	static constexpr bool has_metallic		   = true;
	static constexpr bool has_specular		   = true;
	static constexpr bool has_coat			   = true;
	static constexpr bool has_sheen			   = true;
	static constexpr bool has_transmission	   = true;
	static constexpr bool has_roughness		   = true;
	static constexpr bool has_second_roughness = true;

	static constexpr KernelMaterialSpecialization family = KernelMaterialSpecializationAll;
};

template <>
struct MaterialTraits<DeviceLambertianMaterial>
{
	static constexpr bool is_principled		   = false;
	static constexpr bool has_diffuse		   = true;
	static constexpr bool has_glass			   = false;
	static constexpr bool has_metallic		   = false;
	static constexpr bool has_specular		   = false;
	static constexpr bool has_coat			   = false;
	static constexpr bool has_sheen			   = false;
	static constexpr bool has_transmission	   = false;
	static constexpr bool has_roughness		   = false;
	static constexpr bool has_second_roughness = false;

	static constexpr KernelMaterialSpecialization family = KernelMaterialSpecializationAll;
};

template <>
struct MaterialTraits<DeviceOrenNayarMaterial> : MaterialTraits<DeviceLambertianMaterial>
{
};

template <>
struct MaterialTraits<DevicePrincipledDiffuseMaterial> : MaterialTraits<DeviceLambertianMaterial>
{
	static constexpr bool is_principled = true;

	static constexpr KernelMaterialSpecialization family = KernelMaterialSpecializationDiffuse;
};

template <>
struct MaterialTraits<DevicePrincipledGlassMaterial>
{
	static constexpr bool is_principled		   = true;
	static constexpr bool has_diffuse		   = false;
	static constexpr bool has_glass			   = true;
	static constexpr bool has_metallic		   = false;
	static constexpr bool has_specular		   = false;
	static constexpr bool has_coat			   = false;
	static constexpr bool has_sheen			   = false;
	static constexpr bool has_transmission	   = true;
	static constexpr bool has_roughness		   = true;
	static constexpr bool has_second_roughness = false;

	static constexpr KernelMaterialSpecialization family = KernelMaterialSpecializationGlass;
};

template <>
struct MaterialTraits<DevicePrincipledSingleMetallicMaterial>
{
	static constexpr bool is_principled		   = true;
	static constexpr bool has_diffuse		   = false;
	static constexpr bool has_glass			   = false;
	static constexpr bool has_metallic		   = true;
	static constexpr bool has_specular		   = false;
	static constexpr bool has_coat			   = false;
	static constexpr bool has_sheen			   = false;
	static constexpr bool has_transmission	   = false;
	static constexpr bool has_roughness		   = true;
	static constexpr bool has_second_roughness = false;

	static constexpr KernelMaterialSpecialization family = KernelMaterialSpecializationSingleMetallic;
};

template <>
struct MaterialTraits<DevicePrincipledSpecularDiffuseMaterial>
{
	static constexpr bool is_principled		   = true;
	static constexpr bool has_diffuse		   = true;
	static constexpr bool has_glass			   = false;
	static constexpr bool has_metallic		   = false;
	static constexpr bool has_specular		   = true;
	static constexpr bool has_coat			   = false;
	static constexpr bool has_sheen			   = false;
	static constexpr bool has_transmission	   = false;
	static constexpr bool has_roughness		   = true;
	static constexpr bool has_second_roughness = false;

	static constexpr KernelMaterialSpecialization family = KernelMaterialSpecializationSpecularDiffuse;
};

#endif
