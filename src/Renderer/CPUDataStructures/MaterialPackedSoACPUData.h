/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_PACKED_SOA_CPU_DATA_H
#define HOST_DEVICE_COMMON_MATERIAL_PACKED_SOA_CPU_DATA_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Material/MaterialPackedSoA.h"
#include "Renderer/CPUGPUCommonDataStructures/DevicePackedMaterialSoACPUGPUCommonData.h"

#define DECLARE_ALL_MEMBERS_STD_TIE                                     \
  auto all_members = std::tie(                                          \
      normal_map_emission_index, base_color_roughness_metallic_index,   \
      roughness_and_metallic_index, anisotropic_specular_index,         \
      coat_sheen_index, specular_transmission_index,                    \
                                                                        \
      flags,                                                            \
                                                                        \
      emission,                                                         \
                                                                        \
      base_color_roughness,                                             \
                                                                        \
      oren_nayar_sigma,                                                 \
                                                                        \
      metallic_F90_and_metallic, metallic_F82_packed_and_diffuse_transmission,                   \
      metallic_F90_falloff_exponent,                                    \
      anisotropy_and_rotation_and_second_roughness,                     \
                                                                        \
      specular_color_and_tint_factor,                                   \
      specular_and_darkening_and_coat_roughness, coat_medium_thickness, \
      coat_and_medium_absorption,                                       \
      coat_roughening_darkening_anisotropy_and_rotation, coat_ior,      \
                                                                        \
      sheen_and_color,                                                  \
                                                                        \
      ior, absorption_color_packed, absorption_at_distance,             \
                                                                        \
      sheen_roughness_transmission_dispersion_thin_film,                \
                                                                        \
      dispersion_abbe_number, thin_film_ior, thin_film_thickness,       \
      thin_film_kappa_3, thin_film_base_ior_override,                   \
      alpha_thin_film_hue_dielectric_priority);

/**
 * These two structures here are just there to hold all the buffers created on the CPU
 * 
 * The device pointers of these buffers are then set on to the RenderData of the CPU
 * 
 * For a documentation of what's packed into the members ('specular_and_darkening_and_coat_roughness' for example),
 * see the 'DevicePackedEffectiveMaterialSoA' class
 */

struct DevicePackedEffectiveMaterialSoACPUData : public DevicePackedMaterialSoACPUGPUCommonData
{
    std::vector<UChar8BoolsPacked> flags;

    std::vector<ColorRGB32F> emission;

    std::vector<ColorRGB24bFloat0_1Packed> base_color_roughness;

    std::vector<float> oren_nayar_sigma;
 
    std::vector<ColorRGB24bFloat0_1Packed> metallic_F90_and_metallic;
    std::vector<ColorRGB24bFloat0_1Packed> metallic_F82_packed_and_diffuse_transmission;
    std::vector<float> metallic_F90_falloff_exponent;
    std::vector<Float4xPacked> anisotropy_and_rotation_and_second_roughness;

    std::vector<ColorRGB24bFloat0_1Packed> specular_color_and_tint_factor;
    std::vector<Float4xPacked> specular_and_darkening_and_coat_roughness;
    std::vector<float> coat_medium_thickness;
    std::vector<ColorRGB24bFloat0_1Packed> coat_and_medium_absorption;
    std::vector<Float4xPacked> coat_roughening_darkening_anisotropy_and_rotation;
    std::vector<float> coat_ior;

    std::vector<ColorRGB24bFloat0_1Packed> sheen_and_color;

    std::vector<float> ior;
    std::vector<ColorRGB24bFloat0_1Packed> absorption_color_packed;
    std::vector<float> absorption_at_distance;

    std::vector<Float4xPacked> sheen_roughness_transmission_dispersion_thin_film;

    std::vector<float> dispersion_abbe_number;
    std::vector<float> thin_film_ior;
    std::vector<float> thin_film_thickness;
    std::vector<float> thin_film_kappa_3;
    std::vector<float> thin_film_base_ior_override;
    std::vector<Float2xUChar2xPacked> alpha_thin_film_hue_dielectric_priority;
};

struct DevicePackedTexturedMaterialSoACPUData : public DevicePackedEffectiveMaterialSoACPUData
{
    std::vector<Uint2xPacked> normal_map_emission_index;
    std::vector<Uint2xPacked> base_color_roughness_metallic_index;
    std::vector<Uint2xPacked> roughness_and_metallic_index;
    std::vector<Uint2xPacked> anisotropic_specular_index;
    std::vector<Uint2xPacked> coat_sheen_index;
    std::vector<Uint2xPacked> specular_transmission_index;

    // Resize function using the generic for_each_member
    void resize(size_t new_element_count)
    {
        m_element_count = new_element_count;

        // This declares a std::tie of all the buffers
        DECLARE_ALL_MEMBERS_STD_TIE;

        // Function that will be applied to all the buffers to resize them
        auto resize_lambda_function = [new_element_count](auto& buffer) { buffer.resize(new_element_count); };

        // Applying the resize function to all the buffers
        std::apply([&](auto&... args) { (resize_lambda_function(args), ...); }, all_members);
    }

    void upload_data(std::vector<DevicePackedTexturedMaterial>& gpu_packed_materials)
    {
        DevicePackedTexturedMaterial* data = gpu_packed_materials.data();
        size_t element_count = gpu_packed_materials.size();

        // Textured part
        normal_map_emission_index = expand_from_gpu_packed_materials<Uint2xPacked>(0, data, offsetof(DevicePackedTexturedMaterial, normal_map_emission_index), element_count);
        base_color_roughness_metallic_index = expand_from_gpu_packed_materials<Uint2xPacked>(0, data, offsetof(DevicePackedTexturedMaterial, base_color_roughness_metallic_index), element_count);
        roughness_and_metallic_index = expand_from_gpu_packed_materials<Uint2xPacked>(0, data, offsetof(DevicePackedTexturedMaterial, roughness_and_metallic_index), element_count);
        anisotropic_specular_index = expand_from_gpu_packed_materials<Uint2xPacked>(0, data, offsetof(DevicePackedTexturedMaterial, anisotropic_specular_index), element_count);
        coat_sheen_index = expand_from_gpu_packed_materials<Uint2xPacked>(0, data, offsetof(DevicePackedTexturedMaterial, coat_sheen_index), element_count);
        specular_transmission_index = expand_from_gpu_packed_materials<Uint2xPacked>(0, data, offsetof(DevicePackedTexturedMaterial, specular_transmission_index), element_count);

        // Non textured parameters
        flags = expand_from_gpu_packed_materials<UChar8BoolsPacked>(0, data, offsetof(DevicePackedTexturedMaterial, flags), element_count);

        emission = expand_from_gpu_packed_materials<ColorRGB32F>(0, data, offsetof(DevicePackedTexturedMaterial, emission), element_count);

        base_color_roughness = expand_from_gpu_packed_materials<ColorRGB24bFloat0_1Packed>(0, data, offsetof(DevicePackedTexturedMaterial, base_color_roughness), element_count);

        oren_nayar_sigma = expand_from_gpu_packed_materials<float>(0, data, offsetof(DevicePackedTexturedMaterial, oren_nayar_sigma), element_count);

        metallic_F90_and_metallic = expand_from_gpu_packed_materials<ColorRGB24bFloat0_1Packed>(0, data, offsetof(DevicePackedTexturedMaterial, metallic_F90_and_metallic), element_count);
        metallic_F82_packed_and_diffuse_transmission = expand_from_gpu_packed_materials<ColorRGB24bFloat0_1Packed>(0, data, offsetof(DevicePackedTexturedMaterial, metallic_F82_packed_and_diffuse_transmission), element_count);
        metallic_F90_falloff_exponent = expand_from_gpu_packed_materials<float>(0, data, offsetof(DevicePackedTexturedMaterial, metallic_F90_falloff_exponent), element_count);
        anisotropy_and_rotation_and_second_roughness = expand_from_gpu_packed_materials<Float4xPacked>(0, data, offsetof(DevicePackedTexturedMaterial, anisotropy_and_rotation_and_second_roughness), element_count);

        specular_color_and_tint_factor = expand_from_gpu_packed_materials<ColorRGB24bFloat0_1Packed>(0, data, offsetof(DevicePackedTexturedMaterial, specular_color_and_tint_factor), element_count);
        specular_and_darkening_and_coat_roughness = expand_from_gpu_packed_materials<Float4xPacked>(0, data, offsetof(DevicePackedTexturedMaterial, specular_and_darkening_and_coat_roughness), element_count);
        coat_medium_thickness = expand_from_gpu_packed_materials<float>(0, data, offsetof(DevicePackedTexturedMaterial, coat_medium_thickness), element_count);
        coat_and_medium_absorption = expand_from_gpu_packed_materials<ColorRGB24bFloat0_1Packed>(0, data, offsetof(DevicePackedTexturedMaterial, coat_and_medium_absorption), element_count);
        coat_roughening_darkening_anisotropy_and_rotation = expand_from_gpu_packed_materials<Float4xPacked>(0, data, offsetof(DevicePackedTexturedMaterial, coat_roughening_darkening_anisotropy_and_rotation), element_count);
        coat_ior = expand_from_gpu_packed_materials<float>(0, data, offsetof(DevicePackedTexturedMaterial, coat_ior), element_count);

        sheen_and_color = expand_from_gpu_packed_materials<ColorRGB24bFloat0_1Packed>(0, data, offsetof(DevicePackedTexturedMaterial, sheen_and_color), element_count);

        ior = expand_from_gpu_packed_materials<float>(0, data, offsetof(DevicePackedTexturedMaterial, ior), element_count);
        absorption_color_packed = expand_from_gpu_packed_materials<ColorRGB24bFloat0_1Packed>(0, data, offsetof(DevicePackedTexturedMaterial, absorption_color_packed), element_count);
        absorption_at_distance = expand_from_gpu_packed_materials<float>(0, data, offsetof(DevicePackedTexturedMaterial, absorption_at_distance), element_count);

        sheen_roughness_transmission_dispersion_thin_film = expand_from_gpu_packed_materials<Float4xPacked>(0, data, offsetof(DevicePackedTexturedMaterial, sheen_roughness_transmission_dispersion_thin_film), element_count);

        dispersion_abbe_number = expand_from_gpu_packed_materials<float>(0, data, offsetof(DevicePackedTexturedMaterial, dispersion_abbe_number), element_count);
        thin_film_ior = expand_from_gpu_packed_materials<float>(0, data, offsetof(DevicePackedTexturedMaterial, thin_film_ior), element_count);
        thin_film_thickness = expand_from_gpu_packed_materials<float>(0, data, offsetof(DevicePackedTexturedMaterial, thin_film_thickness), element_count);
        thin_film_kappa_3 = expand_from_gpu_packed_materials<float>(0, data, offsetof(DevicePackedTexturedMaterial, thin_film_kappa_3), element_count);
        thin_film_base_ior_override = expand_from_gpu_packed_materials<float>(0, data, offsetof(DevicePackedTexturedMaterial, thin_film_base_ior_override), element_count);
        alpha_thin_film_hue_dielectric_priority = expand_from_gpu_packed_materials<Float2xUChar2xPacked>(0, data, offsetof(DevicePackedTexturedMaterial, alpha_thin_film_hue_dielectric_priority), element_count);
    }

    DevicePackedTexturedMaterialSoA get_device_SoA_struct()
    {
        DevicePackedTexturedMaterialSoA out;

        out.normal_map_emission_index = normal_map_emission_index.data();
        out.base_color_roughness_metallic_index = base_color_roughness_metallic_index.data();
        out.roughness_and_metallic_index = roughness_and_metallic_index.data();
        out.anisotropic_specular_index = anisotropic_specular_index.data();
        out.coat_sheen_index = coat_sheen_index.data();
        out.specular_transmission_index = specular_transmission_index.data();

        out.flags = flags.data();

        out.emission = emission.data();

        out.base_color_roughness = base_color_roughness.data();

        out.oren_nayar_sigma = oren_nayar_sigma.data();

        out.metallic_F90_and_metallic = metallic_F90_and_metallic.data();
        out.metallic_F82_packed_and_diffuse_transmission = metallic_F82_packed_and_diffuse_transmission.data();
        out.metallic_F90_falloff_exponent = metallic_F90_falloff_exponent.data();
        out.anisotropy_and_rotation_and_second_roughness = anisotropy_and_rotation_and_second_roughness.data();

        out.specular_color_and_tint_factor = specular_color_and_tint_factor.data();
        out.specular_and_darkening_and_coat_roughness = specular_and_darkening_and_coat_roughness.data();
        out.coat_medium_thickness = coat_medium_thickness.data();
        out.coat_and_medium_absorption = coat_and_medium_absorption.data();
        out.coat_roughening_darkening_anisotropy_and_rotation = coat_roughening_darkening_anisotropy_and_rotation.data();
        out.coat_ior = coat_ior.data();

        out.sheen_and_color = sheen_and_color.data();

        out.ior = ior.data();
        out.absorption_color_packed = absorption_color_packed.data();
        out.absorption_at_distance = absorption_at_distance.data();

        out.sheen_roughness_transmission_dispersion_thin_film = sheen_roughness_transmission_dispersion_thin_film.data();

        out.dispersion_abbe_number = dispersion_abbe_number.data();
        out.thin_film_ior = thin_film_ior.data();
        out.thin_film_thickness = thin_film_thickness.data();
        out.thin_film_kappa_3 = thin_film_kappa_3.data();
        out.thin_film_base_ior_override = thin_film_base_ior_override.data();
        out.alpha_thin_film_hue_dielectric_priority = alpha_thin_film_hue_dielectric_priority.data();

        return out;
    }

    size_t m_element_count = 0;
};

#endif
