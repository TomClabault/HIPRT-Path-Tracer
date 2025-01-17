/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_PACKED_SOA_GPU_DATA_H
#define HOST_DEVICE_COMMON_MATERIAL_PACKED_SOA_GPU_DATA_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Material/MaterialPacked.h"

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
      metallic_F90_and_metallic, metallic_F82_packed,                   \
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
 * These two structures here are just there to hold all the buffers created on the GPU
 * 
 * The device pointers of these buffers are then set on to the RenderData of the GPU
 * 
 * For a documentation of what's packed into the members ('specular_and_darkening_and_coat_roughness' for example),
 * see the 'DevicePackedEffectiveMaterialSoA' class
 */

struct DevicePackedEffectiveMaterialSoAGPUData
{
    OrochiBuffer<UChar8BoolsPacked> flags;

    OrochiBuffer<ColorRGB32F> emission;

    OrochiBuffer<ColorRGB24bFloat0_1Packed> base_color_roughness;

    OrochiBuffer<float> oren_nayar_sigma;
 
    OrochiBuffer<ColorRGB24bFloat0_1Packed> metallic_F90_and_metallic;
    OrochiBuffer<ColorRGB24bFloat0_1Packed> metallic_F82_packed;
    OrochiBuffer<float> metallic_F90_falloff_exponent;
    OrochiBuffer<Float4xPacked> anisotropy_and_rotation_and_second_roughness;

    OrochiBuffer<ColorRGB24bFloat0_1Packed> specular_color_and_tint_factor;
    OrochiBuffer<Float4xPacked> specular_and_darkening_and_coat_roughness;
    OrochiBuffer<float> coat_medium_thickness;
    OrochiBuffer<ColorRGB24bFloat0_1Packed> coat_and_medium_absorption;
    OrochiBuffer<Float4xPacked> coat_roughening_darkening_anisotropy_and_rotation;
    OrochiBuffer<float> coat_ior;

    OrochiBuffer<ColorRGB24bFloat0_1Packed> sheen_and_color;

    OrochiBuffer<float> ior;
    OrochiBuffer<ColorRGB24bFloat0_1Packed> absorption_color_packed;
    OrochiBuffer<float> absorption_at_distance;

    OrochiBuffer<Float4xPacked> sheen_roughness_transmission_dispersion_thin_film;

    OrochiBuffer<float> dispersion_abbe_number;
    OrochiBuffer<float> thin_film_ior;
    OrochiBuffer<float> thin_film_thickness;
    OrochiBuffer<float> thin_film_kappa_3;
    OrochiBuffer<float> thin_film_base_ior_override;
    OrochiBuffer<Float2xUChar2xPacked> alpha_thin_film_hue_dielectric_priority;
};

struct DevicePackedTexturedMaterialSoAGPUData : public DevicePackedEffectiveMaterialSoAGPUData
{
    OrochiBuffer<Uint2xPacked> normal_map_emission_index;
    OrochiBuffer<Uint2xPacked> base_color_roughness_metallic_index;
    OrochiBuffer<Uint2xPacked> roughness_and_metallic_index;
    OrochiBuffer<Uint2xPacked> anisotropic_specular_index;
    OrochiBuffer<Uint2xPacked> coat_sheen_index;
    OrochiBuffer<Uint2xPacked> specular_transmission_index;

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
        upload_data_partial(0, gpu_packed_materials.data(), gpu_packed_materials.size());
    }

    void upload_data_partial(unsigned int start_index, const DevicePackedTexturedMaterial* data, size_t element_count)
    {
        // Textured part
        normal_map_emission_index.upload_data_partial(start_index, expand_from_gpu_packed_materials<Uint2xPacked>(start_index, data, offsetof(DevicePackedTexturedMaterial, normal_map_emission_index), element_count).data(), element_count);
        base_color_roughness_metallic_index.upload_data_partial(start_index, expand_from_gpu_packed_materials<Uint2xPacked>(start_index, data, offsetof(DevicePackedTexturedMaterial, base_color_roughness_metallic_index), element_count).data(), element_count);
        roughness_and_metallic_index.upload_data_partial(start_index, expand_from_gpu_packed_materials<Uint2xPacked>(start_index, data, offsetof(DevicePackedTexturedMaterial, roughness_and_metallic_index), element_count).data(), element_count);
        anisotropic_specular_index.upload_data_partial(start_index, expand_from_gpu_packed_materials<Uint2xPacked>(start_index, data, offsetof(DevicePackedTexturedMaterial, anisotropic_specular_index), element_count).data(), element_count);
        coat_sheen_index.upload_data_partial(start_index, expand_from_gpu_packed_materials<Uint2xPacked>(start_index, data, offsetof(DevicePackedTexturedMaterial, coat_sheen_index), element_count).data(), element_count);
        specular_transmission_index.upload_data_partial(start_index, expand_from_gpu_packed_materials<Uint2xPacked>(start_index, data, offsetof(DevicePackedTexturedMaterial, specular_transmission_index), element_count).data(), element_count);

        // Non textured parameters
        flags.upload_data_partial(start_index, expand_from_gpu_packed_materials<UChar8BoolsPacked>(start_index, data, offsetof(DevicePackedTexturedMaterial, flags), element_count).data(), element_count);

        emission.upload_data_partial(start_index, expand_from_gpu_packed_materials<ColorRGB32F>(start_index, data, offsetof(DevicePackedTexturedMaterial, emission), element_count).data(), element_count);

        base_color_roughness.upload_data_partial(start_index, expand_from_gpu_packed_materials<ColorRGB24bFloat0_1Packed>(start_index, data, offsetof(DevicePackedTexturedMaterial, base_color_roughness), element_count).data(), element_count);

        oren_nayar_sigma.upload_data_partial(start_index, expand_from_gpu_packed_materials<float>(start_index, data, offsetof(DevicePackedTexturedMaterial, oren_nayar_sigma), element_count).data(), element_count);

        metallic_F90_and_metallic.upload_data_partial(start_index, expand_from_gpu_packed_materials<ColorRGB24bFloat0_1Packed>(start_index, data, offsetof(DevicePackedTexturedMaterial, metallic_F90_and_metallic), element_count).data(), element_count);
        metallic_F82_packed.upload_data_partial(start_index, expand_from_gpu_packed_materials<ColorRGB24bFloat0_1Packed>(start_index, data, offsetof(DevicePackedTexturedMaterial, metallic_F82_packed), element_count).data(), element_count);
        metallic_F90_falloff_exponent.upload_data_partial(start_index, expand_from_gpu_packed_materials<float>(start_index, data, offsetof(DevicePackedTexturedMaterial, metallic_F90_falloff_exponent), element_count).data(), element_count);
        anisotropy_and_rotation_and_second_roughness.upload_data_partial(start_index, expand_from_gpu_packed_materials<Float4xPacked>(start_index, data, offsetof(DevicePackedTexturedMaterial, anisotropy_and_rotation_and_second_roughness), element_count).data(), element_count);

        specular_color_and_tint_factor.upload_data_partial(start_index, expand_from_gpu_packed_materials<ColorRGB24bFloat0_1Packed>(start_index, data, offsetof(DevicePackedTexturedMaterial, specular_color_and_tint_factor), element_count).data(), element_count);
        specular_and_darkening_and_coat_roughness.upload_data_partial(start_index, expand_from_gpu_packed_materials<Float4xPacked>(start_index, data, offsetof(DevicePackedTexturedMaterial, specular_and_darkening_and_coat_roughness), element_count).data(), element_count);
        coat_medium_thickness.upload_data_partial(start_index, expand_from_gpu_packed_materials<float>(start_index, data, offsetof(DevicePackedTexturedMaterial, coat_medium_thickness), element_count).data(), element_count);
        coat_and_medium_absorption.upload_data_partial(start_index, expand_from_gpu_packed_materials<ColorRGB24bFloat0_1Packed>(start_index, data, offsetof(DevicePackedTexturedMaterial, coat_and_medium_absorption), element_count).data(), element_count);
        coat_roughening_darkening_anisotropy_and_rotation.upload_data_partial(start_index, expand_from_gpu_packed_materials<Float4xPacked>(start_index, data, offsetof(DevicePackedTexturedMaterial, coat_roughening_darkening_anisotropy_and_rotation), element_count).data(), element_count);
        coat_ior.upload_data_partial(start_index, expand_from_gpu_packed_materials<float>(start_index, data, offsetof(DevicePackedTexturedMaterial, coat_ior), element_count).data(), element_count);

        sheen_and_color.upload_data_partial(start_index, expand_from_gpu_packed_materials<ColorRGB24bFloat0_1Packed>(start_index, data, offsetof(DevicePackedTexturedMaterial, sheen_and_color), element_count).data(), element_count);

        ior.upload_data_partial(start_index, expand_from_gpu_packed_materials<float>(start_index, data, offsetof(DevicePackedTexturedMaterial, ior), element_count).data(), element_count);
        absorption_color_packed.upload_data_partial(start_index, expand_from_gpu_packed_materials<ColorRGB24bFloat0_1Packed>(start_index, data, offsetof(DevicePackedTexturedMaterial, absorption_color_packed), element_count).data(), element_count);
        absorption_at_distance.upload_data_partial(start_index, expand_from_gpu_packed_materials<float>(start_index, data, offsetof(DevicePackedTexturedMaterial, absorption_at_distance), element_count).data(), element_count);

        sheen_roughness_transmission_dispersion_thin_film.upload_data_partial(start_index, expand_from_gpu_packed_materials<Float4xPacked>(start_index, data, offsetof(DevicePackedTexturedMaterial, sheen_roughness_transmission_dispersion_thin_film), element_count).data(), element_count);

        dispersion_abbe_number.upload_data_partial(start_index, expand_from_gpu_packed_materials<float>(start_index, data, offsetof(DevicePackedTexturedMaterial, dispersion_abbe_number), element_count).data(), element_count);
        thin_film_ior.upload_data_partial(start_index, expand_from_gpu_packed_materials<float>(start_index, data, offsetof(DevicePackedTexturedMaterial, thin_film_ior), element_count).data(), element_count);
        thin_film_thickness.upload_data_partial(start_index, expand_from_gpu_packed_materials<float>(start_index, data, offsetof(DevicePackedTexturedMaterial, thin_film_thickness), element_count).data(), element_count);
        thin_film_kappa_3.upload_data_partial(start_index, expand_from_gpu_packed_materials<float>(start_index, data, offsetof(DevicePackedTexturedMaterial, thin_film_kappa_3), element_count).data(), element_count);
        thin_film_base_ior_override.upload_data_partial(start_index, expand_from_gpu_packed_materials<float>(start_index, data, offsetof(DevicePackedTexturedMaterial, thin_film_base_ior_override), element_count).data(), element_count);
        alpha_thin_film_hue_dielectric_priority.upload_data_partial(start_index, expand_from_gpu_packed_materials<Float2xUChar2xPacked>(start_index, data, offsetof(DevicePackedTexturedMaterial, alpha_thin_film_hue_dielectric_priority), element_count).data(), element_count);
    }

    DevicePackedTexturedMaterialSoA get_device_SoA_struct()
    {
        DevicePackedTexturedMaterialSoA out;

        out.normal_map_emission_index = normal_map_emission_index.get_device_pointer();
        out.base_color_roughness_metallic_index = base_color_roughness_metallic_index.get_device_pointer();
        out.roughness_and_metallic_index = roughness_and_metallic_index.get_device_pointer();
        out.anisotropic_specular_index = anisotropic_specular_index.get_device_pointer();
        out.coat_sheen_index = coat_sheen_index.get_device_pointer();
        out.specular_transmission_index = specular_transmission_index.get_device_pointer();

        out.flags = flags.get_device_pointer();

        out.emission = emission.get_device_pointer();

        out.base_color_roughness = base_color_roughness.get_device_pointer();

        out.oren_nayar_sigma = oren_nayar_sigma.get_device_pointer();

        out.metallic_F90_and_metallic = metallic_F90_and_metallic.get_device_pointer();
        out.metallic_F82_packed = metallic_F82_packed.get_device_pointer();
        out.metallic_F90_falloff_exponent = metallic_F90_falloff_exponent.get_device_pointer();
        out.anisotropy_and_rotation_and_second_roughness = anisotropy_and_rotation_and_second_roughness.get_device_pointer();

        out.specular_color_and_tint_factor = specular_color_and_tint_factor.get_device_pointer();
        out.specular_and_darkening_and_coat_roughness = specular_and_darkening_and_coat_roughness.get_device_pointer();
        out.coat_medium_thickness = coat_medium_thickness.get_device_pointer();
        out.coat_and_medium_absorption = coat_and_medium_absorption.get_device_pointer();
        out.coat_roughening_darkening_anisotropy_and_rotation = coat_roughening_darkening_anisotropy_and_rotation.get_device_pointer();
        out.coat_ior = coat_ior.get_device_pointer();

        out.sheen_and_color = sheen_and_color.get_device_pointer();

        out.ior = ior.get_device_pointer();
        out.absorption_color_packed = absorption_color_packed.get_device_pointer();
        out.absorption_at_distance = absorption_at_distance.get_device_pointer();

        out.sheen_roughness_transmission_dispersion_thin_film = sheen_roughness_transmission_dispersion_thin_film.get_device_pointer();

        out.dispersion_abbe_number = dispersion_abbe_number.get_device_pointer();
        out.thin_film_ior = thin_film_ior.get_device_pointer();
        out.thin_film_thickness = thin_film_thickness.get_device_pointer();
        out.thin_film_kappa_3 = thin_film_kappa_3.get_device_pointer();
        out.thin_film_base_ior_override = thin_film_base_ior_override.get_device_pointer();
        out.alpha_thin_film_hue_dielectric_priority = alpha_thin_film_hue_dielectric_priority.get_device_pointer();

        return out;
    }

    size_t m_element_count = 0;

private:
    /**
     * Takes a pointer to some 'DevicePackedTexturedMaterial' in the 'gpu_packed_materials' array (which could be std::vector().data() for example) 
     * and returns a vector of type T that contains 'element_count' elements at offset 'offset' of the 'DevicePackedTexturedMaterial' structure
     * 
     * For example:
     * expand_from_gpu_packed_materials<Uint2xPacked>(3, gpu_packed_materials, offsetof(DevicePackedTexturedMaterial, normal_map_emission_index), 2)
     * 
     * return an std::vector that contains the 'normal_map_emission_index' of gpu_packed_materials[3] and gpu_packed_materials[4]
     */
    template <typename T>
    std::vector<T> expand_from_gpu_packed_materials(unsigned int start_index, const DevicePackedTexturedMaterial* gpu_packed_materials, size_t offset_in_struct, size_t element_count)
    {
        std::vector<T> out(element_count);

        for (int i = 0; i < element_count; i++)
            out[i] = *reinterpret_cast<const T*>(reinterpret_cast<const char*>(&gpu_packed_materials[start_index + i]) + offset_in_struct);

        return out;
    }
};

#endif
