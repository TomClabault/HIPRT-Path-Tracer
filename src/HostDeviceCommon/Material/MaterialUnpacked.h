/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_UNPACKED_H
#define HOST_DEVICE_COMMON_MATERIAL_UNPACKED_H

//#include "HostDeviceCommon/Material/MaterialUtils.h"

/**
 * How to add a material property:
 * 
 * 1)   MaterialUnpacked.h
 *
 *      Add the unpacked device material structure what the GPU is going to need in the shaders
 *      For most parameters, this is just the parameters itself
 * 
 *      For some other parameters, some stuff can be precomputed on the CPU and the GPU
 *      can then use only the precomputed stuff. In these cases, you need to add the precomputed
 *      stuff in here. The data needed for the precomputation is only going to be stored
 *      in the CPUMaterial, step 2).
 *      
 *      An example of a precomputed parameter is the emission. On the GPU, the emission is a simple
 *      color but on the CPU the emission is a color + emission strength.
 *      The emission strength is precomputed (multiplied/factored in) into the emission when
 *      packed into the material that the GPU uses
 *      
 * 2)   CPUMaterial.h
 *
 *      Add the parameter to the CPUMaterial structure. Read step 1) for precomputed parameters.
 * 
 *      If the parameter needs clamping to avoid NaNs/singularities/numerical imprecisions, the clamping
 *      must be done in CPUMaterial::make_safe()
 * 
 *      You also need to add a line in CPUMaterial::pack_to_gpu() to define how the GPUMaterial (whose data is packed).
 *      This is most likely just a .set() call like all the other parameters. That setter will be defined in step 3).
 *      If you have some precomputation to do (such as with the emission), it can be done in there (look at the .set_emission() call)
 * 
 * 3)   MaterialPacked.h
 *
 *      Add the parameter to the MaterialPacked structure in MaterialPacked.h (only the parameters that 
 *      the GPU is going to use. So, if there is any precomputation to be done (most parameters do not have precomputation), add only what 
 *      holds the precomputed result that the GPU is directly going to use, not the data needed 
 *      for the precomputation (that's only in CPUMaterial).
 * 
 *      The parameter will need to be packed. It can be added to a member that doesn't have all its "fields" filled yet (look for // TODO)
 *      or a new member needs to be created for the new parameter.
 * 
 *      The new parameter can also be full-range, i.e. not packed if precision is important or packing is impractical
 * 
 *      After the parameter has been added to a packed member, write the getter and setter in the same class (structure)
 * 
 *      The function DevicePackedEffectiveMaterial::pack() needs to be completed (follow what is done for the other parameters).
 *      This is the function that will be called when packing the GPUMaterial into the GBuffer
 *      This looks quite a bit like pack_to_gpu() from before but there is no precomputations to be done here because the precomputation
 *      has already been done before in pack_to_gpu()
 * 
 *      The function DevicePackedEffectiveMaterial::unpack() needs to be completed (follow what is done for the other parameters).
 *      This is the function that will be called when unpacking the material from the G-Buffer
 * 
 *      The function DevicePackedTexturedMaterial::unpack() needs to be completed (follow what is done for the other parameters).
 *      This is the function that will be called when unpacking the material from the materials buffer (when reading the material of the geometry a ray just hit).
 *      The unpacked textured material will then be used to read the textures of the material at the hit point and the whole
 *      will result in a DevicePackedEffectiveMaterial that will be used in the rest of the shaders (or packed into the G-Buffer)
 * 
 * 4)   MaterialPackedSoA.h
 * 
 *      Add a getter for the parameter to DevicePackedEffectiveMaterialSoA.
 *      This is to read the parameter from the structure of arrays (one buffer per each parameter packed) given a material index
 * 
 *      Add the parameter reading in DevicePackedEffectiveMaterialSoA::read_partial_material(). This function is just a handy function
 *      to produce a DevicePackedTexturedMaterial by reading all the arrays of the structure of arrays
 * 
 *      Note that memory traffic can be saved in some case. Let's you're adding a bunch of parameters for a "super metallic" lobe:
 *      - super metallic strength
 *      - super roughness
 *      - super anisotropy
 *      - super flakes
 *      - super fresnel F82 color
 * 
 *      The 'super metallic strength' parameter controls the overall strength of the super metallic lobe. 
 *      If it is 0, then the super metallic lobe is disabled from the BSDF. This is a case where it is not 
 *      needed to read any of the other parameters (super roughness, super anisotropy, ...) because they 
 *      will not be used anyways since the super metallic lobe is disabled.
 * 
 *      This logic to save memory traffic has already been applied to most of the other lobes (coat, glass, ...)
 * 
 * 5)   Add controls to ImGuiObjectsWindow (and the global material overrider)
 */

 /**
  * Unpacked material for use in the shaders
  */
struct DeviceUnpackedEffectiveMaterial
{
    HIPRT_HOST_DEVICE bool is_emissive() const
    {
        return !hippt::is_zero(emission.r)
            || !hippt::is_zero(emission.g)
            || !hippt::is_zero(emission.b)
            || emissive_texture_used;
    }

    ColorRGB32F emission = ColorRGB32F{ 0.0f, 0.0f, 0.0f };
    ColorRGB32F base_color = ColorRGB32F(1.0f);

    float roughness = 0.3f;
    float oren_nayar_sigma = 0.34906585039886591538f; // 20 degrees standard deviation in radian

    // Parameters for Adobe 2023 F82-tint model
    float metallic = 0.0f;
    float metallic_F90_falloff_exponent = 5.0f;
    // F0 is not here as it uses the 'base_color' of the material
    ColorRGB32F metallic_F82 = ColorRGB32F(1.0f);
    ColorRGB32F metallic_F90 = ColorRGB32F(1.0f);
    float anisotropy = 0.0f;
    float anisotropy_rotation = 0.0f;
    float second_roughness_weight = 0.0f;
    float second_roughness = 0.5f;

    // Specular intensity
    float specular = 1.0f;
    // Specular tint intensity.
    // Specular will be white if 0.0f and will be 'specular_color' if 1.0f
    float specular_tint = 1.0f;
    ColorRGB32F specular_color = ColorRGB32F(1.0f);
    // Same as coat darkening but for total internal reflection inside the specular layer
    // that sits on top of the diffuse base
    //
    // Disabled by default for artistic "expectations"
    float specular_darkening = 0.0f;

    float coat = 0.0f;
    ColorRGB32F coat_medium_absorption = ColorRGB32F{ 1.0f, 1.0f, 1.0f };
    // The coat thickness influences the amount of absorption (given by 'coat_medium_absorption')
    // that will happen inside the coat
    float coat_medium_thickness = 5.0f;
    float coat_roughness = 0.0f;
    // Physical accuracy requires that a rough clearcoat also roughens what's underneath it
    // i.e. the specular/metallic/transmission layers.
    // 
    // The option is however given here to artistically disable
    // that behavior by using coat roughening = 0.0f.
    float coat_roughening = 1.0f;
    // Because of the total internal reflection that can happen inside the coat layer (i.e.
    // light bouncing between the coat/BSDF and air/coat interfaces), the BSDF below the
    // clearcoat will appear will increased saturation.
    float coat_darkening = 1.0f;
    float coat_anisotropy = 0.0f;
    float coat_anisotropy_rotation = 0.0f;
    float coat_ior = 1.5f;

    float sheen = 0.0f; // Sheen strength
    float sheen_roughness = 0.5f;
    ColorRGB32F sheen_color = ColorRGB32F(1.0f);

    float ior = 1.40f;
    float specular_transmission = 0.0f;
    float diffuse_transmission = 0.0f;
    // At what distance is the light absorbed to the given absorption_color
    float absorption_at_distance = 1.0f;
    // Color of the light absorption when traveling through the medium
    ColorRGB32F absorption_color = ColorRGB32F(1.0f);
    float dispersion_scale = 0.0f;
    float dispersion_abbe_number = 20.0f;

    float thin_film = 0.0f;
    float thin_film_ior = 1.3f;
    float thin_film_thickness = 500.0f;
    float thin_film_kappa_3 = 0.0f;
    float thin_film_hue_shift_degrees = 0.0f;
    float thin_film_base_ior_override = 1.0f;

    // 1.0f makes the material completely opaque
    // 0.0f completely transparent (becomes invisible)
    float alpha_opacity = 1.0f;

    unsigned char energy_preservation_monte_carlo_samples = 12;
    
    /**
     * The booleans are moved to the end of the structure to avoid too much structure packing
     */

    // Whether or not to do energy compensation of the metallic layer
    // for that material
    bool do_metallic_energy_compensation = true;
    // Whether or not to do energy compensation of the specular/diffuse layer
    // for that material
    bool do_specular_energy_compensation = true;
    // Whether or not to do energy compensation of the clearcoat layer
    // for that material
    bool do_coat_energy_compensation = true;
    bool thin_walled = false;
    // Whether or not to do energy compensation of the glass layer
    // for that material
    bool do_glass_energy_compensation = true;
    bool thin_film_do_ior_override = false;

    // If true, 'energy_preservation_monte_carlo_samples' will be used
    // to compute the directional albedo of this material.
    // This computed directional albedo is then used to ensure perfect energy conservation
    // and preservation. 
    // 
    // This is however very expensive.
    // This is usually only needed on clearcoated materials (but even then, the energy loss due to the absence of multiple scattering between
    // the clearcoat layer and the BSDF below may be acceptable).
    // 
    // Non-clearcoated materials can already ensure perfect (modulo implementation quality) energy 
    // conservation/preservation with the precomputed LUTs [Turquin, 2019]. 
    // 
    // See PrincipledBSDFDoEnergyCompensation in this codebase.
    bool enforce_strong_energy_conservation = false;
    bool emissive_texture_used = false;

    HIPRT_HOST_DEVICE void set_dielectric_priority(unsigned char priority) { dielectric_priority = priority; }

    HIPRT_HOST_DEVICE unsigned char get_dielectric_priority() const
    {
#if BSDFOverride == BSDF_LAMBERTIAN || BSDFOverride == BSDF_OREN_NAYAR
        // These BSDFs do not support tranmission so every material
        // should have the same priority
        return 0;
#else
        return dielectric_priority;
#endif
    }

private:
        // Nested dielectric parameter
        // Private because this may be different depending on the BRDF override
        // being used so we want to control this with getters/setters
        unsigned char dielectric_priority = 0;
};

struct DeviceUnpackedTexturedMaterial : public DeviceUnpackedEffectiveMaterial
{
    int normal_map_texture_index = 65535;

    int emission_texture_index = 65535;
    int base_color_texture_index = 65535;

    // If not 65535, there is only one texture for the metallic and the roughness parameters in which.
    // case the green channel is the roughness and the blue channel is the metalness
    int roughness_metallic_texture_index = 65535;
    int roughness_texture_index = 65535;
    int metallic_texture_index = 65535;
    int anisotropic_texture_index = 65535;

    int specular_texture_index = 65535;
    int coat_texture_index = 65535;
    int sheen_texture_index = 65535;
    int specular_transmission_texture_index = 65535;
};

#endif
