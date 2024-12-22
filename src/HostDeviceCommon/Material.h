/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_H
#define HOST_DEVICE_COMMON_MATERIAL_H

#include "Device/includes/NestedDielectrics.h"

#include "HostDeviceCommon/KernelOptions.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Math.h"
#include "HostDeviceCommon/Packing.h"

struct MaterialUtils
{
    static constexpr int NO_TEXTURE = 65535;
    // When an emissive texture is read and is determine to be
    // constant, no emissive texture will be used. Instead,
    // we'll just set the emission of the material to that constant emission value
    // and the emissive texture index of the material will be replaced by
    // CONSTANT_EMISSIVE_TEXTURE
    static constexpr int CONSTANT_EMISSIVE_TEXTURE = 65534;
    // Maximum number of different textures per scene
    static constexpr int MAX_TEXTURE_COUNT = 65533;

    static constexpr float ROUGHNESS_CLAMP = 1.0e-4f;

    HIPRT_HOST_DEVICE static void get_oren_nayar_AB(float sigma, float& out_oren_A, float& out_oren_B)
    {
        float sigma2 = sigma * sigma;
        out_oren_A = 1.0f - sigma2 / (2.0f * (sigma2 + 0.33f));
        out_oren_B = 0.45f * sigma2 / (sigma2 + 0.09f);
    }

    HIPRT_HOST_DEVICE static void get_alphas(float roughness, float anisotropy, float& out_alpha_x, float& out_alpha_y)
    {
        float aspect = sqrtf(1.0f - 0.9f * anisotropy);
        out_alpha_x = hippt::max(ROUGHNESS_CLAMP, roughness * roughness / aspect);
        out_alpha_y = hippt::max(ROUGHNESS_CLAMP, roughness * roughness * aspect);
    }

    HIPRT_HOST_DEVICE static float get_thin_walled_roughness(bool thin_walled, float base_roughness, float relative_eta)
    {
        if (!thin_walled)
            return base_roughness;

        /*
         * Roughness remapping so that a thin walled interface matches better a
         * properly modeled double interface model. Said otherwise: roughness remapping
         * so that the thin walled approximation matches the non thin walled physically correct equivalent
         *
         * Reference:
         * [Revisiting Physically Based Shading at Imageworks, Christopher Kulla & Alejandro Conty, 2017]
         *
         * https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
         */
        float remapped = base_roughness * sqrt(3.7f * (relative_eta - 1.0f) * hippt::square(relative_eta - 0.5f) / hippt::pow_3(relative_eta));

        // Remapped roughness starts going above 1.0f starting at relative eta around 1.9f
        // and ends up at 1.39f at relative eta 3.5f
        //
        // Because we don't expect the user to input higher IOR values than that,
        // we remap that remapped roughness from [0.0f, 1.39f] to [0.0f, 1.0f]
        // and if the user inputs higher IOR values than 3.5f, we clamp to 1.0f roughness
        // anyways
        return hippt::clamp(0.0f, 1.0f, remapped / 1.39f);
    }
};

/**
 * Packed material for use in the shaders
 */
struct DeviceUnpackedEffectiveMaterial
{
    HIPRT_HOST_DEVICE bool is_emissive() const
    {
        return !hippt::is_zero(emission.r)
            || !hippt::is_zero(emission.g)
            || !hippt::is_zero(emission.b)
            || get_emissive_texture_used();
    }

    HIPRT_HOST_DEVICE ColorRGB32F get_emission() const { return this->emission; }
    HIPRT_HOST_DEVICE bool get_emissive_texture_used() const { return this->emissive_texture_used; }

    HIPRT_HOST_DEVICE ColorRGB32F get_base_color() const { return this->base_color; }

    HIPRT_HOST_DEVICE float get_roughness() const { return this->roughness; }
    HIPRT_HOST_DEVICE float get_oren_nayar_sigma() const { return this->oren_nayar_sigma; }

    HIPRT_HOST_DEVICE float get_metallic() const { return this->metallic; }
    HIPRT_HOST_DEVICE float get_metallic_F90_falloff_exponent() const { return this->metallic_F90_falloff_exponent; }
    HIPRT_HOST_DEVICE ColorRGB32F get_metallic_F82() const { return this->metallic_F82; }
    HIPRT_HOST_DEVICE ColorRGB32F get_metallic_F90() const { return this->metallic_F90; }
    HIPRT_HOST_DEVICE float get_anisotropy() const { return this->anisotropy; }
    HIPRT_HOST_DEVICE float get_anisotropy_rotation() const { return this->anisotropy_rotation; }
    HIPRT_HOST_DEVICE float get_second_roughness_weight() const { return this->second_roughness_weight; }
    HIPRT_HOST_DEVICE float get_second_roughness() const { return this->second_roughness; }

    HIPRT_HOST_DEVICE float get_specular() const { return this->specular; }
    HIPRT_HOST_DEVICE float get_specular_tint() const { return this->specular_tint; }
    HIPRT_HOST_DEVICE ColorRGB32F get_specular_color() const { return this->specular_color; }
    HIPRT_HOST_DEVICE float get_specular_darkening() const { return this->specular_darkening; }

    HIPRT_HOST_DEVICE float get_coat() const { return this->coat; }
    HIPRT_HOST_DEVICE ColorRGB32F get_coat_medium_absorption() const { return this->coat_medium_absorption; }
    HIPRT_HOST_DEVICE float get_coat_medium_thickness() const { return this->coat_medium_thickness; }
    HIPRT_HOST_DEVICE float get_coat_roughness() const { return this->coat_roughness; }
    HIPRT_HOST_DEVICE float get_coat_roughening() const { return this->coat_roughening; }
    HIPRT_HOST_DEVICE float get_coat_darkening() const { return this->coat_darkening; }
    HIPRT_HOST_DEVICE float get_coat_anisotropy() const { return this->coat_anisotropy; }
    HIPRT_HOST_DEVICE float get_coat_anisotropy_rotation() const { return this->coat_anisotropy_rotation; }
    HIPRT_HOST_DEVICE float get_coat_ior() const { return this->coat_ior; }

    HIPRT_HOST_DEVICE float get_sheen() const { return this->sheen; }
    HIPRT_HOST_DEVICE float get_sheen_roughness() const { return this->sheen_roughness; }
    HIPRT_HOST_DEVICE ColorRGB32F get_sheen_color() const { return this->sheen_color; }

    HIPRT_HOST_DEVICE float get_ior() const { return this->ior; }
    HIPRT_HOST_DEVICE float get_specular_transmission() const { return this->specular_transmission; }
    HIPRT_HOST_DEVICE float get_absorption_at_distance() const { return this->absorption_at_distance; }
    HIPRT_HOST_DEVICE ColorRGB32F get_absorption_color() const { return this->absorption_color; }
    HIPRT_HOST_DEVICE float get_dispersion_scale() const { return this->dispersion_scale; }
    HIPRT_HOST_DEVICE float get_dispersion_abbe_number() const { return this->dispersion_abbe_number; }
    HIPRT_HOST_DEVICE bool get_thin_walled() const { return this->thin_walled; }

    HIPRT_HOST_DEVICE float get_thin_film() const { return this->thin_film; }
    HIPRT_HOST_DEVICE float get_thin_film_ior() const { return this->thin_film_ior; }
    HIPRT_HOST_DEVICE float get_thin_film_thickness() const { return this->thin_film_thickness; }
    HIPRT_HOST_DEVICE float get_thin_film_kappa_3() const { return this->thin_film_kappa_3; }
    HIPRT_HOST_DEVICE float get_thin_film_hue_shift_degrees() const { return this->thin_film_hue_shift_degrees; }
    HIPRT_HOST_DEVICE bool get_thin_film_base_ior_override() const { return this->thin_film_base_ior_override; }
    HIPRT_HOST_DEVICE bool get_thin_film_do_ior_override() const { return this->thin_film_do_ior_override; }

    HIPRT_HOST_DEVICE float get_alpha_opacity() const { return this->alpha_opacity; }
    HIPRT_HOST_DEVICE unsigned char get_dielectric_priority() const { return this->dielectric_priority; }

    HIPRT_HOST_DEVICE unsigned char get_energy_preservation_monte_carlo_samples() const { return this->energy_preservation_monte_carlo_samples; }
    HIPRT_HOST_DEVICE bool get_enforce_strong_energy_conservation() const { return this->enforce_strong_energy_conservation; }




    HIPRT_HOST_DEVICE void set_emission(ColorRGB32F emission) { this->emission = emission; }
    HIPRT_HOST_DEVICE void set_emissive_texture_used(bool emissive_texture_used) { this->emissive_texture_used = emissive_texture_used; }

    HIPRT_HOST_DEVICE void set_base_color(ColorRGB32F base_color) { this->base_color = base_color; }

    HIPRT_HOST_DEVICE void set_roughness(float roughness) { this->roughness = roughness; }
    HIPRT_HOST_DEVICE void set_oren_nayar_sigma(float oren_nayar_sigma) { this->oren_nayar_sigma = oren_nayar_sigma; }

    HIPRT_HOST_DEVICE void set_metallic(float metallic) { this->metallic = metallic; }
    HIPRT_HOST_DEVICE void set_metallic_F90_falloff_exponent(float metallic_F90_falloff_exponent) { this->metallic_F90_falloff_exponent = metallic_F90_falloff_exponent; }
    HIPRT_HOST_DEVICE void set_metallic_F82(ColorRGB32F metallic_F82) { this->metallic_F82 = metallic_F82; }
    HIPRT_HOST_DEVICE void set_metallic_F90(ColorRGB32F metallic_F90) { this->metallic_F90 = metallic_F90; }
    HIPRT_HOST_DEVICE void set_anisotropy(float anisotropy) { this->anisotropy = anisotropy; }
    HIPRT_HOST_DEVICE void set_anisotropy_rotation(float anisotropy_rotation) { this->anisotropy_rotation = anisotropy_rotation; }
    HIPRT_HOST_DEVICE void set_second_roughness_weight(float second_roughness_weight) { this->second_roughness_weight = second_roughness_weight; }
    HIPRT_HOST_DEVICE void set_second_roughness(float second_roughness) { this->second_roughness = second_roughness; }

    HIPRT_HOST_DEVICE void set_specular(float specular) { this->specular = specular; }
    HIPRT_HOST_DEVICE void set_specular_tint(float specular_tint) { this->specular_tint = specular_tint; }
    HIPRT_HOST_DEVICE void set_specular_color(ColorRGB32F specular_color) { this->specular_color = specular_color; }
    HIPRT_HOST_DEVICE void set_specular_darkening(float specular_darkening) { this->specular_darkening = specular_darkening; }

    HIPRT_HOST_DEVICE void set_coat(float coat) { this->coat = coat; }
    HIPRT_HOST_DEVICE void set_coat_medium_absorption(ColorRGB32F coat_medium_absorption) { this->coat_medium_absorption = coat_medium_absorption; }
    HIPRT_HOST_DEVICE void set_coat_medium_thickness(float coat_medium_thickness) { this->coat_medium_thickness = coat_medium_thickness; }
    HIPRT_HOST_DEVICE void set_coat_roughness(float coat_roughness) { this->coat_roughness = coat_roughness; }
    HIPRT_HOST_DEVICE void set_coat_roughening(float coat_roughening) { this->coat_roughening = coat_roughening; }
    HIPRT_HOST_DEVICE void set_coat_darkening(float coat_darkening) { this->coat_darkening = coat_darkening; }
    HIPRT_HOST_DEVICE void set_coat_anisotropy(float coat_anisotropy) { this->coat_anisotropy = coat_anisotropy; }
    HIPRT_HOST_DEVICE void set_coat_anisotropy_rotation(float coat_anisotropy_rotation) { this->coat_anisotropy_rotation = coat_anisotropy_rotation; }
    HIPRT_HOST_DEVICE void set_coat_ior(float coat_ior) { this->coat_ior = coat_ior; }

    HIPRT_HOST_DEVICE void set_sheen(float sheen) { this->sheen = sheen; }
    HIPRT_HOST_DEVICE void set_sheen_roughness(float sheen_roughness) { this->sheen_roughness = sheen_roughness; }
    HIPRT_HOST_DEVICE void set_sheen_color(ColorRGB32F sheen_color) { this->sheen_color = sheen_color; }

    HIPRT_HOST_DEVICE void set_ior(float ior) { this->ior = ior; }
    HIPRT_HOST_DEVICE void set_specular_transmission(float specular_transmission) { this->specular_transmission = specular_transmission; }
    HIPRT_HOST_DEVICE void set_absorption_at_distance(float absorption_at_distance) { this->absorption_at_distance = absorption_at_distance; }
    HIPRT_HOST_DEVICE void set_absorption_color(ColorRGB32F absorption_color) { this->absorption_color = absorption_color; }
    HIPRT_HOST_DEVICE void set_dispersion_scale(float dispersion_scale) { this->dispersion_scale = dispersion_scale; }
    HIPRT_HOST_DEVICE void set_dispersion_abbe_number(float dispersion_abbe_number) { this->dispersion_abbe_number = dispersion_abbe_number; }
    HIPRT_HOST_DEVICE void set_thin_walled(bool thin_walled) { this->thin_walled = thin_walled; }

    HIPRT_HOST_DEVICE void set_thin_film(float thin_film) { this->thin_film = thin_film; }
    HIPRT_HOST_DEVICE void set_thin_film_ior(float thin_film_ior) { this->thin_film_ior = thin_film_ior; }
    HIPRT_HOST_DEVICE void set_thin_film_thickness(float thin_film_thickness) { this->thin_film_thickness = thin_film_thickness; }
    HIPRT_HOST_DEVICE void set_thin_film_kappa_3(float thin_film_kappa_3) { this->thin_film_kappa_3 = thin_film_kappa_3; }
    HIPRT_HOST_DEVICE void set_thin_film_hue_shift_degrees(float thin_film_hue_shift_degrees) { this->thin_film_hue_shift_degrees = thin_film_hue_shift_degrees; }
    HIPRT_HOST_DEVICE void set_thin_film_base_ior_override(bool thin_film_base_ior_override) { this->thin_film_base_ior_override = thin_film_base_ior_override; }
    HIPRT_HOST_DEVICE void set_thin_film_do_ior_override(bool thin_film_do_ior_override) { this->thin_film_do_ior_override = thin_film_do_ior_override; }

    HIPRT_HOST_DEVICE void set_alpha_opacity(float alpha_opacity) { this->alpha_opacity = alpha_opacity; }
    HIPRT_HOST_DEVICE void set_dielectric_priority(unsigned char dielectric_priority) { this->dielectric_priority = dielectric_priority; }

    HIPRT_HOST_DEVICE void set_energy_preservation_monte_carlo_samples(unsigned char energy_preservation_monte_carlo_samples) { this->energy_preservation_monte_carlo_samples = energy_preservation_monte_carlo_samples; }
    HIPRT_HOST_DEVICE void set_enforce_strong_energy_conservation(bool enforce_strong_energy_conservation) { this->enforce_strong_energy_conservation = enforce_strong_energy_conservation; }

private:
    ColorRGB32F emission = ColorRGB32F{ 0.0f, 0.0f, 0.0f };
    float emission_strength = 1.0f; // This factor is baked into 'emission' before being sent to the GPU
    float global_emissive_factor = 1.0f; // This factor is baked into 'emission' before being sent to the GPU
    bool emissive_texture_used = false;

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
    // At what distance is the light absorbed to the given absorption_color
    float absorption_at_distance = 1.0f;
    // Color of the light absorption when traveling through the medium
    ColorRGB32F absorption_color = ColorRGB32F(1.0f);
    float dispersion_scale = 0.0f;
    float dispersion_abbe_number = 20.0f;
    bool thin_walled = false;

    float thin_film = 0.0f;
    float thin_film_ior = 1.3f;
    float thin_film_thickness = 500.0f;
    float thin_film_kappa_3 = 0.0f;
    float thin_film_hue_shift_degrees = 0.0f;
    float thin_film_base_ior_override = 1.0f;
    bool thin_film_do_ior_override = false;

    // 1.0f makes the material completely opaque
    // 0.0f completely transparent (becomes invisible)
    float alpha_opacity = 1.0f;

    // Nested dielectric parameter
    unsigned char dielectric_priority = 0;

    int energy_preservation_monte_carlo_samples = 12;
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
    // See PrincipledBSDFGGXUseMultipleScattering in this codebase.
    bool enforce_strong_energy_conservation = false;
};

struct DeviceUnpackedTexturedMaterial : public DeviceUnpackedEffectiveMaterial
{
    HIPRT_HOST_DEVICE unsigned short int get_normal_map_texture_index() const { return normal_map_texture_index; }
    HIPRT_HOST_DEVICE unsigned short int get_emission_texture_index() const { return emission_texture_index; }
    HIPRT_HOST_DEVICE unsigned short int get_base_color_texture_index() const { return base_color_texture_index; }
    HIPRT_HOST_DEVICE unsigned short int get_roughness_metallic_texture_index() const { return roughness_metallic_texture_index; }
    HIPRT_HOST_DEVICE unsigned short int get_roughness_texture_index() const { return roughness_texture_index; }
    HIPRT_HOST_DEVICE unsigned short int get_metallic_texture_index() const { return metallic_texture_index; }
    HIPRT_HOST_DEVICE unsigned short int get_anisotropic_texture_index() const { return anisotropic_texture_index; }
    HIPRT_HOST_DEVICE unsigned short int get_specular_texture_index() const { return specular_texture_index; }
    HIPRT_HOST_DEVICE unsigned short int get_coat_texture_index() const { return coat_texture_index; }
    HIPRT_HOST_DEVICE unsigned short int get_sheen_texture_index() const { return sheen_texture_index; }
    HIPRT_HOST_DEVICE unsigned short int get_specular_transmission_texture_index() const { return specular_transmission_texture_index; }

    HIPRT_HOST_DEVICE void set_normal_map_texture_index(unsigned short normal_map_texture_index) { this->normal_map_texture_index = normal_map_texture_index; }
    HIPRT_HOST_DEVICE void set_emission_texture_index(unsigned short emission_texture_index) { this->emission_texture_index = emission_texture_index; }
    HIPRT_HOST_DEVICE void set_base_color_texture_index(unsigned short base_color_texture_index) { this->base_color_texture_index = base_color_texture_index; }
    HIPRT_HOST_DEVICE void set_roughness_metallic_texture_index(unsigned short roughness_metallic_texture_index) { this->roughness_metallic_texture_index = roughness_metallic_texture_index; }
    HIPRT_HOST_DEVICE void set_roughness_texture_index(unsigned short roughness_texture_index) { this->roughness_texture_index = roughness_texture_index; }
    HIPRT_HOST_DEVICE void set_metallic_texture_index(unsigned short metallic_texture_index) { this->metallic_texture_index = metallic_texture_index; }
    HIPRT_HOST_DEVICE void set_anisotropic_texture_index(unsigned short anisotropic_texture_index) { this->anisotropic_texture_index = anisotropic_texture_index; }
    HIPRT_HOST_DEVICE void set_specular_texture_index(unsigned short specular_texture_index) { this->specular_texture_index = specular_texture_index; }
    HIPRT_HOST_DEVICE void set_coat_texture_index(unsigned short coat_texture_index) { this->coat_texture_index = coat_texture_index; }
    HIPRT_HOST_DEVICE void set_sheen_texture_index(unsigned short sheen_texture_index) { this->sheen_texture_index = sheen_texture_index; }
    HIPRT_HOST_DEVICE void set_specular_transmission_texture_index(unsigned short specular_transmission_texture_index) { this->specular_transmission_texture_index = specular_transmission_texture_index; }

private:
    int normal_map_texture_index = MaterialUtils::NO_TEXTURE;

    int emission_texture_index = MaterialUtils::NO_TEXTURE;
    int base_color_texture_index = MaterialUtils::NO_TEXTURE;

    // If not MaterialUtils::NO_TEXTURE, there is only one texture for the metallic and the roughness parameters in which.
    // case the green channel is the roughness and the blue channel is the metalness
    int roughness_metallic_texture_index = MaterialUtils::NO_TEXTURE;
    int roughness_texture_index = MaterialUtils::NO_TEXTURE;
    int metallic_texture_index = MaterialUtils::NO_TEXTURE;
    int anisotropic_texture_index = MaterialUtils::NO_TEXTURE;

    int specular_texture_index = MaterialUtils::NO_TEXTURE;
    int coat_texture_index = MaterialUtils::NO_TEXTURE;
    int sheen_texture_index = MaterialUtils::NO_TEXTURE;
    int specular_transmission_texture_index = MaterialUtils::NO_TEXTURE;
};

/**
 * Packed material for use in the shaders
 */
struct DevicePackedEffectiveMaterial
{
    enum PackedFlagsIndices : unsigned char
    {
        PACKED_THIN_WALLED = 0,
        PACKED_EMISSIVE_TEXTURE_USED = 1,
        PACKED_THIN_FILM_DO_IOR_OVERRIDE = 2,
        PACKED_ENFORCE_STRONG_ENERGY_CONSERVATION = 3,
    };

    enum PackedAnisotropyGroupIndices : unsigned char
    {
        PACKED_ANISOTROPY = 0,
        PACKED_ANISOTROPY_ROTATION = 1,
        PACKED_SECOND_ROUGHNESS_WEIGHT = 2,
        PACKED_SECOND_ROUGHNESS = 3,
    };

    enum PackedSpecularGroupIndices : unsigned char
    {
        PACKED_SPECULAR = 0,
        PACKED_SPECULAR_DARKENING = 1,
        PACKED_COAT_ROUGHNESS = 2
    };

    enum PackedCoatGroupIndices : unsigned char
    {
        PACKED_COAT_ROUGHENING = 0,
        PACKED_COAT_DARKENING = 1,
        PACKED_COAT_ANISOTROPY = 2,
        PACKED_COAT_ANISOTROPY_ROTATION = 3,
    };

    enum PackedSheenRoughnessGroupIndices : unsigned char
    {
        PACKED_SHEEN_ROUGHNESS = 0,
        PACKED_SPECULAR_TRANSMISSION = 1,
        PACKED_DISPERSION_SCALE = 2,
        PACKED_THIN_FILM = 3,
    };

    enum PackedAlphaOpacityGroupIndices : unsigned char
    {
        PACKED_ALPHA_OPACITY = 0,
        PACKED_THIN_FILM_HUE_SHIFT = 1,
        PACKED_DIELECTRIC_PRIORITY = 0,
        PACKED_ENERGY_PRESERVATION_SAMPLES = 1,
    };

    HIPRT_HOST_DEVICE DevicePackedEffectiveMaterial()
    {
        flags.set_bool<PackedFlagsIndices::PACKED_EMISSIVE_TEXTURE_USED>(false);
        flags.set_bool<PackedFlagsIndices::PACKED_ENFORCE_STRONG_ENERGY_CONSERVATION>(false);
        flags.set_bool<PackedFlagsIndices::PACKED_THIN_FILM_DO_IOR_OVERRIDE>(false);
        flags.set_bool<PackedFlagsIndices::PACKED_THIN_WALLED>(false);

        base_color_roughness.set_color(ColorRGB32F(1.0f));
        base_color_roughness.set_float(0.5f);

        metallic_F90_and_metallic.set_float(0.0f);
        metallic_F90_and_metallic.set_color(ColorRGB32F(1.0f));
        metallic_F82_packed.set_color(ColorRGB32F(1.0f));

        anisotropy_and_rotation_and_second_roughness.set_float<PackedAnisotropyGroupIndices::PACKED_SECOND_ROUGHNESS>(0.5f);
        anisotropy_and_rotation_and_second_roughness.set_float<PackedAnisotropyGroupIndices::PACKED_SECOND_ROUGHNESS_WEIGHT>(0.0f);

        specular_color_and_tint_factor.set_color(ColorRGB32F(1.0f));
        specular_color_and_tint_factor.set_float(0.0f);

        specular_and_darkening_and_coat_roughness.set_float<PackedSpecularGroupIndices::PACKED_SPECULAR>(1.0f);
        specular_and_darkening_and_coat_roughness.set_float<PackedSpecularGroupIndices::PACKED_SPECULAR_DARKENING>(1.0f);

        coat_and_medium_absorption.set_color(ColorRGB32F(1.0f));

        coat_roughening_darkening_anisotropy_and_rotation.set_float<PackedCoatGroupIndices::PACKED_COAT_ROUGHENING>(1.0f);
        coat_roughening_darkening_anisotropy_and_rotation.set_float<PackedCoatGroupIndices::PACKED_COAT_DARKENING>(1.0f);

        sheen_and_color.set_color(ColorRGB32F(1.0f));

        absorption_color_packed.set_color(ColorRGB32F(1.0f));

        alpha_thin_film_hue_dielectric_priority.set_float<PackedAlphaOpacityGroupIndices::PACKED_ALPHA_OPACITY>(1.0f);
        alpha_thin_film_hue_dielectric_priority.set_float<PackedAlphaOpacityGroupIndices::PACKED_THIN_FILM_HUE_SHIFT>(0.35f);
        alpha_thin_film_hue_dielectric_priority.set_uchar<PackedAlphaOpacityGroupIndices::PACKED_ENERGY_PRESERVATION_SAMPLES>(12);
    }

    HIPRT_HOST_DEVICE bool is_emissive() const
    {
        return !hippt::is_zero(emission.r)
            || !hippt::is_zero(emission.g)
            || !hippt::is_zero(emission.b)
            || get_emissive_texture_used();
    }

    HIPRT_HOST_DEVICE ColorRGB32F get_emission() const { return this->emission; }
    HIPRT_HOST_DEVICE bool get_emissive_texture_used() const { return flags.get_bool<PackedFlagsIndices::PACKED_EMISSIVE_TEXTURE_USED>(); }

    HIPRT_HOST_DEVICE ColorRGB32F get_base_color() const { return base_color_roughness.get_color(); }

    HIPRT_HOST_DEVICE float get_roughness() const { return base_color_roughness.get_float(); }
    HIPRT_HOST_DEVICE float get_oren_nayar_sigma() const { return this->oren_nayar_sigma; }

    HIPRT_HOST_DEVICE float get_metallic() const { return metallic_F90_and_metallic.get_float(); }
    HIPRT_HOST_DEVICE float get_metallic_F90_falloff_exponent() const { return this->metallic_F90_falloff_exponent; }
    HIPRT_HOST_DEVICE ColorRGB32F get_metallic_F82() const { return metallic_F82_packed.get_color(); }
    HIPRT_HOST_DEVICE ColorRGB32F get_metallic_F90() const { return metallic_F90_and_metallic.get_color(); }
    HIPRT_HOST_DEVICE float get_anisotropy() const { return anisotropy_and_rotation_and_second_roughness.get_float<PackedAnisotropyGroupIndices::PACKED_ANISOTROPY>(); }
    HIPRT_HOST_DEVICE float get_anisotropy_rotation() const { return anisotropy_and_rotation_and_second_roughness.get_float<PackedAnisotropyGroupIndices::PACKED_ANISOTROPY_ROTATION>(); }
    HIPRT_HOST_DEVICE float get_second_roughness_weight() const { return anisotropy_and_rotation_and_second_roughness.get_float<PackedAnisotropyGroupIndices::PACKED_SECOND_ROUGHNESS_WEIGHT>(); }
    HIPRT_HOST_DEVICE float get_second_roughness() const { return anisotropy_and_rotation_and_second_roughness.get_float<PackedAnisotropyGroupIndices::PACKED_SECOND_ROUGHNESS>(); }

    HIPRT_HOST_DEVICE float get_specular() const { return specular_and_darkening_and_coat_roughness.get_float<PackedSpecularGroupIndices::PACKED_SPECULAR>(); }
    HIPRT_HOST_DEVICE float get_specular_tint() const { return specular_color_and_tint_factor.get_float(); }
    HIPRT_HOST_DEVICE ColorRGB32F get_specular_color() const { return specular_color_and_tint_factor.get_color(); }
    HIPRT_HOST_DEVICE float get_specular_darkening() const { return specular_and_darkening_and_coat_roughness.get_float<PackedSpecularGroupIndices::PACKED_SPECULAR_DARKENING>(); }

    HIPRT_HOST_DEVICE float get_coat() const { return coat_and_medium_absorption.get_float(); }
    HIPRT_HOST_DEVICE ColorRGB32F get_coat_medium_absorption() const { return coat_and_medium_absorption.get_color(); }
    HIPRT_HOST_DEVICE float get_coat_medium_thickness() const { return this->coat_medium_thickness; }
    HIPRT_HOST_DEVICE float get_coat_roughness() const { return specular_and_darkening_and_coat_roughness.get_float<PackedSpecularGroupIndices::PACKED_COAT_ROUGHNESS>(); }
    HIPRT_HOST_DEVICE float get_coat_roughening() const { return coat_roughening_darkening_anisotropy_and_rotation.get_float<PackedCoatGroupIndices::PACKED_COAT_ROUGHENING>(); }
    HIPRT_HOST_DEVICE float get_coat_darkening() const { return coat_roughening_darkening_anisotropy_and_rotation.get_float<PackedCoatGroupIndices::PACKED_COAT_DARKENING>(); }
    HIPRT_HOST_DEVICE float get_coat_anisotropy() const { return coat_roughening_darkening_anisotropy_and_rotation.get_float<PackedCoatGroupIndices::PACKED_COAT_ANISOTROPY>(); }
    HIPRT_HOST_DEVICE float get_coat_anisotropy_rotation() const { return coat_roughening_darkening_anisotropy_and_rotation.get_float<PackedCoatGroupIndices::PACKED_COAT_ANISOTROPY_ROTATION>(); }
    HIPRT_HOST_DEVICE float get_coat_ior() const { return this->coat_ior; }

    HIPRT_HOST_DEVICE float get_sheen() const { return sheen_and_color.get_float(); }
    HIPRT_HOST_DEVICE float get_sheen_roughness() const { return sheen_roughness_transmission_dispersion_thin_film.get_float<PackedSheenRoughnessGroupIndices::PACKED_SHEEN_ROUGHNESS>(); }
    HIPRT_HOST_DEVICE ColorRGB32F get_sheen_color() const { return sheen_and_color.get_color(); }

    HIPRT_HOST_DEVICE float get_ior() const { return this->ior; }
    HIPRT_HOST_DEVICE float get_specular_transmission() const { return sheen_roughness_transmission_dispersion_thin_film.get_float<PackedSheenRoughnessGroupIndices::PACKED_SPECULAR_TRANSMISSION>(); }
    HIPRT_HOST_DEVICE float get_absorption_at_distance() const { return this->absorption_at_distance; }
    HIPRT_HOST_DEVICE ColorRGB32F get_absorption_color() const { return absorption_color_packed.get_color(); }
    HIPRT_HOST_DEVICE float get_dispersion_scale() const { return sheen_roughness_transmission_dispersion_thin_film.get_float<PackedSheenRoughnessGroupIndices::PACKED_DISPERSION_SCALE>(); }
    HIPRT_HOST_DEVICE float get_dispersion_abbe_number() const { return this->dispersion_abbe_number; }
    HIPRT_HOST_DEVICE bool get_thin_walled() const { return flags.get_bool<PackedFlagsIndices::PACKED_THIN_WALLED >(); }

    HIPRT_HOST_DEVICE float get_thin_film() const { return sheen_roughness_transmission_dispersion_thin_film.get_float<PackedSheenRoughnessGroupIndices::PACKED_THIN_FILM>(); }
    HIPRT_HOST_DEVICE float get_thin_film_ior() const { return this->thin_film_ior; }
    HIPRT_HOST_DEVICE float get_thin_film_thickness() const { return this->thin_film_thickness; }
    HIPRT_HOST_DEVICE float get_thin_film_kappa_3() const { return this->thin_film_kappa_3; }
    HIPRT_HOST_DEVICE float get_thin_film_hue_shift_degrees() const { return alpha_thin_film_hue_dielectric_priority.get_float<PackedAlphaOpacityGroupIndices::PACKED_THIN_FILM_HUE_SHIFT>(); }
    HIPRT_HOST_DEVICE float get_thin_film_base_ior_override() const { return this->thin_film_base_ior_override; }
    HIPRT_HOST_DEVICE bool get_thin_film_do_ior_override() const { return flags.get_bool<PackedFlagsIndices::PACKED_THIN_FILM_DO_IOR_OVERRIDE>(); }

    HIPRT_HOST_DEVICE float get_alpha_opacity() const { return alpha_thin_film_hue_dielectric_priority.get_float<PackedAlphaOpacityGroupIndices::PACKED_ALPHA_OPACITY>(); }
    HIPRT_HOST_DEVICE unsigned char get_dielectric_priority() const { return alpha_thin_film_hue_dielectric_priority.get_uchar<PackedAlphaOpacityGroupIndices::PACKED_DIELECTRIC_PRIORITY>(); }

    HIPRT_HOST_DEVICE unsigned char get_energy_preservation_monte_carlo_samples() const { return alpha_thin_film_hue_dielectric_priority.get_uchar<PackedAlphaOpacityGroupIndices::PACKED_ENERGY_PRESERVATION_SAMPLES>(); }
    HIPRT_HOST_DEVICE bool get_enforce_strong_energy_conservation() const { return flags.get_bool<PackedFlagsIndices::PACKED_ENFORCE_STRONG_ENERGY_CONSERVATION>(); }




    HIPRT_HOST_DEVICE void set_emission(ColorRGB32F emission) { this->emission = emission; }
    HIPRT_HOST_DEVICE void set_emissive_texture_used(bool emissive_texture_used) { flags.set_bool<PackedFlagsIndices::PACKED_EMISSIVE_TEXTURE_USED>(emissive_texture_used); }

    HIPRT_HOST_DEVICE void set_base_color(ColorRGB32F base_color) { base_color_roughness.set_color(base_color); }

    HIPRT_HOST_DEVICE void set_roughness(float roughness) { base_color_roughness.set_float(roughness); }
    HIPRT_HOST_DEVICE void set_oren_nayar_sigma(float oren_nayar_sigma) { this->oren_nayar_sigma = oren_nayar_sigma; }

    HIPRT_HOST_DEVICE void set_metallic(float metallic) { metallic_F90_and_metallic.set_float(metallic); }
    HIPRT_HOST_DEVICE void set_metallic_F90_falloff_exponent(float metallic_F90_falloff_exponent) { this->metallic_F90_falloff_exponent = metallic_F90_falloff_exponent; }
    HIPRT_HOST_DEVICE void set_metallic_F82(ColorRGB32F metallic_F82) { metallic_F82_packed.set_color(metallic_F82); }
    HIPRT_HOST_DEVICE void set_metallic_F90(ColorRGB32F metallic_F90) { metallic_F90_and_metallic.set_color(metallic_F90); }
    HIPRT_HOST_DEVICE void set_anisotropy(float anisotropy) { anisotropy_and_rotation_and_second_roughness.set_float<PackedAnisotropyGroupIndices::PACKED_ANISOTROPY>(anisotropy); }
    HIPRT_HOST_DEVICE void set_anisotropy_rotation(float anisotropy_rotation) { anisotropy_and_rotation_and_second_roughness.set_float<PackedAnisotropyGroupIndices::PACKED_ANISOTROPY_ROTATION>(anisotropy_rotation); }
    HIPRT_HOST_DEVICE void set_second_roughness_weight(float second_roughness_weight) { anisotropy_and_rotation_and_second_roughness.set_float<PackedAnisotropyGroupIndices::PACKED_SECOND_ROUGHNESS_WEIGHT>(second_roughness_weight); }
    HIPRT_HOST_DEVICE void set_second_roughness(float second_roughness) { anisotropy_and_rotation_and_second_roughness.set_float<PackedAnisotropyGroupIndices::PACKED_SECOND_ROUGHNESS>(second_roughness); }

    HIPRT_HOST_DEVICE void set_specular(float specular) { specular_and_darkening_and_coat_roughness.set_float<PackedSpecularGroupIndices::PACKED_SPECULAR>(specular); }
    HIPRT_HOST_DEVICE void set_specular_tint(float specular_tint) { specular_color_and_tint_factor.set_float(specular_tint); }
    HIPRT_HOST_DEVICE void set_specular_color(ColorRGB32F specular_color) { specular_color_and_tint_factor.set_color(specular_color); }
    HIPRT_HOST_DEVICE void set_specular_darkening(float specular_darkening) { specular_and_darkening_and_coat_roughness.set_float<PackedSpecularGroupIndices::PACKED_SPECULAR_DARKENING>(specular_darkening); }

    HIPRT_HOST_DEVICE void set_coat(float coat) { coat_and_medium_absorption.set_float(coat); }
    HIPRT_HOST_DEVICE void set_coat_medium_absorption(ColorRGB32F coat_medium_absorption) { coat_and_medium_absorption.set_color(coat_medium_absorption); }
    HIPRT_HOST_DEVICE void set_coat_medium_thickness(float coat_medium_thickness) { this->coat_medium_thickness = coat_medium_thickness; }
    HIPRT_HOST_DEVICE void set_coat_roughness(float coat_roughness) { specular_and_darkening_and_coat_roughness.set_float<PackedSpecularGroupIndices::PACKED_COAT_ROUGHNESS>(coat_roughness); }
    HIPRT_HOST_DEVICE void set_coat_roughening(float coat_roughening) { coat_roughening_darkening_anisotropy_and_rotation.set_float<PackedCoatGroupIndices::PACKED_COAT_ROUGHENING>(coat_roughening); }
    HIPRT_HOST_DEVICE void set_coat_darkening(float coat_darkening) { coat_roughening_darkening_anisotropy_and_rotation.set_float<PackedCoatGroupIndices::PACKED_COAT_DARKENING>(coat_darkening); }
    HIPRT_HOST_DEVICE void set_coat_anisotropy(float coat_anisotropy) { coat_roughening_darkening_anisotropy_and_rotation.set_float<PackedCoatGroupIndices::PACKED_COAT_ANISOTROPY>(coat_anisotropy); }
    HIPRT_HOST_DEVICE void set_coat_anisotropy_rotation(float coat_anisotropy_rotation) { coat_roughening_darkening_anisotropy_and_rotation.set_float<PackedCoatGroupIndices::PACKED_COAT_ANISOTROPY_ROTATION>(coat_anisotropy_rotation); }
    HIPRT_HOST_DEVICE void set_coat_ior(float coat_ior) { this->coat_ior = coat_ior; }

    HIPRT_HOST_DEVICE void set_sheen(float sheen) { sheen_and_color.set_float(sheen); }
    HIPRT_HOST_DEVICE void set_sheen_roughness(float sheen_roughness) { sheen_roughness_transmission_dispersion_thin_film.set_float<PackedSheenRoughnessGroupIndices::PACKED_SHEEN_ROUGHNESS>(sheen_roughness); }
    HIPRT_HOST_DEVICE void set_sheen_color(ColorRGB32F sheen_color) { sheen_and_color.set_color(sheen_color); }

    HIPRT_HOST_DEVICE void set_ior(float ior) { this->ior = ior; }
    HIPRT_HOST_DEVICE void set_specular_transmission(float specular_transmission) { sheen_roughness_transmission_dispersion_thin_film.set_float<PackedSheenRoughnessGroupIndices::PACKED_SPECULAR_TRANSMISSION>(specular_transmission); }
    HIPRT_HOST_DEVICE void set_absorption_at_distance(float absorption_at_distance) { this->absorption_at_distance = absorption_at_distance; }
    HIPRT_HOST_DEVICE void set_absorption_color(ColorRGB32F absorption_color) { absorption_color_packed.set_color(absorption_color); }
    HIPRT_HOST_DEVICE void set_dispersion_scale(float dispersion_scale) { sheen_roughness_transmission_dispersion_thin_film.set_float<PackedSheenRoughnessGroupIndices::PACKED_DISPERSION_SCALE>(dispersion_scale); }
    HIPRT_HOST_DEVICE void set_dispersion_abbe_number(float dispersion_abbe_number) { this->dispersion_abbe_number = dispersion_abbe_number; }
    HIPRT_HOST_DEVICE void set_thin_walled(bool thin_walled) { flags.set_bool<PackedFlagsIndices::PACKED_THIN_WALLED >(thin_walled); }

    HIPRT_HOST_DEVICE void set_thin_film(float thin_film) { sheen_roughness_transmission_dispersion_thin_film.set_float<PackedSheenRoughnessGroupIndices::PACKED_THIN_FILM>(thin_film); }
    HIPRT_HOST_DEVICE void set_thin_film_ior(float thin_film_ior) { this->thin_film_ior = thin_film_ior; }
    HIPRT_HOST_DEVICE void set_thin_film_thickness(float thin_film_thickness) { this->thin_film_thickness = thin_film_thickness; }
    HIPRT_HOST_DEVICE void set_thin_film_kappa_3(float thin_film_kappa_3) { this->thin_film_kappa_3 = thin_film_kappa_3; }
    HIPRT_HOST_DEVICE void set_thin_film_hue_shift_degrees(float thin_film_hue_shift_degrees) { alpha_thin_film_hue_dielectric_priority.set_float<PackedAlphaOpacityGroupIndices::PACKED_THIN_FILM_HUE_SHIFT>(thin_film_hue_shift_degrees); }
    HIPRT_HOST_DEVICE void set_thin_film_base_ior_override(bool thin_film_base_ior_override) { this->thin_film_base_ior_override = thin_film_base_ior_override; }
    HIPRT_HOST_DEVICE void set_thin_film_do_ior_override(bool thin_film_do_ior_override) { flags.set_bool<PackedFlagsIndices::PACKED_THIN_FILM_DO_IOR_OVERRIDE>(thin_film_do_ior_override); }

    HIPRT_HOST_DEVICE void set_alpha_opacity(float alpha_opacity) { alpha_thin_film_hue_dielectric_priority.set_float<PackedAlphaOpacityGroupIndices::PACKED_ALPHA_OPACITY>(alpha_opacity); }
    HIPRT_HOST_DEVICE void set_dielectric_priority(unsigned char dielectric_priority) { alpha_thin_film_hue_dielectric_priority.set_uchar<PackedAlphaOpacityGroupIndices::PACKED_ALPHA_OPACITY>(dielectric_priority); }

    HIPRT_HOST_DEVICE void set_energy_preservation_monte_carlo_samples(unsigned char energy_preservation_monte_carlo_samples) { alpha_thin_film_hue_dielectric_priority.set_uchar<PackedAlphaOpacityGroupIndices::PACKED_ENERGY_PRESERVATION_SAMPLES>(energy_preservation_monte_carlo_samples); }
    HIPRT_HOST_DEVICE void set_enforce_strong_energy_conservation(bool enforce_strong_energy_conservation) { flags.set_bool<PackedFlagsIndices::PACKED_ENFORCE_STRONG_ENERGY_CONSERVATION>(enforce_strong_energy_conservation); }

private:
    // Packed flags of the material:
    //  - thin_walled
    //      Is the material thin walled? i.e. it doesn't have an interior and light doesn't
    //      bend as it goes through
    // 
    //  - emissive_texture_used
    //      Does the material use an emissive texture?
    // 
    //  - thin_film_do_ior_override
    //      Whether or not to override the IORs used for the base material on top of which
    //      the thin film sits.
    // 
    //  - enforce_strong_energy_conservation
    //      If true, 'energy_preservation_monte_carlo_samples' will be used
    //      to compute the directional albedo of this material.
    //      This computed directional albedo is then used to ensure perfect energy conservation
    //      and preservation. 
    // 
    //      This is however very expensive.
    //      This is usually only needed on clearcoated materials (but even then, the energy loss due to the absence of multiple scattering between
    //      the clearcoat layer and the BSDF below may be acceptable).
    // 
    //      Non-clearcoated materials can already ensure perfect (modulo implementation quality) energy 
    //      conservation/preservation with the precomputed LUTs [Turquin, 2019]. 
    // 
    //      See PrincipledBSDFGGXUseMultipleScattering in this codebase.
    //
    //      Values from the 'PackedFlagsIndices' enum should be used
    //      to retrieve/set from the packed flags
    UChar8BoolsPacked flags;

    // Full range emission
    ColorRGB32F emission = ColorRGB32F{ 0.0f, 0.0f, 0.0f };

    // Base color RGB 3x8 bits + roughness uchar [float in [0,1] packed in 8 bit]
    ColorRGB24bFloat0_1Packed base_color_roughness;

    float oren_nayar_sigma = 0.34906585039886591538f; // 20 degrees standard deviation in radian

    // Parameters for Adobe 2023 F82-tint model
    // Packs the SDR F90 color and the metalness parameter
    ColorRGB24bFloat0_1Packed metallic_F90_and_metallic;
    // TODO: PACKED FLOAT IS UNUSED IN HERE
    ColorRGB24bFloat0_1Packed metallic_F82_packed;
    float metallic_F90_falloff_exponent = 5.0f;

    Float4xPacked anisotropy_and_rotation_and_second_roughness;

    // Packed specular color and the intensity of the tint
    // 
    // Specular tint intensity: Specular will be white if 0.0f and will be 'specular_color' if 1.0f
    ColorRGB24bFloat0_1Packed specular_color_and_tint_factor;

    // Packed:
    //  - specular_darkening
    //      Same as coat darkening but for total internal reflection inside the specular layer
    //      that sits on top of the diffuse base
    //
    //      Disabled by default for artistic "expectations"
    //
    //  - Specular
    //      Specular intensity
    //
    //  - Coat roughness
    //      Roughness of the coat 
    // TODO: PACKED 1 FLOAT IS UNUSED IN HERE
    Float4xPacked specular_and_darkening_and_coat_roughness;
    float coat_medium_thickness = 5.0f;

    // Packed:
    //  - Coat
    //      Intensity of the coat. 0.0f disables the coating
    //
    //  - Coat medium absorption color 
    ColorRGB24bFloat0_1Packed coat_and_medium_absorption;

    // Packed:
    //  - Coat roughening
    //      Physical accuracy requires that a rough clearcoat also roughens what's underneath it
    //      i.e. the specular/metallic/transmission layers.
    // 
    //      The option is however given here to artistically disable
    //      that behavior by using coat roughening = 0.0f.
    //
    //  - Coat darkening
    //      Because of the total internal reflection that can happen inside the coat layer (i.e.
    //      light bouncing between the coat/BSDF and air/coat interfaces), the BSDF below the
    //      clearcoat will appear will increased saturation.
    //     
    //  - Coat anisotropy
    //  - Coat anisotropy rotation
    Float4xPacked coat_roughening_darkening_anisotropy_and_rotation;
    float coat_ior = 1.5f;

    // Packed:
    //  - Sheen intensity. 0.0f disables the sheen effect
    //
    //  - Sheen color
    ColorRGB24bFloat0_1Packed sheen_and_color;

    // IOR of the base material
    float ior = 1.40f;

    // Packed:
    //  - Absorption color
    //      Color of the light absorption when traveling through the medium
    // TODO: PACKED FLOAT IS UNUSED IN HERE
    ColorRGB24bFloat0_1Packed absorption_color_packed;
    float absorption_at_distance = 5.0f;

    // Packed:
    //  - Sheen roughness
    // 
    //  - Specular transmission
    //      How much light is transmitted through the material. This essentially controls the glass lobe
    //
    //  - Dispersion scale
    //      Intensity of the dispersion effect in glass objects
    //
    //  - Thin film
    //      Intensity of the thin-film effect
    Float4xPacked sheen_roughness_transmission_dispersion_thin_film;

    float dispersion_abbe_number = 20.0f;
    float thin_film_ior = 1.3f;
    float thin_film_thickness = 500.0f;
    float thin_film_kappa_3 = 0.0f;
    float thin_film_base_ior_override = 1.0f;

    // Packed:
    //  - Alpha opacity
    //      1.0f makes the material completely opaque
    //      0.0f completely transparent (becomes invisible)
    //
    //  - Thin film hue shift in degrees
    // 
    //  - Dielectric priority
    //      Nested dielectric with priority parameter
    //
    //  - Energy preservation samples
    //      How many samples will be computed for the integration of the directional
    //      when the strong energy preservation/conservation of the material is enabled
    Float2xUChar2xPacked alpha_thin_film_hue_dielectric_priority;
};

struct DevicePackedTexturedMaterial : public DevicePackedEffectiveMaterial
{
    enum NormalMapEmissionIndices : unsigned char
    {
        NORMAL_MAP_INDEX = 0,
        EMISSION_INDEX = 1,
    };

    enum BaseColorRoughnessMetallicIndices : unsigned char
    {
        BASE_COLOR_INDEX = 0,
        ROUGHNESS_METALLIC_INDEX = 1,
    };

    enum RoughnessAndMetallicIndices : unsigned char
    {
        ROUGHNESS_INDEX = 0,
        METALLIC_INDEX = 1,
    };

    enum AnisotropicSpecularIndices : unsigned char
    {
        ANISOTROPIC_INDEX = 0,
        SPECULAR_INDEX = 1,
    };

    enum CoatSheenIndices : unsigned char
    {
        COAT_INDEX = 0,
        SHEEN_INDEX = 1,
    };

    enum SpecularTransmissionIndex : unsigned char
    {
        SPECULAR_TRANSMISSION_INDEX = 0,
    };

    HIPRT_HOST_DEVICE DevicePackedTexturedMaterial()
    {
        set_normal_map_texture_index(MaterialUtils::NO_TEXTURE);
        set_emission_texture_index(MaterialUtils::NO_TEXTURE);
        set_base_color_texture_index(MaterialUtils::NO_TEXTURE);

        set_roughness_metallic_texture_index(MaterialUtils::NO_TEXTURE);
        set_roughness_texture_index(MaterialUtils::NO_TEXTURE);
        set_metallic_texture_index(MaterialUtils::NO_TEXTURE);
        set_anisotropic_texture_index(MaterialUtils::NO_TEXTURE);

        set_specular_texture_index(MaterialUtils::NO_TEXTURE);
        set_coat_texture_index(MaterialUtils::NO_TEXTURE);
        set_sheen_texture_index(MaterialUtils::NO_TEXTURE);
        set_specular_transmission_texture_index(MaterialUtils::NO_TEXTURE);
    }

    HIPRT_HOST_DEVICE DeviceUnpackedTexturedMaterial unpack()
    {
        DeviceUnpackedTexturedMaterial out;

        out.set_normal_map_texture_index(this->get_normal_map_texture_index());
        out.set_emission_texture_index(this->get_emission_texture_index());
        out.set_base_color_texture_index(this->get_base_color_texture_index());

        out.set_roughness_metallic_texture_index(this->get_roughness_metallic_texture_index());
        out.set_roughness_texture_index(this->get_roughness_texture_index());
        out.set_metallic_texture_index(this->get_metallic_texture_index());
        out.set_anisotropic_texture_index(this->get_anisotropic_texture_index());

        out.set_specular_texture_index(this->get_specular_texture_index());
        out.set_coat_texture_index(this->get_coat_texture_index());
        out.set_sheen_texture_index(this->get_sheen_texture_index());
        out.set_specular_transmission_texture_index(this->get_specular_transmission_texture_index());






        out.set_emission(this->get_emission());
        out.set_emissive_texture_used(this->get_emissive_texture_used());

        out.set_base_color(this->get_base_color());

        out.set_roughness(this->get_roughness());
        out.set_oren_nayar_sigma(this->get_oren_nayar_sigma());

        // Parameters for Adobe 2023 F82-tint model
        out.set_metallic(this->get_metallic());
        out.set_metallic_F90_falloff_exponent(this->get_metallic_F90_falloff_exponent());
        // F0 is not here as it uses the 'base_color' of the material
        out.set_metallic_F82(this->get_metallic_F82());
        out.set_metallic_F90(this->get_metallic_F90());
        out.set_anisotropy(this->get_anisotropy());
        out.set_anisotropy_rotation(this->get_anisotropy_rotation());
        out.set_second_roughness_weight(this->get_second_roughness_weight());
        out.set_second_roughness(this->get_second_roughness());

        // Specular intensity
        out.set_specular(this->get_specular());
        // Specular tint intensity. 
        // Specular will be white if 0.0f and will be 'specular_color' if 1.0f
        out.set_specular_tint(this->get_specular_tint());
        out.set_specular_color(this->get_specular_color());
        // Same as coat darkening but for total internal reflection inside the specular layer
        // that sits on top of the diffuse base
        //
        // Disabled by default for artistic "expectations"
        out.set_specular_darkening(this->get_specular_darkening());

        out.set_coat(this->get_coat());
        out.set_coat_medium_absorption(this->get_coat_medium_absorption());
        // The coat thickness influences the amount of absorption (given by 'coat_medium_absorption')
        // that will happen inside the coat
        out.set_coat_medium_thickness(this->get_coat_medium_thickness());
        out.set_coat_roughness(this->get_coat_roughness());
        // Physical accuracy requires that a rough clearcoat also roughens what's underneath it
        // i.e. the specular/metallic/transmission layers.
        // 
        // The option is however given here to artistically disable
        // that behavior by using coat roughening = 0.0f.
        out.set_coat_roughening(this->get_coat_roughening());
        // Because of the total internal reflection that can happen inside the coat layer (i.e.
        // light bouncing between the coat/BSDF and air/coat interfaces), the BSDF below the
        // clearcoat will appear will increased saturation.
        out.set_coat_darkening(this->get_coat_darkening());
        out.set_coat_anisotropy(this->get_coat_anisotropy());
        out.set_coat_anisotropy_rotation(this->get_coat_anisotropy_rotation());
        out.set_coat_ior(this->get_coat_ior());

        out.set_sheen(this->get_sheen()); // Sheen strength
        out.set_sheen_roughness(this->get_sheen_roughness());
        out.set_sheen_color(this->get_sheen_color());

        out.set_ior(this->get_ior());
        out.set_specular_transmission(this->get_specular_transmission());

        // At what distance is the light absorbed to the given absorption_color
        out.set_absorption_at_distance(this->get_absorption_at_distance());
        // Color of the light absorption when traveling through the medium
        out.set_absorption_color(this->get_absorption_color());
        out.set_dispersion_scale(this->get_dispersion_scale());
        out.set_dispersion_abbe_number(this->get_dispersion_abbe_number());
        out.set_thin_walled(this->get_thin_walled());

        out.set_thin_film(this->get_thin_film());
        out.set_thin_film_ior(this->get_thin_film_ior());
        out.set_thin_film_thickness(this->get_thin_film_thickness());
        out.set_thin_film_kappa_3(this->get_thin_film_kappa_3());
        // Sending the hue film in [0, 1] to the GPU
        out.set_thin_film_hue_shift_degrees(this->get_thin_film_hue_shift_degrees());
        out.set_thin_film_base_ior_override(this->get_thin_film_base_ior_override());
        out.set_thin_film_do_ior_override(this->get_thin_film_do_ior_override());

        // 1.0f makes the material completely opaque
        // 0.0f completely transparent (becomes invisible)
        out.set_alpha_opacity(this->get_alpha_opacity());

        // Nested dielectric parameter
        out.set_dielectric_priority(this->get_dielectric_priority());

        out.set_energy_preservation_monte_carlo_samples(this->get_energy_preservation_monte_carlo_samples());
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
        // See PrincipledBSDFGGXUseMultipleScattering in this codebase.
        out.set_enforce_strong_energy_conservation(this->get_enforce_strong_energy_conservation());

        return out;
    }

    HIPRT_HOST_DEVICE unsigned short int get_normal_map_texture_index() const { return normal_map_emission_index.get_value<NormalMapEmissionIndices::NORMAL_MAP_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_emission_texture_index() const { return normal_map_emission_index.get_value<NormalMapEmissionIndices::EMISSION_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_base_color_texture_index() const { return base_color_roughness_metallic_index.get_value<BaseColorRoughnessMetallicIndices::BASE_COLOR_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_roughness_metallic_texture_index() const { return base_color_roughness_metallic_index.get_value<BaseColorRoughnessMetallicIndices::ROUGHNESS_METALLIC_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_roughness_texture_index() const { return roughness_and_metallic_index.get_value<RoughnessAndMetallicIndices::ROUGHNESS_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_metallic_texture_index() const { return roughness_and_metallic_index.get_value<RoughnessAndMetallicIndices::METALLIC_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_anisotropic_texture_index() const { return roughness_and_metallic_index.get_value<AnisotropicSpecularIndices::ANISOTROPIC_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_specular_texture_index() const { return anisotropic_specular_index.get_value<AnisotropicSpecularIndices::SPECULAR_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_coat_texture_index() const { return coat_sheen_index.get_value<CoatSheenIndices::COAT_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_sheen_texture_index() const { return coat_sheen_index.get_value<CoatSheenIndices::SHEEN_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_specular_transmission_texture_index() const { return specular_transmission_index_packed.get_value<SpecularTransmissionIndex::SPECULAR_TRANSMISSION_INDEX>(); }

    HIPRT_HOST_DEVICE void set_normal_map_texture_index(unsigned short normal_map_index) { normal_map_emission_index.set_value<NormalMapEmissionIndices::NORMAL_MAP_INDEX>(normal_map_index); }
    HIPRT_HOST_DEVICE void set_emission_texture_index(unsigned short emission_index) { normal_map_emission_index.set_value<NormalMapEmissionIndices::EMISSION_INDEX>(emission_index); }
    HIPRT_HOST_DEVICE void set_base_color_texture_index(unsigned short base_color_index) { base_color_roughness_metallic_index.set_value<BaseColorRoughnessMetallicIndices::BASE_COLOR_INDEX>(base_color_index); }
    HIPRT_HOST_DEVICE void set_roughness_metallic_texture_index(unsigned short roughness_metallic_index) { base_color_roughness_metallic_index.set_value<BaseColorRoughnessMetallicIndices::ROUGHNESS_METALLIC_INDEX>(roughness_metallic_index); }
    HIPRT_HOST_DEVICE void set_roughness_texture_index(unsigned short roughness_index) { roughness_and_metallic_index.set_value<RoughnessAndMetallicIndices::ROUGHNESS_INDEX>(roughness_index); }
    HIPRT_HOST_DEVICE void set_metallic_texture_index(unsigned short metallic_index) { roughness_and_metallic_index.set_value<RoughnessAndMetallicIndices::METALLIC_INDEX>(metallic_index); }
    HIPRT_HOST_DEVICE void set_anisotropic_texture_index(unsigned short anisotropic_index) { roughness_and_metallic_index.set_value<AnisotropicSpecularIndices::ANISOTROPIC_INDEX>(anisotropic_index); }
    HIPRT_HOST_DEVICE void set_specular_texture_index(unsigned short specular_index) { anisotropic_specular_index.set_value<AnisotropicSpecularIndices::SPECULAR_INDEX>(specular_index); }
    HIPRT_HOST_DEVICE void set_coat_texture_index(unsigned short coat_index) { coat_sheen_index.set_value<CoatSheenIndices::COAT_INDEX>(coat_index); }
    HIPRT_HOST_DEVICE void set_sheen_texture_index(unsigned short sheen_index) { coat_sheen_index.set_value<CoatSheenIndices::SHEEN_INDEX>(sheen_index); }
    HIPRT_HOST_DEVICE void set_specular_transmission_texture_index(unsigned short specular_transmission_index) { specular_transmission_index_packed.set_value<SpecularTransmissionIndex::SPECULAR_TRANSMISSION_INDEX>(specular_transmission_index); }

private:
    Uint2xPacked normal_map_emission_index;
    // If the roughness_metallic texture index is not MaterialUtils::NO_TEXTURE, 
    // then there is only one texture for the metallic and the roughness parameters in which.
    // case the green channel is the roughness and the blue channel is the metalness.
    Uint2xPacked base_color_roughness_metallic_index;
    Uint2xPacked roughness_and_metallic_index;
    Uint2xPacked anisotropic_specular_index;
    Uint2xPacked coat_sheen_index;
    // TODO: 1 PACKED UINT IS UNUSED IN HERE
    Uint2xPacked specular_transmission_index_packed;
};

#if MaterialPackingStrategy == MATERIAL_PACK_STRATEGY_USE_PACKED
using DeviceEffectiveMaterial = DevicePackedEffectiveMaterial;
#elif MaterialPackingStrategy == MATERIAL_PACK_STRATEGY_USE_UNPACKED
using DeviceEffectiveMaterial = DeviceUnpackedEffectiveMaterial;
#endif

#if MaterialPackingStrategy == MATERIAL_PACK_STRATEGY_USE_PACKED
using DeviceTexturedMaterial = DevicePackedTexturedMaterial;
#elif MaterialPackingStrategy == MATERIAL_PACK_STRATEGY_USE_UNPACKED
using DeviceTexturedMaterial = DeviceUnpackedTexturedMaterial;
#endif

// Adding this guard to make sure that we never use the CPU materials in the GPU code
#ifndef  __KERNELCC__

// This material structure is only used on the CPU.
// The reason why we have a CPU and a GPU material is because
// we may want to precompute some properties on the CPU before sending them
// to the GPU. This means that the CPU material stores both the precomputed values
// and the values needed for the precomputation itself.
//
// But the GPU only cares about the precomputed values itself, not the ingredients
// to the precomputation so that's why we have separate structures
struct CPUMaterial
{
    DevicePackedTexturedMaterial pack_to_GPU() const
    {
        DevicePackedTexturedMaterial mat;

        mat.set_normal_map_texture_index(this->normal_map_texture_index);
        mat.set_emission_texture_index(this->emission_texture_index);
        mat.set_base_color_texture_index(this->base_color_texture_index);

        mat.set_roughness_metallic_texture_index(this->roughness_metallic_texture_index);
        mat.set_roughness_texture_index(this->roughness_texture_index);
        mat.set_metallic_texture_index(this->metallic_texture_index);
        mat.set_anisotropic_texture_index(this->anisotropic_texture_index);

        mat.set_specular_texture_index(this->specular_texture_index);
        mat.set_coat_texture_index(this->coat_texture_index);
        mat.set_sheen_texture_index(this->sheen_texture_index);
        mat.set_specular_transmission_texture_index(this->specular_transmission_texture_index);






        mat.set_emission(emission * emission_strength * global_emissive_factor);
        mat.set_emissive_texture_used(emissive_texture_used);

        mat.set_base_color(base_color);

        mat.set_roughness(roughness);
        mat.set_oren_nayar_sigma(oren_nayar_sigma);

        // Parameters for Adobe 2023 F82-tint model
        mat.set_metallic(metallic);
        mat.set_metallic_F90_falloff_exponent(metallic_F90_falloff_exponent);
        // F0 is not here as it uses the 'base_color' of the material
        mat.set_metallic_F82(metallic_F82);
        mat.set_metallic_F90(metallic_F90);
        mat.set_anisotropy(anisotropy);
        mat.set_anisotropy_rotation(anisotropy_rotation);
        mat.set_second_roughness_weight(second_roughness_weight);
        mat.set_second_roughness(second_roughness);

        // Specular intensity
        mat.set_specular(specular);
        // Specular tint intensity. 
        // Specular will be white if 0.0f and will be 'specular_color' if 1.0f
        mat.set_specular_tint(specular_tint);
        mat.set_specular_color(specular_color);
        // Same as coat darkening but for total internal reflection inside the specular layer
        // that sits on top of the diffuse base
        //
        // Disabled by default for artistic "expectations"
        mat.set_specular_darkening(specular_darkening);

        mat.set_coat(coat);
        mat.set_coat_medium_absorption(coat_medium_absorption);
        // The coat thickness influences the amount of absorption (given by 'coat_medium_absorption')
        // that will happen inside the coat
        mat.set_coat_medium_thickness(coat_medium_thickness);
        mat.set_coat_roughness(coat_roughness);
        // Physical accuracy requires that a rough clearcoat also roughens what's underneath it
        // i.e. the specular/metallic/transmission layers.
        // 
        // The option is however given here to artistically disable
        // that behavior by using coat roughening = 0.0f.
        mat.set_coat_roughening(coat_roughening);
        // Because of the total internal reflection that can happen inside the coat layer (i.e.
        // light bouncing between the coat/BSDF and air/coat interfaces), the BSDF below the
        // clearcoat will appear will increased saturation.
        mat.set_coat_darkening(coat_darkening);
        mat.set_coat_anisotropy(coat_anisotropy);
        mat.set_coat_anisotropy_rotation(coat_anisotropy_rotation);
        mat.set_coat_ior(coat_ior);

        mat.set_sheen(sheen); // Sheen strength
        mat.set_sheen_roughness(sheen_roughness);
        mat.set_sheen_color(sheen_color);

        mat.set_ior(ior);
        mat.set_specular_transmission(specular_transmission);

        // At what distance is the light absorbed to the given absorption_color
        mat.set_absorption_at_distance(absorption_at_distance);
        // Color of the light absorption when traveling through the medium
        mat.set_absorption_color(absorption_color);
        mat.set_dispersion_scale(dispersion_scale);
        mat.set_dispersion_abbe_number(dispersion_abbe_number);
        mat.set_thin_walled(thin_walled);

        mat.set_thin_film(thin_film);
        mat.set_thin_film_ior(thin_film_ior);
        mat.set_thin_film_thickness(thin_film_thickness);
        mat.set_thin_film_kappa_3(thin_film_kappa_3);
        // Sending the hue film in [0, 1] to the GPU
        mat.set_thin_film_hue_shift_degrees(thin_film_hue_shift_degrees / 360.0f);
        mat.set_thin_film_base_ior_override(thin_film_base_ior_override);
        mat.set_thin_film_do_ior_override(thin_film_do_ior_override);

        // 1.0f makes the material completely opaque
        // 0.0f completely transparent (becomes invisible)
        mat.set_alpha_opacity(alpha_opacity);

        // Nested dielectric parameter
        mat.set_dielectric_priority(dielectric_priority);

        mat.set_energy_preservation_monte_carlo_samples(energy_preservation_monte_carlo_samples);
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
        // See PrincipledBSDFGGXUseMultipleScattering in this codebase.
        mat.set_enforce_strong_energy_conservation(enforce_strong_energy_conservation);

        return mat;
    }

    HIPRT_HOST_DEVICE bool is_emissive() const
    {
        return !hippt::is_zero(emission.r)
            || !hippt::is_zero(emission.g)
            || !hippt::is_zero(emission.b)
            || emissive_texture_used;
    }

    /*
     * Clamps some of the parameters of the material to avoid edge cases like NaNs
     * during rendering (i.e. numerical instabilities)
     */
    HIPRT_HOST_DEVICE void make_safe()
    {
        roughness = hippt::max(MaterialUtils::ROUGHNESS_CLAMP, roughness);
        coat_roughness = hippt::max(MaterialUtils::ROUGHNESS_CLAMP, coat_roughness);
        sheen_roughness = hippt::max(MaterialUtils::ROUGHNESS_CLAMP, sheen_roughness);

        // Clamping to avoid negative emission
        emission = ColorRGB32F::max(ColorRGB32F(0.0f), emission);

        // Avoiding zero
        absorption_at_distance = hippt::max(absorption_at_distance, 1.0e-4f);
        absorption_color = ColorRGB32F::max(absorption_color, ColorRGB32F(1.0f / 512.0f));

        thin_film_ior = hippt::max(1.0005f, thin_film_ior);
    }

    /*
     * Some properties of the material can be precomputed
     * This function does it
     */
    HIPRT_HOST_DEVICE void precompute_properties()
    {
        if (specular_transmission == 0.0f)
            // No transmission means that we should never skip this boundary --> max priority
            dielectric_priority = (1 << StackPriorityEntry::PRIORITY_MAXIMUM) - 1;
    }





    ColorRGB32F emission = ColorRGB32F{ 0.0f, 0.0f, 0.0f };
    float emission_strength = 1.0f; // This factor is baked into 'emission' before being sent to the GPU
    float global_emissive_factor = 1.0f; // This factor is baked into 'emission' before being sent to the GPU
    bool emissive_texture_used = false;

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
    // At what distance is the light absorbed to the given absorption_color
    float absorption_at_distance = 1.0f;
    // Color of the light absorption when traveling through the medium
    ColorRGB32F absorption_color = ColorRGB32F(1.0f);
    float dispersion_scale = 0.0f;
    float dispersion_abbe_number = 20.0f;
    bool thin_walled = false;

    float thin_film = 0.0f;
    float thin_film_ior = 1.3f;
    float thin_film_thickness = 500.0f;
    float thin_film_kappa_3 = 0.0f;
    float thin_film_hue_shift_degrees = 0.0f;
    float thin_film_base_ior_override = 1.0f;
    bool thin_film_do_ior_override = false;

    // 1.0f makes the material completely opaque
    // 0.0f completely transparent (becomes invisible)
    float alpha_opacity = 1.0f;

    // Nested dielectric parameter
    int dielectric_priority = 0;

    int energy_preservation_monte_carlo_samples = 12;
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
    // See PrincipledBSDFGGXUseMultipleScattering in this codebase.
    bool enforce_strong_energy_conservation = false;





    int normal_map_texture_index = MaterialUtils::NO_TEXTURE;

    int emission_texture_index = MaterialUtils::NO_TEXTURE;
    int base_color_texture_index = MaterialUtils::NO_TEXTURE;

    // If not MaterialUtils::NO_TEXTURE, there is only one texture for the metallic and the roughness parameters in which.
    // case the green channel is the roughness and the blue channel is the metalness
    int roughness_metallic_texture_index = MaterialUtils::NO_TEXTURE;
    int roughness_texture_index = MaterialUtils::NO_TEXTURE;
    int metallic_texture_index = MaterialUtils::NO_TEXTURE;
    int anisotropic_texture_index = MaterialUtils::NO_TEXTURE;

    int specular_texture_index = MaterialUtils::NO_TEXTURE;
    int coat_texture_index = MaterialUtils::NO_TEXTURE;
    int sheen_texture_index = MaterialUtils::NO_TEXTURE;
    int specular_transmission_texture_index = MaterialUtils::NO_TEXTURE;
};

#endif // #ifndef  __KERNELCC__

#endif
