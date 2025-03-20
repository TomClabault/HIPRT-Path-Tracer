/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_UTILS_H
#define HOST_DEVICE_COMMON_MATERIAL_UTILS_H

#include "Device/includes/BSDFs/BSDFIncidentLightInfo.h"

#include "HostDeviceCommon/Material/MaterialConstants.h"
#include "HostDeviceCommon/Material/MaterialPacked.h"
#include "HostDeviceCommon/Material/MaterialUnpacked.h"
#include "HostDeviceCommon/KernelOptions/PrincipledBSDFKernelOptions.h"

struct MaterialUtils
{
    HIPRT_HOST_DEVICE static void get_oren_nayar_AB(float sigma, float& out_oren_A, float& out_oren_B)
    {
        float sigma2 = sigma * sigma;
        out_oren_A = 1.0f - sigma2 / (2.0f * (sigma2 + 0.33f));
        out_oren_B = 0.45f * sigma2 / (sigma2 + 0.09f);
    }

    HIPRT_HOST_DEVICE static void get_alphas(float roughness, float anisotropy, float& out_alpha_x, float& out_alpha_y)
    {
        float aspect = sqrtf(1.0f - 0.9f * anisotropy);
        out_alpha_x = hippt::max(MaterialConstants::ROUGHNESS_CLAMP, roughness * roughness / aspect);
        out_alpha_y = hippt::max(MaterialConstants::ROUGHNESS_CLAMP, roughness * roughness * aspect);
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

    HIPRT_HOST_DEVICE static bool is_perfectly_smooth(float roughness, float roughness_threshold = MaterialConstants::PERFECTLY_SMOOTH_ROUGHNESS_THRESHOLD)
    {
        return roughness <= roughness_threshold;
    }

    /**
     * Whether or not it makes sense to even try light sampling with NEE on that material
     *
     * Perfectly smooth materials for example cannot do light sampling because no given light
     * direction is going to align with the delta distribution peak of the BRDF so we can save
     * some performance by not even attempting light sampling in the first place
     */
    HIPRT_HOST_DEVICE static bool can_do_light_sampling(float material_roughness, float material_metallic, float material_specular_transmission, float material_coat, float material_coat_roughness, float material_second_roughness, float material_second_roughness_weight, float roughness_threshold)
    {
#if DirectLightSamplingDeltaDistributionOptimization == KERNEL_OPTION_FALSE || PrincipledBSDFDoMicrofacetRegularization == KERNEL_OPTION_TRUE
        return true;
#endif

#if BSDFOverride == BSDF_LAMBERTIAN || BSDFOverride == BSDF_OREN_NAYAR
        // We can always do light sampling on these BSDFs
        return true;
#endif

        bool smooth_base_layer = MaterialUtils::is_perfectly_smooth(material_roughness, roughness_threshold) && (material_metallic == 1.0f || material_specular_transmission == 1.0f);
        bool smooth_coat = material_coat == 0.0f || (material_coat > 0.0f && MaterialUtils::is_perfectly_smooth(material_coat_roughness, roughness_threshold));
        bool second_roughness_smooth = MaterialUtils::is_perfectly_smooth(material_second_roughness, roughness_threshold) || material_second_roughness_weight == 0.0f;
        if (smooth_base_layer && smooth_coat && second_roughness_smooth)
            // Everything is smooth
            return false;

        return true;
    }

    HIPRT_HOST_DEVICE static bool can_do_light_sampling(const DeviceUnpackedEffectiveMaterial& material, float roughness_threshold = MaterialConstants::PERFECTLY_SMOOTH_ROUGHNESS_THRESHOLD)
    {
        return can_do_light_sampling(material.roughness, material.metallic, material.specular_transmission, material.coat, material.coat_roughness, material.second_roughness, material.second_roughness_weight, roughness_threshold);
    }

    HIPRT_HOST_DEVICE static bool can_do_light_sampling(const DevicePackedEffectiveMaterial& material, float roughness_threshold = MaterialConstants::PERFECTLY_SMOOTH_ROUGHNESS_THRESHOLD)
    {
        return can_do_light_sampling(material.get_roughness(), material.get_metallic(), material.get_specular_transmission(), material.get_coat(), material.get_coat_roughness(), material.get_second_roughness(), material.get_second_roughness_weight(), roughness_threshold);
    }

    enum SpecularDeltaReflectionSampled : int
    {
        NOT_SPECULAR = -1,
        SPECULAR_PEAK_NOT_SAMPLED = 0,
        SPECULAR_PEAK_SAMPLED = 1,
    };

    /**
     * Determines whether a perfectly smooth lobe has any chance of evaluating to non-0.
     * 
     * This is only relevant for perfectly smooth materials/lobe where we don't want to evaluate the specular BRDF
     * with anything other than a direction that was sampled directly from that specular BRDF.
     * 
     * The 'delta_distribution_oughness' and 'delta_distribution_anisotropy' parameters here describe the BRDF lobe
     * that is being evaluated.
     * 
     * 'incident_light_info' is some additional information about the incident light direction used for
     * evaluating the current lobe
     * 
     * Returns 1 only if the specular distribution is worth evaluating, 0 if there's no point because it's going to
     * evaluate to 0 anyways
     * 
     * Returns -1 if the distribution given isn't specular in the first place (delta_distribution_roughness isn't very close to 0)
     */
    HIPRT_HOST_DEVICE static SpecularDeltaReflectionSampled is_specular_delta_reflection_sampled(const DeviceUnpackedEffectiveMaterial& material, float delta_distribution_roughness, float delta_distribution_anisotropy, BSDFIncidentLightInfo incident_light_info)
    {
        if (!MaterialUtils::is_perfectly_smooth(delta_distribution_roughness))
            return SpecularDeltaReflectionSampled::NOT_SPECULAR;

        // For the glass lobe sampled direction to match, we only need it to be a reflection
        // and we need the glass lobe to be perfectly smooth
        bool matching_base_substrate_anisotropy = hippt::abs(delta_distribution_anisotropy - material.anisotropy) < 1.0e-3f;
        bool sampled_from_glass = incident_light_info == BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFLECT_LOBE && MaterialUtils::is_perfectly_smooth(material.roughness) && matching_base_substrate_anisotropy;
        if (sampled_from_glass)
            // We can stop here
            return SpecularDeltaReflectionSampled::SPECULAR_PEAK_SAMPLED;

        // Same for the metal lobe (except that it's alawys a reflection, so it's easy there)
        bool sampled_from_first_metal = incident_light_info == BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_FIRST_METAL_LOBE && MaterialUtils::is_perfectly_smooth(material.roughness) && matching_base_substrate_anisotropy;
        bool sampled_from_second_metal = incident_light_info == BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_SECOND_METAL_LOBE && MaterialUtils::is_perfectly_smooth(material.second_roughness) && matching_base_substrate_anisotropy;
        if (sampled_from_first_metal || sampled_from_second_metal)
            // We can stop here
            return SpecularDeltaReflectionSampled::SPECULAR_PEAK_SAMPLED;

        // Same for the coat
        bool matching_coat_anisotropy = hippt::abs(delta_distribution_anisotropy - material.coat_anisotropy) < 1.0e-3f;
        bool sampled_from_coat = incident_light_info == BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_COAT_LOBE && matching_coat_anisotropy && MaterialUtils::is_perfectly_smooth(material.coat_roughness);
        if (sampled_from_coat)
            // We can stop here
            return SpecularDeltaReflectionSampled::SPECULAR_PEAK_SAMPLED;

        // Same for the specular layer
        bool sampled_from_specular = incident_light_info == BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_SPECULAR_LOBE && MaterialUtils::is_perfectly_smooth(material.roughness) && matching_base_substrate_anisotropy;
        if (sampled_from_specular)
            // We can stop here
            return SpecularDeltaReflectionSampled::SPECULAR_PEAK_SAMPLED;

        return SpecularDeltaReflectionSampled::SPECULAR_PEAK_NOT_SAMPLED;
    }
};

#endif
