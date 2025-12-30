/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_SHADING_PAIRWISE_MIS_H
#define DEVICE_KERNELS_REGIR_SHADING_PAIRWISE_MIS_H

#include "Device/includes/LightSampling/TriangleSampling.h"
#include "Device/includes/ReSTIR/ReGIR/TargetFunction.h"

template <bool canonicalPDF>
HIPRT_DEVICE float ReGIR_get_reservoir_sample_ReGIR_PDF(const HIPRTRenderData& render_data, const ReGIRGridFillSurface& surface, bool primary_hit, float PDF_normalization, float3 point_on_light, float3 light_source_normal, ColorRGB32F emission, Xorshift32Generator& random_number_generator)
{
    float sample_PDF_unnormalized;
    if constexpr (canonicalPDF)
        sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_canonical_target_function(render_data, surface, primary_hit, emission, light_source_normal, point_on_light, random_number_generator);
    else
        sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, surface, primary_hit, emission, light_source_normal, point_on_light, random_number_generator);

    return sample_PDF_unnormalized / PDF_normalization;
}

template <bool canonicalPDF>
HIPRT_DEVICE float ReGIR_get_reservoir_sample_ReGIR_PDF(const HIPRTRenderData& render_data,
    const ReGIRGridFillSurface& surface, unsigned int grid_cell_index, bool primary_hit,
    float3 point_on_light, float3 light_source_normal, ColorRGB32F emission, Xorshift32Generator& random_number_generator)
{
    float RIS_integral;
    if constexpr (canonicalPDF)
        RIS_integral = render_data.render_settings.regir_settings.get_canonical_pre_integration_factor(grid_cell_index, primary_hit);
    else
        RIS_integral = render_data.render_settings.regir_settings.get_non_canonical_pre_integration_factor(grid_cell_index, primary_hit);
    if (RIS_integral == 0.0f)
        RIS_integral = 1.0f;

    return ReGIR_get_reservoir_sample_ReGIR_PDF<canonicalPDF>(render_data, surface, primary_hit, RIS_integral, point_on_light, light_source_normal, emission, random_number_generator);
}

template <bool canonicalPDF>
HIPRT_DEVICE float ReGIR_get_reservoir_sample_ReGIR_PDF(const HIPRTRenderData& render_data, float3 point_on_light, float3 light_source_normal, ColorRGB32F emission, unsigned int grid_cell_index, bool primary_hit, Xorshift32Generator& random_number_generator)
{
    if (emission.is_black())
        return 0.0f;

    ReGIRGridFillSurface surface = ReGIR_get_cell_surface(render_data, grid_cell_index, primary_hit);
    return ReGIR_get_reservoir_sample_ReGIR_PDF<canonicalPDF>(render_data, surface, grid_cell_index, primary_hit, point_on_light, light_source_normal, emission, random_number_generator);
}

template <bool canonicalPDF>
HIPRT_DEVICE float ReGIR_get_reservoir_sample_ReGIR_PDF(const HIPRTRenderData& render_data, float3 point_on_light, float3 light_source_normal, ColorRGB32F emission, unsigned int grid_cell_index, float RIS_integral, bool primary_hit, Xorshift32Generator& random_number_generator)
{
    if (emission.is_black())
        return 0.0f;

    ReGIRGridFillSurface surface = ReGIR_get_cell_surface(render_data, grid_cell_index, primary_hit);
    return ReGIR_get_reservoir_sample_ReGIR_PDF<canonicalPDF>(render_data, surface, primary_hit, RIS_integral, point_on_light, light_source_normal, emission, random_number_generator);
}

template <bool canonicalPDF>
HIPRT_DEVICE float ReGIR_get_reservoir_sample_ReGIR_PDF(const HIPRTRenderData& render_data, const ReGIRReservoir& reservoir, unsigned int grid_cell_index, bool primary_hit, Xorshift32Generator& random_number_generator)
{
    if (reservoir.UCW <= 0.0f)
        return 0.0f;

    float3 light_source_normal = hippt::normalize(triangle_load_normal_not_normalized(render_data, reservoir.sample.emissive_triangle_global_index));
    float3 point_on_light = reservoir.sample.point_on_light;
    ColorRGB32F emission = triangle_load_emission(render_data, reservoir.sample.emissive_triangle_global_index);

    return ReGIR_get_reservoir_sample_ReGIR_PDF<canonicalPDF>(render_data, point_on_light, light_source_normal, emission, grid_cell_index, primary_hit, random_number_generator);
}

template <bool canonicalPDF>
HIPRT_DEVICE float ReGIR_get_reservoir_sample_ReGIR_PDF(const HIPRTRenderData& render_data, const ReGIRReservoir& reservoir, unsigned int grid_cell_index, float RIS_integral, bool primary_hit, Xorshift32Generator& random_number_generator)
{
    if (reservoir.UCW <= 0.0f)
        return 0.0f;

    float3 light_source_normal = hippt::normalize(triangle_load_normal_not_normalized(render_data, reservoir.sample.emissive_triangle_global_index));
    float3 point_on_light = reservoir.sample.point_on_light;
    ColorRGB32F emission = triangle_load_emission(render_data, reservoir.sample.emissive_triangle_global_index);

    return ReGIR_get_reservoir_sample_ReGIR_PDF<canonicalPDF>(render_data, point_on_light, light_source_normal, emission, grid_cell_index, RIS_integral, primary_hit, random_number_generator);
}

HIPRT_DEVICE float ReGIR_get_reservoir_sample_BSDF_PDF(const HIPRTRenderData& render_data,
    float3 point_on_light, float3 light_source_normal, ColorRGB32F emission,
    float3 view_direction, float3 shading_point, float3 shading_normal, float3 geometric_normal, BSDFIncidentLightInfo incident_light_info, RayPayload& ray_payload, int last_hit_primitive_index)
{
    if (emission.is_black())
        return 0.0f;

    float3 to_light_direction = point_on_light - shading_point;
    float distance_to_light = hippt::length(to_light_direction);
    to_light_direction /= distance_to_light; // Normalization

    BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, to_light_direction, incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
    float bsdf_pdf = bsdf_dispatcher_pdf(render_data, bsdf_context);

    float area_measure_bsdf_pdf = solid_angle_to_area_pdf(bsdf_pdf, distance_to_light, compute_cosine_term_at_light_source(light_source_normal, -to_light_direction));

    return area_measure_bsdf_pdf;
}

HIPRT_DEVICE float ReGIR_get_reservoir_sample_BSDF_PDF(const HIPRTRenderData& render_data, const ReGIRReservoir& reservoir,
    float3 view_direction, float3 shading_point, float3 shading_normal, float3 geometric_normal, BSDFIncidentLightInfo incident_light_info, RayPayload& ray_payload, int last_hit_primitive_index, Xorshift32Generator& random_number_generator)
{
    if (reservoir.UCW <= 0.0f)
        return 0.0f;

    float3 light_source_normal = hippt::normalize(triangle_load_normal_not_normalized(render_data, reservoir.sample.emissive_triangle_global_index));
    float3 point_on_light = reservoir.sample.point_on_light;
    ColorRGB32F emission = triangle_load_emission(render_data, reservoir.sample.emissive_triangle_global_index);

    return ReGIR_get_reservoir_sample_BSDF_PDF(render_data,
        point_on_light, light_source_normal, emission,
        view_direction, shading_point, shading_normal, geometric_normal, incident_light_info, ray_payload, last_hit_primitive_index);
}

struct ReGIRPairwiseMIS
{
    HIPRT_DEVICE float compute_MIS_weight_normalization(const HIPRTRenderData& render_data, unsigned int valid_non_canonical_neighbors)
    {
        unsigned int number_of_samples = 0;
        number_of_samples += valid_non_canonical_neighbors; // non canonical samples
        if (number_of_samples == 0)
            return 0.0f;

        return 1.0f / number_of_samples;
    }

    HIPRT_DEVICE void sum_non_canonical_sample_to_canonical_weights(const HIPRTRenderData& render_data,
        float3 canonical_technique_1_point_on_light, float3 canonical_technique_1_light_normal, ColorRGB32F canonical_technique_1_emission,
        float3 canonical_technique_2_point_on_light, float3 canonical_technique_2_light_normal, ColorRGB32F canonical_technique_2_emission,
        float3 canonical_technique_3_point_on_light, float3 canonical_technique_3_light_normal, ColorRGB32F canonical_technique_3_emission,

        float canonical_technique_1_canonical_reservoir_1_pdf, float canonical_technique_1_canonical_reservoir_2_pdf, float canonical_technique_1_canonical_reservoir_3_pdf,
        float canonical_technique_2_canonical_reservoir_1_pdf, float canonical_technique_2_canonical_reservoir_2_pdf, float canonical_technique_2_canonical_reservoir_3_pdf,
        float canonical_technique_3_canonical_reservoir_1_pdf, float canonical_technique_3_canonical_reservoir_2_pdf, float canonical_technique_3_canonical_reservoir_3_pdf,
        float mis_weight_normalization,

        ReGIRGridFillSurface neighbor_surface, float neighbor_non_canonical_RIS_integral, bool is_primary_hit,
        Xorshift32Generator& random_number_generator)
    {
        float non_canonical_neighbor_technique_canonical_reservoir_1_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, neighbor_surface, is_primary_hit, neighbor_non_canonical_RIS_integral, canonical_technique_1_point_on_light, canonical_technique_1_light_normal, canonical_technique_1_emission, random_number_generator);
        m_sum_canonical_weight_1 += canonical_technique_1_canonical_reservoir_1_pdf * mis_weight_normalization / (non_canonical_neighbor_technique_canonical_reservoir_1_pdf + canonical_technique_1_canonical_reservoir_1_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_1_pdf * mis_weight_normalization + canonical_technique_3_canonical_reservoir_1_pdf * mis_weight_normalization);

        float non_canonical_neighbor_technique_canonical_reservoir_2_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, neighbor_surface, is_primary_hit, neighbor_non_canonical_RIS_integral, canonical_technique_2_point_on_light, canonical_technique_2_light_normal, canonical_technique_2_emission, random_number_generator);
        m_sum_canonical_weight_2 += canonical_technique_2_canonical_reservoir_2_pdf * mis_weight_normalization / (non_canonical_neighbor_technique_canonical_reservoir_2_pdf + canonical_technique_1_canonical_reservoir_2_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_2_pdf * mis_weight_normalization + canonical_technique_3_canonical_reservoir_2_pdf * mis_weight_normalization);

#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
        float non_canonical_neighbor_technique_canonical_reservoir_3_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, neighbor_surface, is_primary_hit, neighbor_non_canonical_RIS_integral, canonical_technique_3_point_on_light, canonical_technique_3_light_normal, canonical_technique_3_emission, random_number_generator);
        m_sum_canonical_weight_3 += canonical_technique_3_canonical_reservoir_3_pdf * mis_weight_normalization / (non_canonical_neighbor_technique_canonical_reservoir_3_pdf + canonical_technique_1_canonical_reservoir_3_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_3_pdf * mis_weight_normalization + canonical_technique_3_canonical_reservoir_3_pdf * mis_weight_normalization);
#endif
    }

    HIPRT_DEVICE float compute_MIS_weight_for_non_canonical_sample(const HIPRTRenderData& render_data,
        float3 sample_point_on_light, float3 sample_light_source_normal, ColorRGB32F sample_emission, int sample_triangle_index,
        float sample_shading_target_function,

        float3 canonical_technique_1_point_on_light, float3 canonical_technique_1_light_normal, ColorRGB32F canonical_technique_1_emission,
        float3 canonical_technique_2_point_on_light, float3 canonical_technique_2_light_normal, ColorRGB32F canonical_technique_2_emission,
        float3 canonical_technique_3_point_on_light, float3 canonical_technique_3_light_normal, ColorRGB32F canonical_technique_3_emission,

        const ReGIRGridFillSurface& center_grid_cell_surface, bool primary_hit,

        float canonical_technique_1_canonical_reservoir_1_pdf, float canonical_technique_1_canonical_reservoir_2_pdf, float canonical_technique_1_canonical_reservoir_3_pdf,
        float canonical_technique_2_canonical_reservoir_1_pdf, float canonical_technique_2_canonical_reservoir_2_pdf, float canonical_technique_2_canonical_reservoir_3_pdf,
        float canonical_technique_3_canonical_reservoir_1_pdf, float canonical_technique_3_canonical_reservoir_2_pdf, float canonical_technique_3_canonical_reservoir_3_pdf,

        float mis_weight_normalization,

        float non_canonical_RIS_integral_center_grid_cell, float canonical_RIS_integral_center_grid_cell,
        float non_canonical_sample_PDF,

        ReGIRGridFillSurface neighbor_surface, float neighbor_non_canonical_RIS_integral,

        float3 view_direction, float3 shading_point, float3 shading_normal, float3 geometric_normal, RayPayload& ray_payload, int last_hit_primitive_index,

        Xorshift32Generator& random_number_generator)
    {
        float mis_weight = 0.0f;
        // TODO TEST PERF OF THIS
        // if (sample_shading_target_function > 0.0f)
        {
            // PDFs for the canonical techniques
            float non_canonical_PDF = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, center_grid_cell_surface, primary_hit, non_canonical_RIS_integral_center_grid_cell, sample_point_on_light, sample_light_source_normal, sample_emission, random_number_generator);
#if ReGIR_ShadingResamplingCanonicalCandidatesLightTreeATS == KERNEL_OPTION_TRUE
            float canonical_PDF = pdf_of_emissive_triangle_light_tree_ats(render_data, shading_point, shading_normal, sample_triangle_index) / triangle_load_area(render_data, sample_triangle_index);
#else
            float canonical_PDF = ReGIR_get_reservoir_sample_ReGIR_PDF<true>(render_data, center_grid_cell_surface, primary_hit, canonical_RIS_integral_center_grid_cell, sample_point_on_light, sample_light_source_normal, sample_emission, random_number_generator);
#endif
#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
            float bsdf_PDF = ReGIR_get_reservoir_sample_BSDF_PDF(render_data, sample_point_on_light, sample_light_source_normal, sample_emission, view_direction, shading_point, shading_normal, geometric_normal, BSDFIncidentLightInfo::NO_INFO, ray_payload, last_hit_primitive_index);
#else
            float bsdf_PDF = 0.0f;
#endif
            mis_weight = mis_weight_normalization * (non_canonical_sample_PDF / (non_canonical_sample_PDF + non_canonical_PDF * mis_weight_normalization + canonical_PDF * mis_weight_normalization + bsdf_PDF * mis_weight_normalization));
        }

        // Summing the weights for the canonical MIS weight computation
        float non_canonical_neighbor_technique_canonical_reservoir_1_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, neighbor_surface, primary_hit, neighbor_non_canonical_RIS_integral, canonical_technique_1_point_on_light, canonical_technique_1_light_normal, canonical_technique_1_emission, random_number_generator);
        m_sum_canonical_weight_1 += canonical_technique_1_canonical_reservoir_1_pdf * mis_weight_normalization / (non_canonical_neighbor_technique_canonical_reservoir_1_pdf + canonical_technique_1_canonical_reservoir_1_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_1_pdf * mis_weight_normalization + canonical_technique_3_canonical_reservoir_1_pdf * mis_weight_normalization);

        // Add if() here to avoid computing this is the canonical technique 2 doesn't have a sample anyway
        float non_canonical_neighbor_technique_canonical_reservoir_2_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, neighbor_surface, primary_hit, neighbor_non_canonical_RIS_integral, canonical_technique_2_point_on_light, canonical_technique_2_light_normal, canonical_technique_2_emission, random_number_generator);
        m_sum_canonical_weight_2 += canonical_technique_2_canonical_reservoir_2_pdf * mis_weight_normalization / (non_canonical_neighbor_technique_canonical_reservoir_2_pdf + canonical_technique_1_canonical_reservoir_2_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_2_pdf * mis_weight_normalization + canonical_technique_3_canonical_reservoir_2_pdf * mis_weight_normalization);

#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
        // Add if() here to avoid computing this is the canonical technique 3 doesn't have a sample anyway
        float non_canonical_neighbor_technique_canonical_reservoir_3_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, neighbor_surface, primary_hit, neighbor_non_canonical_RIS_integral, canonical_technique_3_point_on_light, canonical_technique_3_light_normal, canonical_technique_3_emission, random_number_generator);
        m_sum_canonical_weight_3 += canonical_technique_3_canonical_reservoir_3_pdf * mis_weight_normalization / (non_canonical_neighbor_technique_canonical_reservoir_3_pdf + canonical_technique_1_canonical_reservoir_3_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_3_pdf * mis_weight_normalization + canonical_technique_3_canonical_reservoir_3_pdf * mis_weight_normalization);
#endif

        return mis_weight;
    }

    HIPRT_DEVICE float get_canonical_MIS_weight_1(float canonical_technique_1_canonical_reservoir_1_pdf, float canonical_technique_2_canonical_reservoir_1_pdf, float canonical_technique_3_canonical_reservoir_1_pdf, float mis_weight_normalization)
    {
        if (mis_weight_normalization == 0.0f)
            // We only have the canonical techniques available, we're going to go for a balance heuristic between them
            return canonical_technique_1_canonical_reservoir_1_pdf / (canonical_technique_1_canonical_reservoir_1_pdf + canonical_technique_2_canonical_reservoir_1_pdf + canonical_technique_3_canonical_reservoir_1_pdf);

        return m_sum_canonical_weight_1 * mis_weight_normalization;
    }

    HIPRT_DEVICE float get_canonical_MIS_weight_2(float canonical_technique_1_canonical_reservoir_2_pdf, float canonical_technique_2_canonical_reservoir_2_pdf, float canonical_technique_3_canonical_reservoir_2_pdf, float mis_weight_normalization)
    {
        if (mis_weight_normalization == 0.0f)
            // We only have the canonical techniques available, we're going to go for a balance heuristic between them
            return canonical_technique_2_canonical_reservoir_2_pdf / (canonical_technique_1_canonical_reservoir_2_pdf + canonical_technique_2_canonical_reservoir_2_pdf + canonical_technique_3_canonical_reservoir_2_pdf);

        return m_sum_canonical_weight_2 * mis_weight_normalization;
    }

    HIPRT_DEVICE float get_canonical_MIS_weight_3(float canonical_technique_1_canonical_reservoir_3_pdf, float canonical_technique_2_canonical_reservoir_3_pdf, float canonical_technique_3_canonical_reservoir_3_pdf, float mis_weight_normalization)
    {
        if (mis_weight_normalization == 0.0f)
            // We only have the canonical techniques available, we're going to go for a balance heuristic between them
            return canonical_technique_3_canonical_reservoir_3_pdf / (canonical_technique_1_canonical_reservoir_3_pdf + canonical_technique_2_canonical_reservoir_3_pdf + canonical_technique_3_canonical_reservoir_3_pdf);

        return m_sum_canonical_weight_3 * mis_weight_normalization;
    }

    // 1st is non-canonical samples
    float m_sum_canonical_weight_1 = 0.0f;
    // 2nd technique is canonical samples
    float m_sum_canonical_weight_2 = 0.0f;
    // 3rd technique is BSDF samples
    float m_sum_canonical_weight_3 = 0.0f;
};

#endif
