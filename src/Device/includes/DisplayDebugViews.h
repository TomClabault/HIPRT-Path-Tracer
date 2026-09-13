/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_DISPLAY_DEBUG_VIEWS_H
#define DEVICE_INCLUDES_DISPLAY_DEBUG_VIEWS_H

#include "Device/includes/Heatmap.h"
#include "Device/includes/Tonemapping.h"
#include "HostDeviceCommon/DisplayPostProcessSettings.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE ColorRGB32F display_view_denoised_blend(const HIPRTRenderData& render_data, unsigned int pixel_index)
{
	ColorRGB32F noisy_color = render_data.buffers.accumulated_ray_colors[pixel_index];
	if (render_data.buffers.gmon_estimator.result_framebuffer != nullptr)
		noisy_color = render_data.buffers.gmon_estimator.result_framebuffer[pixel_index];

	ColorRGB32F denoised_color = noisy_color;
	if (render_data.buffers.denoised_ray_colors != nullptr)
		denoised_color = render_data.buffers.denoised_ray_colors[pixel_index];

	unsigned int noisy_sample_count	   = static_cast<unsigned int>(hippt::max(1, render_data.display_post_process_settings.denoised_blend_noisy_sample_count));
	unsigned int denoised_sample_count = static_cast<unsigned int>(hippt::max(1, render_data.display_post_process_settings.denoised_blend_sample_count));
	noisy_color						   = noisy_color / static_cast<float>(noisy_sample_count);
	denoised_color					   = denoised_color / static_cast<float>(denoised_sample_count);

	if (render_data.display_post_process_settings.do_tonemapping == 1)
	{
		float exposure = render_data.display_post_process_settings.exposure;
		float gamma	   = render_data.display_post_process_settings.gamma;
		noisy_color	   = tonemap_exponential(noisy_color, exposure, gamma);
		denoised_color = tonemap_exponential(denoised_color, exposure, gamma);
	}

	float blend_factor = render_data.display_post_process_settings.denoised_blend_factor;
	return noisy_color * (1.0f - blend_factor) + denoised_color * blend_factor;
}

HIPRT_DEVICE ColorRGB32F display_view_denoiser_albedo(const HIPRTRenderData& render_data, unsigned int pixel_index)
{
	if (render_data.aux_buffers.denoiser_albedo == nullptr)
		return ColorRGB32F(0.0f);

	return render_data.aux_buffers.denoiser_albedo[pixel_index];
}

HIPRT_DEVICE ColorRGB32F display_view_denoiser_normals(const HIPRTRenderData& render_data, unsigned int pixel_index)
{
	if (render_data.aux_buffers.denoiser_normals == nullptr)
		return ColorRGB32F(0.0f);

	float3_t normal = render_data.aux_buffers.denoiser_normals[pixel_index];
	return ColorRGB32F((normal.x + 1.0f) * 0.5f, (normal.y + 1.0f) * 0.5f, (normal.z + 1.0f) * 0.5f);
}

HIPRT_DEVICE ColorRGB32F display_view_apply_white_furnace_threshold(ColorRGB32F color, const DisplayPostProcessSettings& display_settings)
{
	if (display_settings.white_furnace_use_high_threshold == 1 && (color.r > 0.505f || color.g > 0.505f || color.b > 0.505f))
		return ColorRGB32F(1.0f, 0.0f, 0.0f);

	if (display_settings.white_furnace_use_low_threshold == 1 && (color.r < 0.495f || color.g < 0.495f || color.b < 0.495f))
		return ColorRGB32F(0.0f, 1.0f, 0.0f);

	return color;
}

HIPRT_DEVICE bool display_view_compute_adaptive_sampling_debug_value(const HIPRTRenderData& render_data, int pixel_index, float& out_debug_value)
{
	if (!render_data.render_settings.has_access_to_adaptive_sampling_buffers() || render_data.aux_buffers.pixel_converged_sample_count == nullptr)
		return false;

	int minimum_sample_count = render_data.render_settings.adaptive_sampling_min_samples;
	int maximum_sample_count = static_cast<int>(render_data.render_settings.sample_number + 1);
	if (maximum_sample_count < minimum_sample_count)
		maximum_sample_count = minimum_sample_count;

	int pixel_converged_sample_count = render_data.aux_buffers.pixel_converged_sample_count[pixel_index];
	if (pixel_converged_sample_count == -1)
		pixel_converged_sample_count = maximum_sample_count;

	if (maximum_sample_count == minimum_sample_count)
	{
		out_debug_value = 1.0f;
		return true;
	}

	float clamped_sample_count =
		hippt::clamp(static_cast<float>(minimum_sample_count), static_cast<float>(maximum_sample_count), static_cast<float>(pixel_converged_sample_count));
	out_debug_value = (clamped_sample_count - static_cast<float>(minimum_sample_count)) /
					  (static_cast<float>(maximum_sample_count) - static_cast<float>(minimum_sample_count));

	return true;
}

HIPRT_DEVICE ColorRGB32F display_view_map_adaptive_sampling_heatmap(float scalar_0_1, int heatmap_index)
{
	switch (heatmap_index)
	{
	case HEATMAP_INDEX_MAGMA:
		return map_0_1_to_heatmap_color_by_index<HEATMAP_INDEX_MAGMA>(scalar_0_1);
	case HEATMAP_INDEX_INFERNO:
		return map_0_1_to_heatmap_color_by_index<HEATMAP_INDEX_INFERNO>(scalar_0_1);
	case HEATMAP_INDEX_VIRIDIS:
		return map_0_1_to_heatmap_color_by_index<HEATMAP_INDEX_VIRIDIS>(scalar_0_1);
	case HEATMAP_INDEX_GRAYSCALE:
		return map_0_1_to_heatmap_color_by_index<HEATMAP_INDEX_GRAYSCALE>(scalar_0_1);
	case HEATMAP_INDEX_BLUE_GREEN_RED:
	default:
		return map_0_1_to_heatmap_color_by_index<HEATMAP_INDEX_BLUE_GREEN_RED>(scalar_0_1);
	}
}

HIPRT_DEVICE bool display_view_compute_adaptive_sampling_debug_color(
	const HIPRTRenderData& render_data, int pixel_index, int adaptive_sampling_display_view, int heatmap_index, ColorRGB32F& out_debug_color)
{
	switch (adaptive_sampling_display_view)
	{
	case DISPLAY_ADAPTIVE_SAMPLING_PIXEL_CONVERGENCE_HEATMAP:
	{
		if (!render_data.render_settings.enable_adaptive_sampling)
			return false;

		float adaptive_sampling_debug_value;
		if (!display_view_compute_adaptive_sampling_debug_value(render_data, pixel_index, adaptive_sampling_debug_value))
			return false;

		out_debug_color = display_view_map_adaptive_sampling_heatmap(adaptive_sampling_debug_value, heatmap_index);
		return true;
	}

	case DISPLAY_ADAPTIVE_SAMPLING_PIXEL_CONVERGED_MAP:
	{
		if (!render_data.render_settings.enable_adaptive_sampling || !render_data.render_settings.has_access_to_adaptive_sampling_buffers() ||
			render_data.aux_buffers.pixel_converged_sample_count == nullptr)
			return false;

		// The buffer is initialized to -1 and is assigned a sample count exactly when the pixel converges.
		out_debug_color = render_data.aux_buffers.pixel_converged_sample_count[pixel_index] == -1 ? ColorRGB32F(0.0f) : ColorRGB32F(1.0f);
		return true;
	}

	case DISPLAY_ADAPTIVE_SAMPLING_HIERARCHICAL_REGION_STATE_MAP:
	{
		if (!render_data.render_settings.use_hierarchical_adaptive_sampling() || !render_data.render_settings.has_access_to_adaptive_sampling_buffers() ||
			render_data.aux_buffers.pixel_active == nullptr)
			return false;

		out_debug_color = render_data.aux_buffers.pixel_active[pixel_index] != 0 ? ColorRGB32F(1.0f, 0.0f, 0.0f) : ColorRGB32F(0.0f, 1.0f, 0.0f);
		return true;
	}

	case DISPLAY_ADAPTIVE_SAMPLING_HIERARCHICAL_PIXEL_NOISE:
	{
		if (!render_data.render_settings.use_hierarchical_adaptive_sampling() || !render_data.render_settings.has_access_to_adaptive_sampling_buffers() ||
			render_data.aux_buffers.hierarchical_adaptive_sampling_error == nullptr)
			return false;

		float relative_noise   = render_data.aux_buffers.hierarchical_adaptive_sampling_error[pixel_index];
		float noise_threshold  = render_data.render_settings.hierarchical_adaptive_sampling_target_error;
		float normalized_noise = noise_threshold > 0.0f ? relative_noise / noise_threshold : (relative_noise > 0.0f ? 1.0f : 0.0f);
		out_debug_color		   = display_view_map_adaptive_sampling_heatmap(hippt::clamp(0.0f, 1.0f, normalized_noise), heatmap_index);
		return true;
	}

	case DISPLAY_ADAPTIVE_SAMPLING_NONE:
	default:
		return false;
	}
}

#endif // #ifndef DEVICE_INCLUDES_DISPLAY_DEBUG_VIEWS_H
