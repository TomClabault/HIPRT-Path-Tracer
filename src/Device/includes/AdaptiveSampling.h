/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_ADAPTIVE_SAMPLING_H
#define DEVICE_ADAPTIVE_SAMPLING_H

#include "HostDeviceCommon/RenderData.h"
#include "Device/includes/Tonemapping.h"

HIPRT_DEVICE static float compute_adaptive_sampling_display_luminance(float linear_luminance, const DisplayPostProcessSettings& display_settings)
{
	float clamped_luminance = hippt::clamp(0.0f, 1.0e35f, linear_luminance);
	if (display_settings.do_tonemapping == 1)
		return tonemap_exponential(ColorRGB32F(clamped_luminance), display_settings.exposure, display_settings.gamma).luminance();

	return hippt::clamp(0.0f, 1.0f, clamped_luminance);
}

HIPRT_DEVICE static float compute_adaptive_sampling_display_error(float average_luminance,
																  float confidence_interval,
																  const DisplayPostProcessSettings& display_settings)
{
	float lower_luminance = hippt::max(0.0f, average_luminance - confidence_interval);
	float upper_luminance = average_luminance + confidence_interval;

	float displayed_average_luminance = compute_adaptive_sampling_display_luminance(average_luminance, display_settings);
	float displayed_lower_luminance	  = compute_adaptive_sampling_display_luminance(lower_luminance, display_settings);
	float displayed_upper_luminance	  = compute_adaptive_sampling_display_luminance(upper_luminance, display_settings);

	float lower_display_error = displayed_average_luminance - displayed_lower_luminance;
	float upper_display_error = displayed_upper_luminance - displayed_average_luminance;
	return hippt::max(lower_display_error, upper_display_error);
}

HIPRT_DEVICE static float get_pixel_confidence_interval(const HIPRTRenderData& render_data, int pixel_index, int pixel_sample_count, float& average_luminance)
{
	// The accumulated framebuffer can contain a debug view, so adaptive sampling keeps its luminance sum in a separate buffer.
	float luminance	  = render_data.aux_buffers.pixel_luminance[pixel_index];
	average_luminance = luminance / (pixel_sample_count + 1);

	float squared_luminance = render_data.aux_buffers.pixel_squared_luminance[pixel_index];
	float pixel_variance	= (squared_luminance - luminance * average_luminance) / (pixel_sample_count + 1);

	return 1.96f * hippt::sqrt(pixel_variance) / hippt::sqrt(pixel_sample_count + 1);
}

/**
 * pixel_converged is set to true if the given pixel has reached the noise
 * threshold given in render_data.render_settings.stop_pixel_percentage_converged. It
 * is set to false otherwise.
 *
 * Returns true if the pixel needs more sample according to adaptive sampling (or if adaptive sampling is disabled).
 * Returns false otherwise
 */
HIPRT_DEVICE static bool adaptive_sampling(const HIPRTRenderData& render_data, int pixel_index, bool& pixel_converged)
{
	const HIPRTRenderSettings& render_settings = render_data.render_settings;
	const AuxiliaryBuffers& aux_buffers		   = render_data.aux_buffers;

	if (!render_settings.has_access_to_adaptive_sampling_buffers())
		// Adaptive sampling is not on so returning true to indicate
		// that this pixel is going to need sampling
		return true;

	if (render_settings.enable_adaptive_sampling)
	{
		// Computing pixel convergence according to adaptive sampling to
		// know whether to keep sampling that pixel or not

		if (aux_buffers.pixel_converged_sample_count[pixel_index] != -1)
			// Pixel is already converged because we have a value != -1 in the
			// pixels converged sample count buffer
			return false;

		int pixel_sample_count = aux_buffers.pixel_sample_count[pixel_index];
		if (pixel_sample_count > render_settings.adaptive_sampling_min_samples)
		{
			float average_luminance;
			float confidence_interval = get_pixel_confidence_interval(render_data, pixel_index, pixel_sample_count, average_luminance);
			float display_error = compute_adaptive_sampling_display_error(average_luminance, confidence_interval, render_data.display_post_process_settings);

			bool pixel_needs_sampling = display_error > render_settings.adaptive_sampling_noise_threshold;
			if (!pixel_needs_sampling)
			{
				if (aux_buffers.pixel_converged_sample_count[pixel_index] == -1)
					// Indicates no need to sample anymore by indicating that this pixel has converged
					// only if we hadn't indicated the convergence before already
					aux_buffers.pixel_converged_sample_count[pixel_index] = pixel_sample_count;

				return false;
			}
		}

		return true;
	}
	// Only counting the convergence of pixels according to
	// the pixel stop noise threshold if adaptive sampling is not enabled
	//
	// The rationale is that if we have both adaptive sampling and pixel stop noise threshold
	// enabled, we probably want to use adaptive sampling only but also stop rendering after
	// a certain proportion of pixels have converged and we don't actually want to use the
	// "stop pixel noise threshold" but only the "stop pixel convergence proportion"
	else if (render_settings.stop_pixel_noise_threshold > 0.0f && render_settings.use_pixel_stop_noise_threshold)
	{
		int pixel_sample_count = aux_buffers.pixel_sample_count[pixel_index];

		float average_luminance;
		float confidence_interval = get_pixel_confidence_interval(render_data, pixel_index, pixel_sample_count, average_luminance);

		// The value of pixel_converged will be used outside of this function
		pixel_converged =
			// Converged enough
			(confidence_interval <= render_settings.stop_pixel_noise_threshold * average_luminance)
			// At least 2 samples because we can't evaluate the variance with only 1 sample
			&& (render_settings.sample_number > 1);

		int current_converged_count = aux_buffers.pixel_converged_sample_count[pixel_index];
		if (pixel_converged && current_converged_count == -1)
			// If the pixel has converged, storing the number of samples at which it has converged.
			// We're only storing the number of samples if we hadn't already (if the value in the buffer is -1)
			aux_buffers.pixel_converged_sample_count[pixel_index] = pixel_sample_count;
		else if (!pixel_converged)
			// If the pixel hasn't converged
			aux_buffers.pixel_converged_sample_count[pixel_index] = -1;
	}

	return true;
}

#endif // #ifndef DEVICE_ADAPTIVE_SAMPLING_H
