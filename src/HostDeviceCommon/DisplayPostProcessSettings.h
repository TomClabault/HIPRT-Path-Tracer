/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DISPLAY_POST_PROCESS_SETTINGS_H
#define DISPLAY_POST_PROCESS_SETTINGS_H

enum DisplayPostProcessView
{
	DISPLAY_POST_PROCESS_DEFAULT = 0,
	DISPLAY_POST_PROCESS_DENOISER_ALBEDO,
	DISPLAY_POST_PROCESS_DENOISER_NORMALS,
	DISPLAY_POST_PROCESS_WHITE_FURNACE_THRESHOLD
};

enum DisplayAdaptiveSamplingView
{
	DISPLAY_ADAPTIVE_SAMPLING_NONE = 0,
	DISPLAY_ADAPTIVE_SAMPLING_PIXEL_CONVERGENCE_HEATMAP,
	DISPLAY_ADAPTIVE_SAMPLING_PIXEL_CONVERGED_MAP,
	DISPLAY_ADAPTIVE_SAMPLING_HIERARCHICAL_REGION_STATE_MAP,
	DISPLAY_ADAPTIVE_SAMPLING_HIERARCHICAL_PIXEL_NOISE
};

struct DisplayPostProcessSettings
{
	int do_tonemapping = 1;
	float gamma		   = 2.2f;
	float exposure	   = 1.8f;

	DisplayPostProcessView display_view	 = DISPLAY_POST_PROCESS_DEFAULT;
	int white_furnace_use_low_threshold	 = 0;
	int white_furnace_use_high_threshold = 1;
	int white_furnace_sample_count		 = 1;

	// Adaptive sampling debug views are evaluated by the final display post-process pass at runtime.
	int adaptive_sampling_display_view	= DISPLAY_ADAPTIVE_SAMPLING_NONE;
	int adaptive_sampling_heatmap_index = 0;
};

#endif // #ifndef DISPLAY_POST_PROCESS_SETTINGS_H
