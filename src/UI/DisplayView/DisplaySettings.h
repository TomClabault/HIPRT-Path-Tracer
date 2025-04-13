/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DISPLAY_SETTINGS_H
#define DISPLAY_SETTINGS_H

struct DisplaySettings
{
	// If 1.0f, 100% of the denoised result is displayed in the viewport.
	// If 0.0f, 100% of the noisy framebuffer is displayed in the viewport
	// Linearly interpoalted between the two for intermediate values
	float denoiser_blend = 1.0f;
	// Overrides the blending factor for the blend-2-textures display shader
	// 0.0f displays 100% of texture 1.
	// 1.0f gives 100% of texture 2.
	// -1.0f disables the override
	float blend_override = -1.0f;

	// Whether or not to do tonemapping for display fragment shader that support it
	bool do_tonemapping = true;
	// Tone mapping gamma
	float tone_mapping_gamma = 2.2f;
	// Tone mapping exposure
	float tone_mapping_exposure = 1.8f;

	// If true, the white furnace threshold shader will display
	// pixel that lose energy as green. Pixels will not be highlighted
	// if false
	bool white_furnace_display_use_low_threshold = false;
	// If true, the white furnace threshold shader will display
	// pixel that gain energy as red. Pixels will not be highlighted
	// if false
	bool white_furnace_display_use_high_threshold = true;
};

#endif
