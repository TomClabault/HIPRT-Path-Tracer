#ifndef APPLICATION_SETTINGS_H
#define APPLICATION_SETTINGS_H

#include <string>
#include <vector>

#include "Renderer/denoiser_debug_view.h"

struct ApplicationSettings
{
	int selected_kernel = 0;
	std::vector<std::string> kernel_files = { "Kernels/path_tracer_kernel.h", "Kernels/normals_kernel.h" };
	std::vector<std::string> kernel_functions = { "PathTracerKernel", "NormalsKernel"};

	DenoiserDebugView debug_display_denoiser = DenoiserDebugView::NONE;
	// How many samples were denoised by the last denoiser call
	int last_denoised_sample_count;
	bool denoise_at_target_sample_count = false;

	// How much to divide the translation distance by when the mouse
	// has been dragged over the window to move the camera
	// This is necessary because if 1 pixel of movement equalled
	// 1 world unit of translation, it would be way too fast!
	double view_translation_sldwn_x = 300.0f, view_translation_sldwn_y = 300.0f;
	double view_rotation_sldwn_x = 3.5f, view_rotation_sldwn_y = 3.5f;
	double view_zoom_sldwn = 5.0f;

	int stop_render_at = 0;

	float tone_mapping_gamma = 2.2f;
	float tone_mapping_exposure = 1.0f;
};

#endif