/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ANIMATION_STATE_H
#define RENDERER_ANIMATION_STATE_H

#include <filesystem>

struct RendererAnimationState
{
	// If true, objects will be animated in the scene at each frame
	bool do_animations = false;
	// If true, then this means that the renderer is currently rendering a frame
	// sequence (typically an animation). This means, for example, that animations
	// will only step after the current frame has converged, etc...
	bool is_rendering_frame_sequence = false;
	// This boolean is read by the various components of the scene that can
	// be animated.
	// 
	// If true, this boolean is true, the components are allowed to step their animation.
	bool can_step_animation = false;

	// How many frames have been rendered so far
	int frames_rendered_so_far = 0;
	// How many frames to render for the frame sequence
	int number_of_animation_frames = 100;

	std::string frames_output_folder = "FrameSequence";

	std::string get_frame_filepath()
	{
		return frames_output_folder + "/" + std::to_string(frames_rendered_so_far) + ".png";
	}

	void ensure_output_folder_exists()
	{
		if (!std::filesystem::exists(frames_output_folder)) 
			// Creates the folder and any necessary parent directories if it doesn't exist yet
        	std::filesystem::create_directories(frames_output_folder); 
	}

	void reset()
	{
		frames_rendered_so_far = 0;

		std::stringstream ss;

		ss << "FrameSeq - ";
		Utils::get_current_date_string(ss);

		frames_output_folder = ss.str();
	}
};

#endif
