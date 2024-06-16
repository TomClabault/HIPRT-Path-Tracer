/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef SCREENSHOTER_H
#define SCREENSHOTER_H

#include "GL/glew.h"
#include "OpenGL/OpenGLProgram.h"
#include "Renderer/GPURenderer.h"

class RenderWindow;

class Screenshoter
{
public:
	void set_renderer(std::shared_ptr<GPURenderer> renderer);
	void set_render_window(RenderWindow* render_window);

	void select_compute_program(DisplayView display_view);
	void initialize_programs();
	void resize_output_image(int width, int height);

	/**
	 * A filename with a time stamp, the render resolution and the
	 * number of samples is automatically generated:
	 * 
	 * 03.17.2024 1024sp @ 1280x720.png
	 * 
	 * for example
	 */
	void write_to_png();
	void write_to_png(const char* filepath);

private:
	std::shared_ptr<GPURenderer> m_renderer = nullptr;
	RenderWindow* m_render_window = nullptr;

	bool m_compute_shader_initialized = false;
	OpenGLProgram m_active_compute_program;
	OpenGLProgram m_default_compute_program;
	OpenGLProgram m_blend_2_compute_program;
	OpenGLProgram m_normal_compute_program;
	OpenGLProgram m_albedo_compute_program;
	OpenGLProgram m_adaptive_sampling_compute_program;

	GLuint m_output_image = 0;
	int m_compute_output_image_width = -1;
	int m_compute_output_image_height = -1;
};

#endif