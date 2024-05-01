#ifndef IMAGE_WRITER_H
#define IMAGE_WRITER_H

#include "GL/glew.h"
#include "OpenGL/OpenGLProgram.h"
#include "Renderer/renderer.h"

class RenderWindow;

class Screenshoter
{
public:
	void set_renderer(Renderer* renderer);
	void set_render_window(RenderWindow* render_window);

	void select_compute_program(DisplayView display_view);
	void initialize_programs();
	void prepare_output_image(int width, int height);

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
	Renderer* m_renderer;
	RenderWindow* m_render_window;

	bool m_compute_shader_initialized = false;
	OpenGLProgram m_active_compute_program;
	OpenGLProgram m_default_compute_program;
	OpenGLProgram m_normal_compute_program;
	OpenGLProgram m_albedo_compute_program;
	OpenGLProgram m_adaptative_sampling_compute_program;

	GLuint m_output_image;
	int m_compute_output_image_width = -1;
	int m_compute_output_image_height = -1;
};

#endif