#ifndef IMAGE_WRITER_H
#define IMAGE_WRITER_H

#include "GL/glew.h"
#include "Renderer/renderer.h"

class RenderWindow;

class ImageWriter
{
public:
	ImageWriter() : m_renderer(nullptr), m_render_window(nullptr) {};

	void set_renderer(Renderer* renderer);
	void set_render_window(RenderWindow* render_window);

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

	GLuint m_compute_shader = -1;
	GLuint m_compute_output_image = -1;
	int m_compute_output_image_width = -1;
	int m_compute_output_image_height = -1;
};

#endif