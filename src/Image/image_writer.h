#ifndef IMAGE_WRITER_H
#define IMAGE_WRITER_H

#include "GL/glew.h"
#include "Renderer/renderer.h"
//#include "UI/app_window.h"

class AppWindow;

class ImageWriter
{
public:
	ImageWriter() : m_renderer(nullptr), m_render_window(nullptr) {};

	void init_shader();

	void set_renderer(Renderer* renderer);
	void set_render_window(AppWindow* render_window);

	void write_to_png(const char* filepath);

private:
	Renderer* m_renderer;
	AppWindow* m_render_window;

	GLuint m_compute_shader;
	GLuint m_compute_output_image;
	int m_compute_output_image_width = -1;
	int m_compute_output_image_height = -1;
};

#endif