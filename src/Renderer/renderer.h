#ifndef RENDERER_H
#define RENDERER_H

#include "Image/color.h"

#include <vector>

class Renderer
{
public:
	Renderer(int width, int height) : m_framebuffer_width(width), m_framebuffer_height(height), m_framebuffer_data(width * height) {}
	Renderer() : Renderer(1280, 720) {}

	void render();
	void resize(int new_width, int new_height);

	std::vector<Color>& get_cpu_data();





	int m_framebuffer_width, m_framebuffer_height;

	int samples_per_pixel;
	int bounces;

private:
	std::vector<Color> m_framebuffer_data;

};

#endif

