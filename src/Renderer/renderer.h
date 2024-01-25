#ifndef RENDERER_H
#define RENDERER_H

#include "Image/color.h"
#include "Renderer/orochi_buffer.h"

#include <vector>
#include <Scene/scene_parser.h>

class Renderer
{
public:
	Renderer(int width, int height) : m_framebuffer_width(width), m_framebuffer_height(height), m_framebuffer(width * height) {}
	Renderer() {}

	void render();
	void resize(int new_width, int new_height);

	OrochiBuffer<float>& get_orochi_framebuffer();

	void set_scene(const Scene& scene);





	int m_framebuffer_width, m_framebuffer_height;

	int samples_per_pixel;
	int bounces;

private:
	OrochiBuffer<float> m_framebuffer;
};

#endif

