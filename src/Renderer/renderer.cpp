#include "renderer.h"

void Renderer::render()
{
	static int debug_counter = 0;

	float counter_float = debug_counter / 59.0f;

	std::vector<float> framebuffer_float(m_framebuffer_height * m_framebuffer_width * 4);
	for (int y = 0; y < m_framebuffer_height; y++)
		for (int x = 0; x < m_framebuffer_width; x++)
			*((Color*)&framebuffer_float[(y * m_framebuffer_width + x) * 4]) = Color(counter_float, 1.0f - counter_float, 0.0f);

	m_framebuffer.upload_pixels(framebuffer_float);

	debug_counter++;
	debug_counter %= 60;
}

void Renderer::resize(int new_width, int new_height)
{
	m_framebuffer_width = new_width;
	m_framebuffer_height = new_height;

	// * 4 for RGBA
	m_framebuffer.resize(new_width * new_height * 4);
}

OrochiBuffer<float>& Renderer::get_orochi_framebuffer()
{
	return m_framebuffer;
}

void Renderer::set_scene(const Scene& scene)
{

}
