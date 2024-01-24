#include "renderer.h"

void Renderer::render()
{
	static int debug_counter = 0;

	float counter_float = debug_counter / 59.0f;

	for (int y = 0; y < m_framebuffer_height; y++)
		for (int x = 0; x < m_framebuffer_width; x++)
			m_framebuffer_data[y * m_framebuffer_width + x] = Color(counter_float, 1.0f - counter_float, 0.0f);

	debug_counter = (debug_counter + 1) % 60;
}

void Renderer::resize(int new_width, int new_height)
{
	m_framebuffer_width = new_width;
	m_framebuffer_height = new_height;

	m_framebuffer_data.resize(new_width * new_height);
}

std::vector<Color>& Renderer::get_cpu_data()
{
	return m_framebuffer_data;
}
