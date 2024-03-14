#include "Image/image_writer.h"

#include "GL/glew.h"
#include "stb_image_write.h"
#include "UI/app_window.h"
#include "Utils/utils.h"

ImageWriter::ImageWriter() : m_renderer(nullptr)
{

}

void ImageWriter::set_renderer(Renderer* renderer)
{
	m_renderer = renderer;
}

void ImageWriter::set_render_window(AppWindow* render_window)
{
	m_render_window = render_window;
}

void ImageWriter::write_to_png(const char* filepath)
{
	int width = m_renderer->m_render_width;
	int height = m_renderer->m_render_height;

	if (m_render_window->get_application_settings().render_resolution_scale == 1.0f)
	{
		// Fast path when no resolution scaling
		std::vector<unsigned char> tonemaped_data(width * height * 3);

		glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, tonemaped_data.data());

		stbi_flip_vertically_on_write(true);
		if (stbi_write_png(filepath, width, height, 3, tonemaped_data.data(), width * sizeof(unsigned char) * 3))
			std::cout << "Render written to \"" << filepath << "\"" << std::endl;
	}
	else
	{
		// We're going to have to go through a compute shader for a resolution scaling
		// different than 1.0f because OpenGL's viewport doesn't match the render size
		// (so we can't just dump the viewport's pixels to a file as in the 1.0f case)

		// TODO use a compute shader with the same shader code as the fragment shader of the display
		// shader


	}
}