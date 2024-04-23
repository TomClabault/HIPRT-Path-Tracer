#include "Image/image_writer.h"

#include "GL/glew.h"
#include "stb_image_write.h"
#include "UI/render_window.h"
#include "Utils/utils.h"
#include "Utils/opengl_utils.h"

void ImageWriter::set_renderer(Renderer* renderer)
{
	m_renderer = renderer;
}

void ImageWriter::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;
}

void ImageWriter::write_to_png()
{
	std::stringstream filename;
	std::time_t t = std::time(0);
	std::tm* now = std::localtime(&t);

	filename << std::put_time(now, "%m.%d.%Y.%H.%M.%S - ") << m_renderer->get_render_settings().sample_number << "sp @ " << m_renderer->m_render_width << "x" << m_renderer->m_render_height << ".png";

	write_to_png(filename.str().c_str());
}

void ImageWriter::write_to_png(const char* filepath)
{
	return;

	int width = m_renderer->m_render_width;
	int height = m_renderer->m_render_height;

	if (m_render_window->get_application_settings().render_resolution_scale == 1.0f)
	{
		// Fast path when no resolution scaling
		std::vector<unsigned char> mapped_data(width * height * 3);
		glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, mapped_data.data());

		stbi_flip_vertically_on_write(true);
		if (stbi_write_png(filepath, width, height, 3, mapped_data.data(), width * sizeof(unsigned char) * 3))
			std::cout << "Render written to \"" << filepath << "\"" << std::endl;
	}
	else
	{
		// We're going to have to go through a compute shader for a resolution scaling
		// different than 1.0f because OpenGL's viewport doesn't match the render size
		// (so we can't just dump the viewport's pixels to a file as in the 1.0f case)

		if (!m_compute_shader_initialized)
		{
			OpenGLShader compute_shader(GLSL_SHADERS_DIRECTORY "/display.frag", OpenGLShader::COMPUTE_SHADER);
			m_compute_shader.attach(compute_shader);
			m_compute_shader.link();
		}

		bool texture_needs_creation = false;
		if (m_compute_output_image_width == -1)
		{
			texture_needs_creation = true;
		}
		else if (m_compute_output_image_width != width || m_compute_output_image_height != height)
		{
			glDeleteTextures(1, &m_output_image);
			texture_needs_creation = true;
		}

		if (texture_needs_creation)
		{
			m_compute_output_image_width = width;
			m_compute_output_image_height = height;

			glGenTextures(1, &m_output_image);
			glActiveTexture(GL_TEXTURE0 + RenderWindow::DISPLAY_COMPUTE_IMAGE_UNIT);
			glBindTexture(GL_TEXTURE_2D, m_output_image);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
			glBindImageTexture(2, m_output_image, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
		}
		else
			glBindTexture(GL_TEXTURE_2D, m_output_image);

		GLint threads[3];
		m_compute_shader.get_compute_threads(threads);

		int nb_groups_x = std::ceil(width / (float)threads[0]);
		int nb_groups_y = std::ceil(height / (float)threads[1]);

		m_compute_shader.use();
		m_compute_shader.set_uniform("u_texture", RenderWindow::DISPLAY_TEXTURE_UNIT);
		m_compute_shader.set_uniform("u_sample_number", m_renderer->get_sample_number());
		m_compute_shader.set_uniform("u_exposure", m_render_window->get_application_settings().tone_mapping_exposure);
		m_compute_shader.set_uniform("u_gamma", m_render_window->get_application_settings().tone_mapping_gamma);
		m_compute_shader.set_uniform("u_output_image", RenderWindow::DISPLAY_COMPUTE_IMAGE_UNIT);
		//m_render_window->setup_display_uniforms(m_compute_shader);

		glDispatchCompute(nb_groups_x, nb_groups_y, 1);
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		std::vector<unsigned char> mapped_data(width * height * 4);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, mapped_data.data());

		stbi_flip_vertically_on_write(true);
		if (stbi_write_png(filepath, width, height, 4, mapped_data.data(), width * sizeof(unsigned char) * 4))
			std::cout << "Render written to \"" << filepath << "\"" << std::endl;
	}
}

