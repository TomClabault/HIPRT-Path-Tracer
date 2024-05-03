/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "GL/glew.h"
#include "stb_image_write.h"
#include "UI/RenderWindow.h"
#include "UI/Screenshoter.h"
#include "Utils/Utils.h"

void Screenshoter::set_renderer(GPURenderer* renderer)
{
	m_renderer = renderer;
}

void Screenshoter::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;
}

void Screenshoter::select_compute_program(DisplayView display_view)
{
	switch (display_view)
	{
	case DisplayView::DEFAULT:
		m_active_compute_program = m_default_compute_program;
		break;

	case DisplayView::DISPLAY_NORMALS:
	case DisplayView::DISPLAY_DENOISED_NORMALS:
		m_active_compute_program = m_normal_compute_program;
		break;

	case DisplayView::DISPLAY_ALBEDO:
	case DisplayView::DISPLAY_DENOISED_ALBEDO:
		m_active_compute_program = m_albedo_compute_program;
		break;

	case DisplayView::ADAPTIVE_SAMPLING_MAP:
		m_active_compute_program = m_adaptative_sampling_compute_program;
		break;


	default:
		break;
	}
}

void Screenshoter::write_to_png()
{
	std::stringstream filename;
	std::time_t t = std::time(0);
	std::tm* now = std::localtime(&t);

	filename << std::put_time(now, "%m.%d.%Y.%H.%M.%S - ") << m_renderer->get_render_settings().sample_number << "sp @ " << m_renderer->m_render_width << "x" << m_renderer->m_render_height << ".png";

	write_to_png(filename.str().c_str());
}

void Screenshoter::initialize_programs()
{
	std::vector<std::string> macro = { "#define COMPUTE_SCREENSHOTER" };
	OpenGLShader default_display_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/default_display.frag", OpenGLShader::COMPUTE_SHADER, macro);
	OpenGLShader normal_display_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/normal_display.frag", OpenGLShader::COMPUTE_SHADER, macro);
	OpenGLShader albedo_display_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/albedo_display.frag", OpenGLShader::COMPUTE_SHADER, macro);
	OpenGLShader adaptative_display_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/heatmap_int.frag", OpenGLShader::COMPUTE_SHADER, macro);

	m_default_compute_program.attach(default_display_shader);
	m_default_compute_program.link();

	m_normal_compute_program.attach(normal_display_shader);
	m_normal_compute_program.link();

	m_albedo_compute_program.attach(albedo_display_shader);
	m_albedo_compute_program.link();

	m_adaptative_sampling_compute_program.attach(adaptative_display_shader);
	m_adaptative_sampling_compute_program.link();

	select_compute_program(m_render_window->get_application_settings().display_view);
}

void Screenshoter::prepare_output_image(int width, int height)
{
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
		glBindImageTexture(/* location in the shader */ 2, m_output_image, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
	}
	else
		glBindTexture(GL_TEXTURE_2D, m_output_image);
}

void Screenshoter::write_to_png(const char* filepath)
{
	int width = m_renderer->m_render_width;
	int height = m_renderer->m_render_height;

	if (m_render_window->get_application_settings().render_resolution_scale == 1.0f)
	{
		// Fast path when no resolution scaling, we can just dump the viewport to a file
		// because the viewport is the same resolution as the render resolution so the viewport
		// is exactly what we should have in the screenshot
		std::vector<unsigned char> mapped_data(width * height * 3);
		glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, mapped_data.data());

		stbi_flip_vertically_on_write(true);
		if (stbi_write_png(filepath, width, height, 3, mapped_data.data(), width * sizeof(unsigned char) * 3))
			std::cout << "Render written to \"" << filepath << "\"" << std::endl;
	}
	else
	{
		// If the viewport isn't the same resolution as the render resolution (when render_resolution_scale != 1.0f),
		// we're going to have to get the buffer of the renderer and apply post-processing to it 
		// (tone mapping, heatmap visualization, ...) in order the get the same visuals as in the viewport
		// This post-processing is done using a compute shader. 
		// 
		// We're using OpenGL compute shader here and not an HIP kernel because we want to be able to use the same 
		// fragment shader files that we use for the displaying. If we were doing the post-processing with an HIP kernel, 
		// we would have to write HIP kernels that would basically be copy-pasting of the OpenGL display shaders 
		// with just some syntax changes. That would basically mean duplicating code which would be annoying to 
		// maintain because we would have to update the HIP kernels everytime we changed the OpenGL display shader 
		// so that the screenshoter outputs the correct image (and needless to say that we would forget, most of time, 
		// to update the HIP kernels so that's why code duplication here is annoying)

		if (!m_compute_shader_initialized)
		{
			// We're initializing the compute shader here because if we do it in the constructor, the 
			// constructor of RenderWindow which contains an instance of Screenshoter is going to try and 
			// use OpenGL functions even though OpenGL hasn't even been initialized in the application yet
			initialize_programs();
			m_compute_shader_initialized = true;
		}

		prepare_output_image(width, height);

		GLint threads[3];
		m_active_compute_program.get_compute_threads(threads);

		int nb_groups_x = std::ceil(width / (float)threads[0]);
		int nb_groups_y = std::ceil(height / (float)threads[1]);

		m_render_window->update_program_uniforms(m_active_compute_program);

		glDispatchCompute(nb_groups_x, nb_groups_y, 1);
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		std::vector<unsigned char> mapped_data(width * height * 4);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, mapped_data.data());

		stbi_flip_vertically_on_write(true);
		if (stbi_write_png(filepath, width, height, 4, mapped_data.data(), width * sizeof(unsigned char) * 4))
			std::cout << "Render written to \"" << filepath << "\"" << std::endl;
	}
}

