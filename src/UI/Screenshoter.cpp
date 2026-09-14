/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "GL/glew.h"
#include "stb_image_write.h"
#include "UI/ImGui/ImGuiLogger.h"
#include "UI/RenderWindow.h"
#include "UI/Screenshoter.h"
#include "Utils/Utils.h"

#include "OpenGL/OpenGLInteropBuffer.h"

#include <algorithm>

extern ImGuiLogger g_imgui_logger;

Screenshoter::Screenshoter()
{
	std::vector<std::string> macro = { "#define COMPUTE_SCREENSHOTER" };

	OpenGLShader default_display_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/default_display.frag", OpenGLShader::COMPUTE_SHADER, macro);

	std::shared_ptr<OpenGLProgram> default_display_program = std::make_shared<OpenGLProgram>();

	default_display_program->attach(default_display_shader);
	default_display_program->link();

	m_compute_programs[DisplayViewType::DEFAULT] = default_display_program;

	select_compute_program(DisplayViewType::DEFAULT);
}

void Screenshoter::set_renderer(std::shared_ptr<GPURenderer> renderer)
{
	m_renderer = renderer;
}

void Screenshoter::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;
}

void Screenshoter::select_compute_program(DisplayViewType display_view)
{
	m_active_compute_program = m_compute_programs[display_view];
}

void Screenshoter::write_to_png()
{
	std::stringstream filename;

	Utils::get_current_date_string(filename);
	filename << " - " << m_renderer->get_render_settings().sample_number << "spp - " << m_render_window->get_current_render_time_ms() / 1000.0f << "s"
			 << ".png";

	write_to_png(filename.str().c_str());
}

void Screenshoter::resize_output_image(int width, int height)
{
	bool texture_needs_creation = false;
	if (m_compute_output_image_width == -1)
		texture_needs_creation = true;
	else if (m_compute_output_image_width != width || m_compute_output_image_height != height)
	{
		glDeleteTextures(1, &m_output_image);
		texture_needs_creation = true;
	}

	if (texture_needs_creation)
	{
		m_compute_output_image_width  = width;
		m_compute_output_image_height = height;

		glGenTextures(1, &m_output_image);
		glActiveTexture(GL_TEXTURE0 + DisplayViewSystem::DISPLAY_COMPUTE_IMAGE_UNIT);
		glBindTexture(GL_TEXTURE_2D, m_output_image);
		glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8UI, width, height);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
		glBindImageTexture(/* location in the shader */ 2, m_output_image, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8UI);
	}
	else
	{
		glActiveTexture(GL_TEXTURE0 + DisplayViewSystem::DISPLAY_COMPUTE_IMAGE_UNIT);
		glBindTexture(GL_TEXTURE_2D, m_output_image);
	}
}

void Screenshoter::write_to_png(std::string filepath)
{
	Image8Bit image = get_image();
	if (image.write_image_png(filepath.c_str(), false))
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Screenshot written to \"%s\"", filepath.c_str());
}

Image8Bit Screenshoter::get_image(bool flip_y)
{
	int width					 = m_renderer->m_render_resolution.x;
	int height					 = m_renderer->m_render_resolution.y;
	DisplayViewType display_view = m_render_window->get_display_view_system()->get_current_display_view_type();

	if (display_view == DisplayViewType::DEFAULT || display_view == DisplayViewType::GMON_BLEND || display_view == DisplayViewType::DENOISED_BLEND ||
		display_view == DisplayViewType::DISPLAY_DENOISER_ALBEDO || display_view == DisplayViewType::DISPLAY_DENOISER_NORMALS ||
		display_view == DisplayViewType::WHITE_FURNACE_THRESHOLD)
		return get_final_output_image(flip_y);

	m_renderer->synchronize_all_kernels();
	m_renderer->unmap_buffers();
	// We upload the data to the OpenGL textures for displaying
	m_render_window->get_display_view_system()->upload_relevant_buffers_to_texture();

	resize_output_image(width, height);
	select_compute_program(display_view);

	GLint threads[3];
	m_active_compute_program->get_compute_threads(threads);

	int nb_groups_x = std::ceil(width / (float)threads[0]);
	int nb_groups_y = std::ceil(height / (float)threads[1]);

	DisplayViewSystem::update_display_program_uniforms(m_render_window->get_display_view_system().get(), m_active_compute_program, m_renderer,
													   m_render_window->get_application_settings());

	glDispatchCompute(nb_groups_x, nb_groups_y, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	std::vector<unsigned char> mapped_data(width * height * 4);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, mapped_data.data());

	Image8Bit image(mapped_data, width, height, 4);
	if (flip_y)
		image.flip_vertically();

	return image;
}

Image8Bit Screenshoter::get_final_output_image(bool flip_y)
{
	int width  = m_renderer->m_render_resolution.x;
	int height = m_renderer->m_render_resolution.y;

	// The device pass is the single source of truth for every display view.
	m_renderer->launch_display_post_process();

	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> final_framebuffer = m_renderer->get_display_post_process_interop_framebuffer();
	ColorRGB32F* final_framebuffer_device_pointer						= final_framebuffer->map();
	std::vector<ColorRGB32F> final_colors =
		OrochiBuffer<ColorRGB32F>::download_data(final_framebuffer_device_pointer, static_cast<size_t>(width) * static_cast<size_t>(height));
	final_framebuffer->unmap();

	std::vector<unsigned char> image_data(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
	for (size_t pixel_index = 0; pixel_index < final_colors.size(); pixel_index++)
	{
		ColorRGB32F final_color = final_colors[pixel_index];
		final_color.r			= std::clamp(final_color.r, 0.0f, 1.0f);
		final_color.g			= std::clamp(final_color.g, 0.0f, 1.0f);
		final_color.b			= std::clamp(final_color.b, 0.0f, 1.0f);

		image_data[pixel_index * 4 + 0] = static_cast<unsigned char>(final_color.r * 255.0f);
		image_data[pixel_index * 4 + 1] = static_cast<unsigned char>(final_color.g * 255.0f);
		image_data[pixel_index * 4 + 2] = static_cast<unsigned char>(final_color.b * 255.0f);
		image_data[pixel_index * 4 + 3] = 255;
	}

	Image8Bit image(image_data, width, height, 4);
	if (flip_y)
		image.flip_vertically();

	return image;
}
