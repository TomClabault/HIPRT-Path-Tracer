/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "GL/glew.h"
#include "stb_image_write.h"
#include "UI/ImGui/ImGuiLogger.h"
#include "UI/RenderWindow.h"
#include "UI/Screenshoter.h"
#include "Utils/Utils.h"

extern ImGuiLogger g_imgui_logger;

Screenshoter::Screenshoter()
{
	std::vector<std::string> macro = { "#define COMPUTE_SCREENSHOTER" };

	OpenGLShader default_display_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/default_display.frag", OpenGLShader::COMPUTE_SHADER, macro);
	OpenGLShader blend_2_display_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/blend_2_display.frag", OpenGLShader::COMPUTE_SHADER, macro);
	OpenGLShader normal_display_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/normal_display.frag", OpenGLShader::COMPUTE_SHADER, macro);
	OpenGLShader albedo_display_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/albedo_display.frag", OpenGLShader::COMPUTE_SHADER, macro);
	OpenGLShader adaptive_display_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/heatmap_int.frag", OpenGLShader::COMPUTE_SHADER, macro);
	OpenGLShader pixel_converged_map_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/boolmap_int.frag", OpenGLShader::COMPUTE_SHADER, macro);
	OpenGLShader white_furnace_display_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/white_furnace_threshold.frag", OpenGLShader::COMPUTE_SHADER, macro);

	std::shared_ptr<OpenGLProgram> default_display_program = std::make_shared<OpenGLProgram>();
	std::shared_ptr<OpenGLProgram> blend_2_display_program = std::make_shared<OpenGLProgram>();
	std::shared_ptr<OpenGLProgram> normal_display_program = std::make_shared<OpenGLProgram>();
	std::shared_ptr<OpenGLProgram> albedo_display_program = std::make_shared<OpenGLProgram>();
	std::shared_ptr<OpenGLProgram> pixel_convergence_heatmap_display_program = std::make_shared<OpenGLProgram>();
	std::shared_ptr<OpenGLProgram> pixel_converged_map_display_program = std::make_shared<OpenGLProgram>();
	std::shared_ptr<OpenGLProgram> white_furnace_display_program = std::make_shared<OpenGLProgram>();

	default_display_program->attach(default_display_shader);
	default_display_program->link();

	blend_2_display_program->attach(blend_2_display_shader);
	blend_2_display_program->link();

	normal_display_program->attach(normal_display_shader);
	normal_display_program->link();

	albedo_display_program->attach(albedo_display_shader);
	albedo_display_program->link();

	pixel_convergence_heatmap_display_program->attach(adaptive_display_shader);
	pixel_convergence_heatmap_display_program->link();

	pixel_converged_map_display_program->attach(pixel_converged_map_shader);
	pixel_converged_map_display_program->link();

	white_furnace_display_program->attach(white_furnace_display_shader);
	white_furnace_display_program->link();

	m_compute_programs[DisplayViewType::DEFAULT] = default_display_program;
	m_compute_programs[DisplayViewType::DENOISED_BLEND] = blend_2_display_program;
	m_compute_programs[DisplayViewType::DISPLAY_DENOISER_ALBEDO] = albedo_display_program;
	m_compute_programs[DisplayViewType::DISPLAY_DENOISED_ALBEDO] = albedo_display_program;
	m_compute_programs[DisplayViewType::DISPLAY_DENOISER_NORMALS] = normal_display_program;
	m_compute_programs[DisplayViewType::DISPLAY_DENOISED_NORMALS] = normal_display_program;
	m_compute_programs[DisplayViewType::PIXEL_CONVERGENCE_HEATMAP] = pixel_convergence_heatmap_display_program;
	m_compute_programs[DisplayViewType::PIXEL_CONVERGED_MAP] = pixel_converged_map_display_program;
	m_compute_programs[DisplayViewType::WHITE_FURNACE_THRESHOLD] = white_furnace_display_program;

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
	filename << " - " << m_renderer->get_render_settings().sample_number << "sp @ " << m_renderer->m_render_resolution.x << "x" << m_renderer->m_render_resolution.y << " - " << m_render_window->get_current_render_time() / 1000.0f << "s" << ".png";

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
		m_compute_output_image_width = width;
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
	write_to_png(filepath.c_str());
}

void Screenshoter::write_to_png(const char* filepath)
{
	int width = m_renderer->m_render_resolution.x;
	int height = m_renderer->m_render_resolution.y;


	// We're using OpenGL compute shader here and not an HIP kernel because we want to be able to use the same 
	// fragment shader files that we use for the displaying. If we were doing the post-processing with an HIP kernel, 
	// we would have to write HIP kernels that would basically be copy-pasting of the OpenGL display shaders 
	// with just some syntax changes. That would basically mean duplicating code which would be annoying to 
	// maintain because we would have to update the HIP kernels everytime we changed the OpenGL display shader 
	// so that the screenshoter outputs the correct image (and needless to say that we would forget, most of time, 
	// to update the HIP kernels so that's why code duplication here is annoying)

	m_renderer->synchronize_kernel();
	m_renderer->unmap_buffers();
	// We upload the data to the OpenGL textures for displaying
	m_render_window->get_display_view_system()->upload_relevant_buffers_to_texture();

	resize_output_image(width, height);
	select_compute_program(m_render_window->get_display_view_system()->get_current_display_view_type());

	GLint threads[3];
	m_active_compute_program->get_compute_threads(threads);

	int nb_groups_x = std::ceil(width / (float)threads[0]);
	int nb_groups_y = std::ceil(height / (float)threads[1]);

	DisplayViewSystem::update_display_program_uniforms(m_render_window->get_display_view_system().get(), m_active_compute_program, m_renderer, m_render_window->get_application_settings());

	glDispatchCompute(nb_groups_x, nb_groups_y, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	std::vector<unsigned char> mapped_data(width * height * 4);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, mapped_data.data());

	stbi_flip_vertically_on_write(true);
	if (stbi_write_png(filepath, width, height, 4, mapped_data.data(), width * sizeof(unsigned char) * 4))
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Screenshot written to \"%s\"", filepath);
}

