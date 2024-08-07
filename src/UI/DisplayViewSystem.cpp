/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/ApplicationSettings.h"
#include "UI/DisplayViewSystem.h"
#include "UI/RenderWindow.h"

DisplayViewSystem::DisplayViewSystem(std::shared_ptr<GPURenderer> renderer, RenderWindow* render_window)
{
	m_renderer = renderer;
	m_render_window = render_window;

	// Creating the texture that will contain the path traced data to be displayed
	// by the shader.
	glGenTextures(1, &m_display_texture_1.first);
	glGenTextures(1, &m_display_texture_2.first);

	// This empty VAO is necessary on NVIDIA drivers even though
	// we're hardcoding our full screen quad in the vertex shader
	glCreateVertexArrays(1, &m_vao);

	OpenGLShader fullscreen_quad_vertex_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/fullscreen_quad.vert", OpenGLShader::VERTEX_SHADER);
	OpenGLShader default_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/default_display.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader blend_2_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/blend_2_display.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader normal_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/normal_display.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader albedo_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/albedo_display.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader adaptive_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/heatmap_int.frag", OpenGLShader::FRAGMENT_SHADER);

	// Making shared_ptr<OpenGLProgram>s here because multiple display views may share the same OpenGLProgram
	std::shared_ptr<OpenGLProgram> default_display_program = std::make_shared<OpenGLProgram>(fullscreen_quad_vertex_shader, default_display_fragment_shader);
	std::shared_ptr<OpenGLProgram> denoise_blend_display_program = std::make_shared<OpenGLProgram>(fullscreen_quad_vertex_shader, blend_2_display_fragment_shader);
	std::shared_ptr<OpenGLProgram> normal_display_program = std::make_shared<OpenGLProgram>(fullscreen_quad_vertex_shader, normal_display_fragment_shader);
	std::shared_ptr<OpenGLProgram> albedo_display_program = std::make_shared<OpenGLProgram>(fullscreen_quad_vertex_shader, albedo_display_fragment_shader);
	std::shared_ptr<OpenGLProgram> adaptive_sampling_display_program = std::make_shared<OpenGLProgram>(fullscreen_quad_vertex_shader, adaptive_display_fragment_shader);

	// Creating all the texture views
	DisplayView default_display_view = DisplayView(DisplayViewType::DEFAULT, default_display_program);
	DisplayView denoise_blend_display_view = DisplayView(DisplayViewType::DENOISED_BLEND, denoise_blend_display_program);
	DisplayView normals_display_view = DisplayView(DisplayViewType::DISPLAY_NORMALS, normal_display_program);
	DisplayView normals_denoised_display_view = DisplayView(DisplayViewType::DISPLAY_DENOISED_NORMALS, normal_display_program);
	DisplayView albedo_display_view = DisplayView(DisplayViewType::DISPLAY_ALBEDO, albedo_display_program);
	DisplayView albedo_denoised_display_view = DisplayView(DisplayViewType::DISPLAY_DENOISED_ALBEDO, albedo_display_program);
	DisplayView adaptive_sampling_display_view = DisplayView(DisplayViewType::ADAPTIVE_SAMPLING_MAP, adaptive_sampling_display_program);

	// Adding the display views to the map
	m_display_views[DisplayViewType::DEFAULT] = default_display_view;
	m_display_views[DisplayViewType::DENOISED_BLEND] = denoise_blend_display_view;
	m_display_views[DisplayViewType::DISPLAY_NORMALS] = normals_display_view;
	m_display_views[DisplayViewType::DISPLAY_DENOISED_NORMALS] = normals_denoised_display_view;
	m_display_views[DisplayViewType::DISPLAY_ALBEDO] = albedo_display_view;
	m_display_views[DisplayViewType::DISPLAY_DENOISED_ALBEDO] = albedo_denoised_display_view;
	m_display_views[DisplayViewType::ADAPTIVE_SAMPLING_MAP] = adaptive_sampling_display_view;

	// Denoiser blend by default because we want 
	DisplayViewType default_display_view_type;
	default_display_view_type = m_render_window->get_application_settings()->enable_denoising ? DisplayViewType::DENOISED_BLEND : DisplayViewType::DEFAULT;
	queue_display_view_change(default_display_view_type);
}

DisplayViewSystem::~DisplayViewSystem()
{
	glDeleteTextures(1, &m_display_texture_1.first);
	glDeleteTextures(1, &m_display_texture_2.first);
	glDeleteVertexArrays(1, &m_vao);
}

bool DisplayViewSystem::update_selected_display_view()
{
	if (m_queued_display_view_change != DisplayViewType::UNDEFINED)
	{
		// Adjusting the denoiser setting according to the selected view
		m_render_window->get_application_settings()->enable_denoising = (m_queued_display_view_change == DisplayViewType::DENOISED_BLEND);

		m_current_display_view = &m_display_views[m_queued_display_view_change];

		internal_recreate_display_textures_from_display_view(m_queued_display_view_change);

		m_queued_display_view_change = DisplayViewType::UNDEFINED;

		return true;
	}
	
	return false;
}

void DisplayViewSystem::display()
{
	// Binding an empty VAO here (empty because we're hardcoding our full-screen quad vertices
	// in our vertex shader) because this is required on NVIDIA drivers
	glBindVertexArray(m_vao);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

DisplayViewType DisplayViewSystem::get_current_display_view_type()
{
	return m_current_display_view->get_display_view_type();
}

const DisplayView* DisplayViewSystem::get_current_display_view() const
{
	return m_current_display_view;
}

std::shared_ptr<OpenGLProgram> DisplayViewSystem::get_active_display_program()
{
	return m_current_display_view->get_display_program();
}

void DisplayViewSystem::update_display_program_uniforms(const DisplayViewSystem* display_view_system, std::shared_ptr<OpenGLProgram> program, std::shared_ptr<GPURenderer> renderer, std::shared_ptr<ApplicationSettings> application_settings)
{
	const DisplayView* display_view = display_view_system->get_current_display_view();
	
	HIPRTRenderSettings render_settings = renderer->get_render_settings();
	render_settings.sample_number = std::max(1, render_settings.sample_number); 

	bool display_low_resolution = display_view_system->get_render_low_resolution();
	int render_low_resolution_scaling = display_low_resolution ? render_settings.render_low_resolution_scaling : 1;

	program->use();

	switch (display_view->get_display_view_type())
	{
	case DisplayViewType::DEFAULT:
		int sample_number;
		if (application_settings->enable_denoising && application_settings->last_denoised_sample_count != -1)
			sample_number = application_settings->last_denoised_sample_count;
		else
			sample_number = render_settings.sample_number;

		program->set_uniform("u_texture", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		program->set_uniform("u_sample_number", sample_number);
		program->set_uniform("u_do_tonemapping", application_settings->do_tonemapping);
		program->set_uniform("u_resolution_scaling", render_low_resolution_scaling);
		program->set_uniform("u_gamma", application_settings->tone_mapping_gamma);
		program->set_uniform("u_exposure", application_settings->tone_mapping_exposure);

		break;

	case DisplayViewType::DENOISED_BLEND:
		int noisy_sample_number;
		int denoised_sample_number;

		noisy_sample_number = render_settings.sample_number;
		denoised_sample_number = application_settings->last_denoised_sample_count;

		if (application_settings->blend_override != -1.0f)
			program->set_uniform("u_blend_factor", application_settings->blend_override);
		else
			program->set_uniform("u_blend_factor", application_settings->denoiser_blend);
		program->set_uniform("u_texture_1", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		program->set_uniform("u_texture_2", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_2);
		program->set_uniform("u_sample_number_1", noisy_sample_number);
		program->set_uniform("u_sample_number_2", denoised_sample_number);
		program->set_uniform("u_do_tonemapping", application_settings->do_tonemapping);
		program->set_uniform("u_resolution_scaling", render_low_resolution_scaling);
		program->set_uniform("u_gamma", application_settings->tone_mapping_gamma);
		program->set_uniform("u_exposure", application_settings->tone_mapping_exposure);

		break;

	case DisplayViewType::DISPLAY_ALBEDO:
	case DisplayViewType::DISPLAY_DENOISED_ALBEDO:
		program->set_uniform("u_texture", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		program->set_uniform("u_resolution_scaling", render_low_resolution_scaling);

		break;

	case DisplayViewType::DISPLAY_NORMALS:
	case DisplayViewType::DISPLAY_DENOISED_NORMALS:
		program->set_uniform("u_texture", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		program->set_uniform("u_resolution_scaling", render_low_resolution_scaling);
		program->set_uniform("u_do_tonemapping", application_settings->do_tonemapping);
		program->set_uniform("u_gamma", application_settings->tone_mapping_gamma);
		program->set_uniform("u_exposure", application_settings->tone_mapping_exposure);

		break;

	case DisplayViewType::ADAPTIVE_SAMPLING_MAP:
		std::vector<ColorRGB32F> color_stops = { ColorRGB32F(0.0f, 0.0f, 1.0f), ColorRGB32F(0.0f, 1.0f, 0.0f), ColorRGB32F(1.0f, 0.0f, 0.0f) };

		float min_val = (float)render_settings.adaptive_sampling_min_samples;
		float max_val = std::max((float)render_settings.sample_number, min_val);

		program->set_uniform("u_texture", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		program->set_uniform("u_resolution_scaling", render_low_resolution_scaling);
		program->set_uniform("u_color_stops", 3, (float*)color_stops.data());
		program->set_uniform("u_nb_stops", 3);
		program->set_uniform("u_min_val", min_val);
		program->set_uniform("u_max_val", max_val);

		break;
	}
}

void DisplayViewSystem::update_current_display_program_uniforms()
{
	DisplayViewSystem::update_display_program_uniforms(this, get_active_display_program(), m_renderer, m_render_window->get_application_settings());
}

void DisplayViewSystem::upload_relevant_buffers_to_texture()
{
	DisplayViewType current_display_view_type = get_current_display_view_type();

	switch (current_display_view_type)
	{
	case DisplayViewType::DENOISED_BLEND:
		internal_upload_buffer_to_texture(m_renderer->get_color_framebuffer(), m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		internal_upload_buffer_to_texture(m_renderer->get_denoised_framebuffer(), m_display_texture_2, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_2);
		break;

	case DisplayViewType::DISPLAY_ALBEDO:
		internal_upload_buffer_to_texture(m_renderer->get_denoiser_albedo_AOV_buffer(), m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		break;

	case DisplayViewType::DISPLAY_NORMALS:
		internal_upload_buffer_to_texture(m_renderer->get_denoiser_normals_AOV_buffer(), m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		break;

		// TODO fix
		/*case DisplayViewType::DISPLAY_DENOISED_NORMALS:
			m_render_window_denoiser->denoise_normals();
			display(m_render_window_denoiser->get_denoised_normals_pointer());
			break;*/


			// TODO fix
			/*case DisplayViewType::DISPLAY_DENOISED_ALBEDO:
				m_render_window_denoiser->denoise_albedo();
				display(m_render_window_denoiser->get_denoised_albedo_pointer());
				break;*/

	case DisplayViewType::ADAPTIVE_SAMPLING_MAP:
		internal_upload_buffer_to_texture(m_renderer->get_pixels_sample_count_buffer(), m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		break;

		// TODO fix
		/*case DisplayViewType::ADAPTIVE_SAMPLING_ACTIVE_PIXELS:
			display(m_renderer->get_debug_pixel_active_buffer().download_data().data());
			break;*/

	case DisplayViewType::DEFAULT:
	default:
		internal_upload_buffer_to_texture(m_renderer->get_color_framebuffer(), m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		break;
	}
}

void DisplayViewSystem::queue_display_view_change(DisplayViewType display_view)
{
	m_queued_display_view_change = display_view;
}

void DisplayViewSystem::set_render_low_resolution(bool low_resolution_or_not)
{
	m_displaying_low_resolution = low_resolution_or_not;
}

bool DisplayViewSystem::get_render_low_resolution() const
{
	return m_displaying_low_resolution;
}

void DisplayViewSystem::resize(int new_render_width, int new_render_height)
{
	internal_recreate_display_texture(m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1, m_display_texture_1.second, new_render_width, new_render_height);
	internal_recreate_display_texture(m_display_texture_2, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_2, m_display_texture_2.second, new_render_width, new_render_height);
}

void DisplayViewSystem::internal_recreate_display_textures_from_display_view(DisplayViewType display_view)
{
	DisplayTextureType texture_1_type_needed = DisplayTextureType::UNINITIALIZED;
	DisplayTextureType texture_2_type_needed = DisplayTextureType::UNINITIALIZED;

	switch (display_view)
	{
	case DisplayViewType::DISPLAY_NORMALS:
	case DisplayViewType::DISPLAY_DENOISED_NORMALS:
	case DisplayViewType::DISPLAY_ALBEDO:
	case DisplayViewType::DISPLAY_DENOISED_ALBEDO:
		texture_1_type_needed = DisplayTextureType::FLOAT3;
		break;

	case DisplayViewType::ADAPTIVE_SAMPLING_MAP:
		texture_1_type_needed = DisplayTextureType::INT;
		break;

	case DisplayViewType::DENOISED_BLEND:
		texture_1_type_needed = DisplayTextureType::FLOAT3;
		texture_2_type_needed = DisplayTextureType::FLOAT3;
		break;

	case DisplayViewType::DEFAULT:
	default:
		texture_1_type_needed = DisplayTextureType::FLOAT3;
		break;
	}

	if (m_display_texture_1.second != texture_1_type_needed)
		internal_recreate_display_texture(m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1, texture_1_type_needed, m_renderer->m_render_width, m_renderer->m_render_height);

	if (m_display_texture_2.second != texture_2_type_needed)
		internal_recreate_display_texture(m_display_texture_2, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_2, texture_2_type_needed, m_renderer->m_render_width, m_renderer->m_render_height);
}

void DisplayViewSystem::internal_recreate_display_texture(std::pair<GLuint, DisplayTextureType>& display_texture, GLenum display_texture_unit, DisplayTextureType new_texture_type, int width, int height)
{
	bool freeing = false;
	if (new_texture_type == DisplayTextureType::UNINITIALIZED)
	{
		if (display_texture.second != DisplayTextureType::UNINITIALIZED)
		{
			// If the texture was valid before and we've given UNINITIALIZED as the new type, this means
			// that we're not using the texture anymore. We're going to queue_resize the texture to 1x1,
			// essentially freeing it but without really destroying the OpenGL object
			width = height = 1;

			// Not changing the texture type, just resizing
			new_texture_type = display_texture.second;

			freeing = true;
		}
		else
			// Else, the texture is already UNINITIALIZED
			return;
	}

	GLint internal_format = new_texture_type.get_gl_internal_format();
	GLenum format = new_texture_type.get_gl_format();
	GLenum type = new_texture_type.get_gl_type();

	// Making sure the buffer isn't bound
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glActiveTexture(GL_TEXTURE0 + display_texture_unit);
	glBindTexture(GL_TEXTURE_2D, display_texture.first);
	glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, type, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	if (freeing)
		// If we just freed the texture, setting it as UNINITIALIZED so that it is basically invalidated
		// and will be recreated correctly next time
		display_texture.second = DisplayTextureType::UNINITIALIZED;
	else
		display_texture.second = new_texture_type;
}
