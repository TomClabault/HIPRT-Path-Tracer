#include "UI/app_window.h"

#include <functional>
#include <iostream>
#include "Scene/scene_parser.h"
#include "Utils/utils.h"

//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// TODO Code Organization:
// 
// - create image class instead of vector of HIPRTColor pretty much everywhere
// - rename HIPRTColor to Color
// - create env map class that encapsulates image + cdf + sampling functions
// - delete tests, they are obsolete
// - overload +=, *=, ... operators for HIPRTColor most notably on the GPU side
// - use constructors instead of struct {} syntax in gpu code
// - rename HIPRT_xorshift32 generator without underscores for consistency
// - separate path tracer kernel functions in header files
// - do not duplicate functions. Make a common h file that uses the float3 type (cosine_weighted_direction_around_normal, hiprt_lambertian.h:7)
// - instead of duplicating structures (RendererMaterial + HIPRTRendererMaterial, HIPRTBRDF + BRDF, ...), it would be better to create a folder
//		HostDeviceCommon containing the structures that are used both by the GPU and CPU renderer
// - imgui controller to put all the imgui code in one class

// TODO Features:
// 
// - cutout filters
// - write scene details to imgui (nb vertices, triangles, ...)
// - check perf of aiPostProcessSteps::aiProcess_ImproveCacheLocality
// - ImGui to choose the flags at runtime and be able to compare the performance
// - try to use the denoising buffer directly in the GPU instead of having to copy from gpu to denoising buffer everytime
// - use memcpy for efficiency in the denoiser to copy
// - light sampling: go through transparent surfaces instead of considering them opaque
// - BVH compaction + imgui checkbox
// - shader cache
// - indirect / direct lighting clamping
// - env map support
// - choose env map at runtime imgui
// - env map rotation imgui
// - albedo and normals denoising
// - choose scene file at runtime imgui
// - lock camera checkbox to avoid messing up when big render in progress
// - choose render resolution in imgui
// - choose viewport resolution in imgui
// - compute shader for tone mapping images ? unless transfering memory to open gl is too expensive

void wait_and_exit(const char* message)
{
	std::cerr << message << std::endl;
	std::getchar();

	std::exit(1);
}

void glfw_window_resized_callback(GLFWwindow* window, int width, int height)
{
	int new_width_pixels, new_height_pixels;
	glfwGetFramebufferSize(window, &new_width_pixels, &new_height_pixels);

	if (new_width_pixels == 0 || new_height_pixels == 0)
		// This probably means that the application has been minimized, we're not doing anything then
		return;
	else
		// We've stored a pointer to the AppWindow in the "WindowUserPointer" of glfw
		reinterpret_cast<AppWindow*>(glfwGetWindowUserPointer(window))->resize_frame(width, height);
}

void glfw_mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos)
{
	ImGuiIO& io = ImGui::GetIO();
	if (!io.WantCaptureMouse)
	{
		AppWindow* app_window = reinterpret_cast<AppWindow*>(glfwGetWindowUserPointer(window));

		float xposf = static_cast<float>(xpos);
		float yposf = static_cast<float>(ypos);

		std::pair<float, float> old_position = app_window->get_cursor_position();
		if (old_position.first == -1 && old_position.second == -1)
			;
		// If this is the first position of the cursor, nothing to do
		else
		{
			// Computing the difference in movement
			std::pair<float, float> difference = std::make_pair(xposf - old_position.first, yposf - old_position.second);

			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
				app_window->update_renderer_view_translation(-difference.first, difference.second);

			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
				app_window->update_renderer_view_rotation(-difference.first, -difference.second);
		}

		// Updating the position
		app_window->set_cursor_position(std::make_pair(xposf, yposf));
	}
}

static bool z_pressed, q_pressed, s_pressed, d_pressed, space_pressed, lshift_pressed;
void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	switch (key)
	{
	case GLFW_KEY_W:
	case GLFW_KEY_Z:
		z_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
		break;

	case GLFW_KEY_A:
	case GLFW_KEY_Q:
		q_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
		break;

	case GLFW_KEY_S:
		s_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);

		break;

	case GLFW_KEY_D:
		d_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
		break;

	case GLFW_KEY_SPACE:
		space_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
		break;

	case GLFW_KEY_LEFT_SHIFT:
		lshift_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
		break;

	default:
		break;
	}

	float zoom = 0.0f;
	std::pair<float, float> translation;
	if (z_pressed)
		zoom += 1.0f;
	if (q_pressed)
		translation.first += 36.0f;
	if (s_pressed)
		zoom -= 1.0f;
	if (d_pressed)
		translation.first -= 36.0f; // TODO fix should be opposite of Q
	if (space_pressed)
		translation.second += 36.0f;
	if (lshift_pressed)
		translation.second -= 36.0f;


	AppWindow* app_window = reinterpret_cast<AppWindow*>(glfwGetWindowUserPointer(window));
	app_window->update_renderer_view_translation(-translation.first, translation.second);
	app_window->update_renderer_view_zoom(-zoom);
}

void glfw_mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	ImGuiIO& io = ImGui::GetIO();
	if (!io.WantCaptureMouse)
	{
		AppWindow* app_window = reinterpret_cast<AppWindow*>(glfwGetWindowUserPointer(window));

		app_window->update_renderer_view_zoom(static_cast<float>(-yoffset));
	}
}

// Implementation from https://learnopengl.com/In-Practice/Debugging
void APIENTRY AppWindow::gl_debug_output_callback(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar* message,
	const void* userParam)
{
	// ignore non-significant error/warning codes
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

	std::cout << "---------------" << std::endl;
	std::cout << "Debug message (" << id << "): " << message << std::endl;

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
	case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
	} std::cout << std::endl;

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
	case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
	case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
	} std::cout << std::endl;

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
	case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
	} std::cout << std::endl;
	std::cout << std::endl;
}

AppWindow::AppWindow(int width, int height) : m_viewport_width(width), m_viewport_height(height)
{
	if (!glfwInit())
		wait_and_exit("Could not initialize GLFW...");

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
	m_window = glfwCreateWindow(width, height, "HIPRT Path Tracer", NULL, NULL);
	if (!m_window)
		wait_and_exit("Could not initialize the GLFW window...");

	glfwMakeContextCurrent(m_window);
	// Setting a pointer to this instance of AppWindow inside the m_window GLFWwindow so that
	// we can retrieve a pointer to this instance of AppWindow in the callback functions
	// such as the window_resized_callback function for example
	glfwSetWindowUserPointer(m_window, this);
	glfwSetWindowSizeCallback(m_window, glfw_window_resized_callback);
	glfwSetCursorPosCallback(m_window, glfw_mouse_cursor_callback);
	glfwSetKeyCallback(m_window, glfw_key_callback);
	glfwSetScrollCallback(m_window, glfw_mouse_scroll_callback);
	glfwSwapInterval(1);

	glewInit();

	glViewport(0, 0, width, height);

	// Initializing the debug output of OpenGL to catch errors
	// when calling OpenGL function with an incorrect OpenGL state
	int flags;
	glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
	if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
	{
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(AppWindow::gl_debug_output_callback, nullptr);
		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
	}

	// Setting ImGui up
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	ImGui_ImplGlfw_InitForOpenGL(m_window, true);
	ImGui_ImplOpenGL3_Init();

	setup_display_program();
	m_renderer.init_ctx(0);
	m_renderer.compile_trace_kernel(m_application_settings.kernel_files[m_application_settings.selected_kernel].c_str(),
		m_application_settings.kernel_functions[m_application_settings.selected_kernel].c_str());
	m_renderer.change_render_resolution(width, height);
	m_denoiser.resize_buffers(width, height);
}

AppWindow::~AppWindow()
{
	glfwDestroyWindow(m_window);
	glfwTerminate();
}

void AppWindow::resize_frame(int pixels_width, int pixels_height)
{
	if (pixels_width == m_viewport_width && pixels_height == m_viewport_height)
	{
		// Already the right size, nothing to do. This can happen
		// when the window comes out of the minized state. Getting
		// in the minimized state triggers a resize event with a new size
		// of (0, 0) and getting out of the minimized state triggers a resize
		// event with a size equal to the one before the minimization, which means
		// that the window wasn't actually resized and there is nothing to do

		return;
	}

	glViewport(0, 0, pixels_width, pixels_height);

	m_viewport_width = pixels_width;
	m_viewport_height = pixels_height;

	RenderSettings& render_settings = m_renderer.get_render_settings();
	// Taking resolution scaling into account
	float& resolution_scale = render_settings.render_resolution_scale;
	if (render_settings.keep_same_resolution)
		resolution_scale = render_settings.target_width / (float)pixels_width; // TODO what about the height changing ?

	int new_render_width = std::floor(pixels_width * resolution_scale);
	int new_render_height = std::floor(pixels_height * resolution_scale);
	m_renderer.change_render_resolution(new_render_width, new_render_height);
	m_denoiser.resize_buffers(new_render_width, new_render_height);

	// Recreating the OpenGL display texture
	glActiveTexture(GL_TEXTURE0 + AppWindow::DISPLAY_TEXTURE_UNIT);
	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, new_render_width, new_render_height, 0, GL_RGB, GL_FLOAT, nullptr);

	reset_sample_number();
}

void AppWindow::change_resolution_scaling(float new_scaling)
{
	float new_render_width = std::floor(m_viewport_width * new_scaling);
	float new_render_height = std::floor(m_viewport_height * new_scaling);

	m_renderer.change_render_resolution(new_render_width, new_render_height);
	m_denoiser.resize_buffers(new_render_width, new_render_height);

	glActiveTexture(GL_TEXTURE0 + AppWindow::DISPLAY_TEXTURE_UNIT);
	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, new_render_width, new_render_height, 0, GL_RGB, GL_FLOAT, nullptr);
}

int AppWindow::get_width()
{
	return m_viewport_width;
}

int AppWindow::get_height()
{
	return m_viewport_height;
}

void AppWindow::setup_display_program()
{
	// Creating the shaders for displaying the path traced render

	const char* vertex_shader_text = "#version 330\n"
		"out vec2 vs_tex_coords;\n"

		"void main()\n"
		"{\n"
		"vec2 triangle_vertices[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, -1), vec2(1, 1), vec2(-1, 1));\n"
		"vec2 triangle_tex_coords[6] = vec2[6](vec2(0, 0), vec2(1, 0), vec2(0, 1), vec2(1, 0), vec2(1, 1), vec2(0, 1));\n"

		"gl_Position = vec4(triangle_vertices[gl_VertexID], 1, 1);\n"
		"vs_tex_coords = triangle_tex_coords[gl_VertexID];\n"
		"}";

	// Tone mapping fragment shader
	const char* fragment_shader_text = "#version 330\n"
		"uniform sampler2D u_texture;\n"
		"uniform int u_sample_number;\n"
		"uniform float u_gamma;\n"
		"uniform float u_exposure;\n"

		"in vec2 vs_tex_coords;\n"

		"void main()\n"
		"{\n"
		"vec4 hdr_color = texture(u_texture, vs_tex_coords) / float(u_sample_number + 1);\n"
		"vec4 tone_mapped = 1.0f - exp(-hdr_color * u_exposure);\n"
		"vec4 gamma_corrected = pow(tone_mapped, vec4(1.0f / u_gamma));\n"
		"gl_FragColor = vec4(gamma_corrected.rgb, 1.0f);\n"
		"}\n";

	GLuint m_vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(m_vertex_shader, 1, &vertex_shader_text, NULL);
	glCompileShader(m_vertex_shader);

	GLuint m_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(m_fragment_shader, 1, &fragment_shader_text, NULL);
	glCompileShader(m_fragment_shader);
	GLint isCompiled = 0;
	glGetShaderiv(m_fragment_shader, GL_COMPILE_STATUS, &isCompiled);
	if (isCompiled == GL_FALSE)
	{
		GLint maxLength = 0;
		glGetShaderiv(m_fragment_shader, GL_INFO_LOG_LENGTH, &maxLength);

		// The maxLength includes the NULL character
		std::vector<GLchar> errorLog(maxLength);
		glGetShaderInfoLog(m_fragment_shader, maxLength, &maxLength, &errorLog[0]);

		std::cout << errorLog.data() << std::endl;

		// Provide the infolog in whatever manor you deem best.
		// Exit with failure.
		glDeleteShader(m_fragment_shader); // Don't leak the shader.
		std::exit(-1);
	}


	m_display_program = glCreateProgram();
	glAttachShader(m_display_program, m_vertex_shader);
	glAttachShader(m_display_program, m_fragment_shader);
	glLinkProgram(m_display_program);

	// Creating the texture that will contain the path traced data to be displayed
	// by the shader.
	glGenTextures(1, &m_display_texture);
	glActiveTexture(GL_TEXTURE0 + AppWindow::DISPLAY_TEXTURE_UNIT);
	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_viewport_width, m_viewport_height, 0, GL_RGB, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glUseProgram(m_display_program);
	glUniform1i(glGetUniformLocation(m_display_program, "u_texture"), AppWindow::DISPLAY_TEXTURE_UNIT);
	glUniform1i(glGetUniformLocation(m_display_program, "u_sample_number"), 0);
	glUniform1f(glGetUniformLocation(m_display_program, "u_gamma"), m_application_settings.tone_mapping_gamma);
	glUniform1f(glGetUniformLocation(m_display_program, "u_exposure"), m_application_settings.tone_mapping_exposure);
}

void AppWindow::set_renderer_scene(Scene& scene)
{
	std::shared_ptr<Renderer::HIPRTScene> hiprt_scene = m_renderer.create_hiprt_scene_from_scene(scene);
	m_renderer.set_hiprt_scene(hiprt_scene);
}

void AppWindow::update_renderer_view_translation(float translation_x, float translation_y)
{
	if (translation_x == 0.f && translation_y == 0.0f)
		return;

	reset_sample_number();

	glm::vec3 translation = glm::vec3(translation_x / m_application_settings.view_translation_sldwn_x, translation_y / m_application_settings.view_translation_sldwn_y, 0.0f);
	m_renderer.translate_camera_view(translation);
}

void AppWindow::update_renderer_view_rotation(float offset_x, float offset_y)
{
	reset_sample_number();

	float rotation_x, rotation_y;

	rotation_x = offset_x / m_viewport_width * 2.0f * M_PI / m_application_settings.view_rotation_sldwn_x;
	rotation_y = offset_y / m_viewport_height * 2.0f * M_PI / m_application_settings.view_rotation_sldwn_y;

	m_renderer.rotate_camera_view(glm::vec3(rotation_x, rotation_y, 0.0f));
}

void AppWindow::update_renderer_view_zoom(float offset)
{
	if (offset == 0.0f)
		return;

	reset_sample_number();

	m_renderer.zoom_camera_view(offset / m_application_settings.view_zoom_sldwn);
}

void AppWindow::increment_sample_number()
{
	m_sample_number += m_renderer.get_render_settings().samples_per_frame;

	m_renderer.set_sample_number(m_sample_number);

	glUseProgram(m_display_program);
	glUniform1i(glGetUniformLocation(m_display_program, "u_sample_number"), m_sample_number);
}

void AppWindow::reset_sample_number()
{
	m_startRenderTime = std::chrono::high_resolution_clock::now();
	m_sample_number = 0;
	m_renderer.set_sample_number(0);

	glUseProgram(m_display_program);
	glUniform1i(glGetUniformLocation(m_display_program, "u_sample_number"), 0);
}

Renderer& AppWindow::get_renderer()
{
	return m_renderer;
}

std::pair<float, float> AppWindow::get_cursor_position()
{
	return m_cursor_position;
}

void AppWindow::set_cursor_position(std::pair<float, float> new_position)
{
	m_cursor_position = new_position;
}

void AppWindow::run()
{
	while (!glfwWindowShouldClose(m_window))
	{
		glfwPollEvents();
		glClear(GL_COLOR_BUFFER_BIT);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		if (!(m_application_settings.stop_render_at != 0 && m_sample_number + 1 > m_application_settings.stop_render_at))
		{
			m_renderer.render();
			increment_sample_number();
		}

		if (m_renderer.get_render_settings().enable_denoising)
			display(m_denoiser.denoise(m_renderer.m_render_width, m_renderer.m_render_height, m_renderer.get_orochi_framebuffer().download_pixels()));
		else
			display(m_renderer.get_orochi_framebuffer());
		display_imgui();

		glfwSwapBuffers(m_window);
	}

	quit();
}

// TODO display feedback for 5 seconds after dumping a screenshot to disk
void AppWindow::display_imgui()
{
	ImGuiIO& io = ImGui::GetIO();

	ImGui::Begin("Settings");

	auto now_time = std::chrono::high_resolution_clock::now();
	float render_time = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - m_startRenderTime).count();
	ImGui::Text("Render time: %.3fs", render_time / 1000.0f);
	ImGui::Text("%d samples | %.2f samples/s @ %dx%d", m_sample_number + 1, 1.0f / io.DeltaTime * m_renderer.get_render_settings().samples_per_frame, m_renderer.m_render_width, m_renderer.m_render_height);

	ImGui::Separator();

	if (ImGui::Button("Save render PNG (tonemapped)"))
	{
		std::vector<unsigned char> tonemaped_data = Utils::tonemap_hdr_image(m_renderer.get_orochi_framebuffer().download_pixels(), m_sample_number, m_application_settings.tone_mapping_gamma, m_application_settings.tone_mapping_exposure);

		stbi_flip_vertically_on_write(true);
		if (stbi_write_png("Render tonemapped.png", m_renderer.m_render_width, m_renderer.m_render_height, 3, tonemaped_data.data(), m_renderer.m_render_width * sizeof(unsigned char) * 3))
			std::cout << "Render written to \"Render tonemapped.png\"" << std::endl;
	}
	if (ImGui::Button("Save render HDR (non-tonemapped)"))
	{
		std::vector<float> hdr_data = m_renderer.get_orochi_framebuffer().download_pixels();

#pragma omp parallel for
		for (int i = 0; i < m_renderer.m_render_width * m_renderer.m_render_height; i++)
			hdr_data[i] = hdr_data[i] / (float)m_sample_number;

		stbi_flip_vertically_on_write(true);
		if (stbi_write_hdr("Render tonemapped.hdr", m_renderer.m_render_width, m_renderer.m_render_height, 3, reinterpret_cast<float*>(hdr_data.data())));
			std::cout << "Render written to \"Render tonemapped.hdr\"" << std::endl;
	}
	
	ImGui::Separator();

	if (ImGui::Combo("Render Kernel", &m_application_settings.selected_kernel, "Full Path Tracer\0Normals Visualisation\0\0"))
	{
		m_renderer.compile_trace_kernel(m_application_settings.kernel_files[m_application_settings.selected_kernel].c_str(), m_application_settings.kernel_functions[m_application_settings.selected_kernel].c_str());

		reset_sample_number();
	}

	if (m_renderer.get_render_settings().keep_same_resolution) // TODO Put this setting in application settings ?
		ImGui::BeginDisabled();
	float resolution_scaling_backup = m_renderer.get_render_settings().render_resolution_scale;
	if (ImGui::InputFloat("Render resolution scale", &m_renderer.get_render_settings().render_resolution_scale))
	{
		float& resolution_scale = m_renderer.get_render_settings().render_resolution_scale;
		if (resolution_scale <= 0)
			resolution_scale = resolution_scaling_backup;

		change_resolution_scaling(resolution_scale);
		reset_sample_number();
	}
	if (m_renderer.get_render_settings().keep_same_resolution)
		ImGui::EndDisabled();

	// TODO for the denoising with normals / albedo, add imgui buttons to display normals / albedo buffer
	// TODO possibility to denoise normals and albedo AOV buffer
	// TODO imgui renderer class to put all of this away in its own class

	// TODO don't reset frame number on resize when keep same render resolution is checked
	if (ImGui::Checkbox("Keep same render resolution", &m_renderer.get_render_settings().keep_same_resolution))
	{
		if (m_renderer.get_render_settings().keep_same_resolution)
		{
			// Remembering the width and height we need to target
			m_renderer.get_render_settings().target_width = m_renderer.m_render_width;
			m_renderer.get_render_settings().target_height = m_renderer.m_render_height;
		}
	}
	
	ImGui::Separator();

	if (ImGui::InputInt("Stop render at sample count", &m_application_settings.stop_render_at))
		m_application_settings.stop_render_at = std::max(m_application_settings.stop_render_at, 0);
	ImGui::InputInt("Samples per frame", &m_renderer.get_render_settings().samples_per_frame);
	if (ImGui::InputInt("Max bounces", &m_renderer.get_render_settings().nb_bounces))
	{
		// Clamping to 0 in case the user input a negative number of bounces	
		m_renderer.get_render_settings().nb_bounces = std::max(m_renderer.get_render_settings().nb_bounces, 0);

		reset_sample_number();
	}

	ImGui::Checkbox("Enable denoiser", &m_renderer.get_render_settings().enable_denoising);
	//ImGui::Checkbox("Denoise every frame", &m_renderer.get_render_settings().denoise_every_frame);
	ImGui::DragFloat("Denoise strength", &m_renderer.get_render_settings().denoising_strength, 1.0f, 0.0f, 1.0f);

	ImGui::Separator();



	if (ImGui::InputFloat("Gamma", &m_application_settings.tone_mapping_gamma))
		glUniform1f(glGetUniformLocation(m_display_program, "u_gamma"), m_application_settings.tone_mapping_gamma);
	if (ImGui::InputFloat("Exposure", &m_application_settings.tone_mapping_exposure))
		glUniform1f(glGetUniformLocation(m_display_program, "u_exposure"), m_application_settings.tone_mapping_exposure);

	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void AppWindow::quit()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glDeleteTextures(1, &m_display_texture);
	glDeleteProgram(m_display_program);

	std::exit(0);
}
