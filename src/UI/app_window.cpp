#include "UI/app_window.h"

#include <functional>
#include <iostream>
#include <Scene/scene_parser.h>

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

	// We've stored a pointer to the AppWindow in the "WindowUserPointer" of glfw
	reinterpret_cast<AppWindow*>(glfwGetWindowUserPointer(window))->resize_frame(width, height);
}

void glfw_mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos)
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
		else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
			app_window->update_renderer_view_rotation(-difference.first, -difference.second);
	}

	// Updating the position
	app_window->set_cursor_position(std::make_pair(xposf, yposf));
}

void glfw_mouse_scroll_callback(GLFWwindow * window, double xoffset, double yoffset)
{
	AppWindow* app_window = reinterpret_cast<AppWindow*>(glfwGetWindowUserPointer(window));

	app_window->update_renderer_view_zoom(static_cast<float>(-yoffset));
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

AppWindow::AppWindow(int width, int height) : m_width(width), m_height(height)
{
	if (!glfwInit())
		wait_and_exit("Could not initialize GLFW...");

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
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
	m_renderer.compile_trace_kernel("Kernels/normals_kernel.h", "NormalsKernel");
	m_renderer.resize_frame(width, height);
}

AppWindow::~AppWindow()
{
	glfwDestroyWindow(m_window);
	glfwTerminate();
}

void AppWindow::resize_frame(int pixels_width, int pixels_height)
{
	glViewport(0, 0, pixels_width, pixels_height);

	m_width = pixels_width;
	m_height = pixels_height;
	m_renderer.resize_frame(pixels_width, pixels_height);

	// Recreating the OpenGL display texture
	glActiveTexture(GL_TEXTURE0 + AppWindow::DISPLAY_TEXTURE_UNIT);
	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, nullptr);
}

int AppWindow::get_width()
{
	return m_width;
}

int AppWindow::get_height()
{
	return m_height;
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

	const char* fragment_shader_text = "#version 330\n"
		"uniform sampler2D u_texture;\n"

		"in vec2 vs_tex_coords;\n"

		"void main()\n"
		"{\n"
		"gl_FragColor = texture(u_texture, vs_tex_coords);\n"
		"}\n";

	GLuint m_vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(m_vertex_shader, 1, &vertex_shader_text, NULL);
	glCompileShader(m_vertex_shader);

	GLuint m_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(m_fragment_shader, 1, &fragment_shader_text, NULL);
	glCompileShader(m_fragment_shader);

	m_display_program = glCreateProgram();
	glAttachShader(m_display_program, m_vertex_shader);
	glAttachShader(m_display_program, m_fragment_shader);
	glLinkProgram(m_display_program);

	// Creating the texture that will contain the path traced data to be displayed
	// by the shader.
	glGenTextures(1, &m_display_texture);
	glActiveTexture(GL_TEXTURE0 + AppWindow::DISPLAY_TEXTURE_UNIT);
	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glUseProgram(m_display_program);
	glUniform1i(glGetUniformLocation(m_display_program, "u_texture"), AppWindow::DISPLAY_TEXTURE_UNIT);
}

void AppWindow::set_renderer_scene(Scene& scene)
{
	Renderer::HIPRTScene hiprt_scene = m_renderer.create_hiprt_scene_from_scene(scene);
	m_renderer.set_hiprt_scene(hiprt_scene);
}

void AppWindow::update_renderer_view_translation(float translation_x, float translation_y)
{
	glm::vec3 translation = glm::vec3(translation_x / m_application_settings.view_translation_sldwn_x, translation_y / m_application_settings.view_translation_sldwn_y, 0.0f);
	m_renderer.translate_camera_view(translation);
}

void AppWindow::update_renderer_view_rotation(float offset_x, float offset_y)
{
	float rotation_x, rotation_y;

	rotation_x = offset_x / m_width * 2.0f * M_PI / m_application_settings.view_rotation_sldwn_x;
	rotation_y = offset_y / m_height * 2.0f * M_PI / m_application_settings.view_rotation_sldwn_y;

	m_renderer.rotate_camera_view(glm::vec3(rotation_x, rotation_y, 0.0f));
}

void AppWindow::update_renderer_view_zoom(float offset)
{
	m_renderer.zoom_camera_view(offset / m_application_settings.view_zoom_sldwn);
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

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		m_renderer.render();

		display(m_renderer.get_orochi_framebuffer());
		display_imgui();

		glfwSwapBuffers(m_window);
	}

	quit();
}

void AppWindow::display(const std::vector<Color>& image_data)
{
	glUseProgram(m_display_program);

	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, image_data.data());

	glDrawArrays(GL_TRIANGLES, 0, 6);
}

void AppWindow::display(OrochiBuffer<float>& orochi_buffer)
{
	glUseProgram(m_display_program);

	// TODO
	// This is very sub optimal and should absolutely be replaced by a 
	// buffer that is shared between hiprt and opengl
	// Unfortunately, this is unavailable in Orochi so we would have
	// to switch to HIP (or CUDA but it doesn't support AMD) to get access
	// to the hipGraphicsMapResources() functions family
	std::vector<float> pixels_data = orochi_buffer.download_pixels();

	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, pixels_data.data());

	glDrawArrays(GL_TRIANGLES, 0, 6);
}

void AppWindow::display_imgui()
{
	ImGui::ShowDemoWindow();

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

