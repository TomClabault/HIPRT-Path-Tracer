#include "UI/app_window.h"

#include <functional>
#include <iostream>

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

	static_cast<AppWindow*>(glfwGetWindowUserPointer(window))->resize(width, height);
}

AppWindow::AppWindow(int width, int height) : m_width(width), m_height(height), m_renderer(width, height)
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

	setup_display_program();
}

AppWindow::~AppWindow()
{
	glfwDestroyWindow(m_window);
	glfwTerminate();
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

void AppWindow::resize(int pixels_width, int pixels_height)
{
	glViewport(0, 0, pixels_width, pixels_height);
	m_renderer.resize(pixels_width, pixels_height);
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

void AppWindow::run()
{
	while (!glfwWindowShouldClose(m_window))
	{
		m_renderer.render();
		display_cpu_data(m_renderer.get_cpu_data());

		glfwPollEvents();
		glfwSwapBuffers(m_window);
	}

	quit();
}

void AppWindow::display_cpu_data(const std::vector<Color>& image_data)
{
	glUseProgram(m_display_program);

	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, image_data.data());

	glDrawArrays(GL_TRIANGLES, 0, 6);
}

void AppWindow::quit()
{
	std::exit(0);
}
