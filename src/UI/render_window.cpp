/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/render_window.h"

#include <functional>
#include <iostream>
#include "Scene/scene_parser.h"
#include "Utils/utils.h"
#include "Utils/opengl_utils.h"

#include "stb_image_write.h"

// test performance when reducing number of triangles of the pbrt dragon

// TODO bugs
// - fix screenshot writer compute shader since OpenGL refactor
// - anisotropic rotation brightness buggued ?
// - Why is the rough dragon having black fringes even with normal flipping ?
// - normals AOV not converging correctly ?
//		- for the denoiser normals convergence issue, is it an error at the end of the Path Tracer kernel where we're accumulating ? Should we have
//		render_data.aux_buffers.denoiser_albedo[index] * render_data.render_settings.sample_number 
//		instead of 
//		render_data.aux_buffers.denoiser_albedo[index] * render_data.render_settings.frame_number
//		?
// - denoiser not accounting for tranmission correctly since Disney 



// TODO Code Organization:
// - Rename class files in camel case
// - Destroy buffers when disabling adaptative sampling to save VRAM
// - Can we have access to HoL when calling disney_metallic_fresnel to avoid passing the two vectors and recomputing the dot product in the return statement ?
// - rename HIPRT kernel files without the hiprt prefix
// - DO THE DISNEY SHADING IN SHADING SPACE. WHAT THE H IS THIS CODE BUILDING ONB IN EVERY FUNCTION HUH?
// - reorganize methods order in RenderWindow
// - separate (still work to do) path tracer kernel functions in header files
// - do not duplicate render functions. Make a common h file that uses the float3 type (cosine_weighted_direction_around_normal, hiprt_lambertian.h:7 for example)
// - imgui controller to put all the imgui code in one class
// - put mouse / keyboard code in an interactor
//		- Have the is_interacting boolean in this interactor class and poll it from the main loop to check whether we need to render the frame at a lower resolution or not
// - check for level of abstractions in functions



// TODO Features:
// - image comparator slider
// - auto adaptative sample per frame with adaptative sampling to keep GPU busy
// - Maybe look at better Disney sampling (luminance?)
// - Imgui panel with a lot of performance metrics
// - thin materials
// - Look at what Orochi & HIPCC can do in terms of displaying registers used / options to specify shared stack size / block size (-DBLOCK_SIZE, -DSHARED_STACK_SIZE)
// - Have the UI run at its own framerate to avoid having the UI come to a crawl when the path tracing is expensive
// - Denoiser blend to allow blending the original noisy image and the perfect denoised result by a given factor
// - When modifying the emission of a material with the material editor, it should be reflected in the scene and allow the direct sampling of the geometry
// - Color fallof (change of material base base_color based on the angle with the view direction and the normal
// - Ray binning for performance
// - Starting rays further away from the camera for performance
// - Visualizing ray depth (only 1 frame otherwise it would flicker a lot [or choose the option to have it flicker] )
// - Visualizing pixel time with the clock() instruction. Pixel heatmap:
//		- https://developer.nvidia.com/blog/profiling-dxr-shaders-with-timer-instrumentation/
//		- https://github.com/libigl/libigl/issues/1388
//		- https://github.com/libigl/libigl/issues/1534
// - Visualizing russian roulette depth termination
// - Add tooltips when hovering over a parameter in the UI
// - Statistics on russian roulette efficiency
// - Being able to enable / disable env map importance sampling
// - Being able to enable / disable MIS
// - Better ray origin offset to avoid self intersections
// - Realistic Camera Model
// - Focus blur
// - Textures for each parameter of the Disney BSDF
// - Bump mapping
// - Flakes BRDF (maybe look at OSPRay implementation for a reference ?)
// - ImGuizmo
// - Paths roughness regularization
// - choose disney diffuse model (disney, lambertian, oren nayar)
// - enable lower resolution on mouse scroll for like ~10 frames
// - display feedback for 3 seconds after dumping a screenshot to disk
// - choose denoiser quality in imgui
// - try async buffer copy for the denoiser (maybe run a kernel to generate normals and another to generate albedo buffer before the path tracing kernel to be able to async copy while the path tracing kernel is running?)
// - enable denoising with all combinations of beauty/normal/albedo via imgui
// - show denoised normals / denoised albedo when ticking the Show Normals / Show albedo checkboxes in Imgui to visualize the albedo/normals used by the denoiser
// - uniform float3 type to use everywhere instead of Vector and hiprtFloat3
// - cutout filters
// - write scene details to imgui (nb vertices, triangles, ...)
// - check perf of aiPostProcessSteps::aiProcess_ImproveCacheLocality
// - ImGui to choose the BVH flags at runtime and be able to compare the performance
// - ImGui widgets for SBVH / LBVH
// - light sampling: go through transparent surfaces instead of considering them opaque (?)
// - BVH compaction + imgui checkbox
// - shader cache (write our own or wait for HIPRT to fix it?)
// - indirect / direct lighting clamping
// - env map support
// - choose env map at runtime imgui
// - env map rotation imgui
// - choose scene file at runtime imgui
// - lock camera checkbox to avoid messing up when big render in progress
// - choose render resolution in imgui
// - choose viewport resolution in imgui
// - compute shader for tone mapping images ? unless transfering memory to open gl is too expensive
// - use defines insead of IFs in the kernel code and recompile kernel everytime (for some options at least)
// - stuff to multithread when loading everything ? (scene, BVH, textures, ...)
// - PBRT v3 scene parser
// - Wavefront path tracing
// - Manifold Next Event Estimation (for refractive caustics) https://jo.dreggn.org/home/2015_mnee.pdf
// - Efficiency Aware Russian roulette and splitting
// - ReSTIR PT

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
		// We've stored a pointer to the RenderWindow in the "WindowUserPointer" of glfw
		reinterpret_cast<RenderWindow*>(glfwGetWindowUserPointer(window))->resize_frame(width, height);
}

static bool interacting_left_button = false, interacting_right_button = false;
void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	bool imgui_wants_mouse = ImGui::GetIO().WantCaptureMouse;

	switch (button)
	{
	case GLFW_MOUSE_BUTTON_LEFT:
		interacting_left_button = (action == GLFW_PRESS) && !imgui_wants_mouse;

		break;

	case GLFW_MOUSE_BUTTON_RIGHT:
		interacting_right_button = (action == GLFW_PRESS) && !imgui_wants_mouse;

		break;
	}
	
	reinterpret_cast<RenderWindow*>(glfwGetWindowUserPointer(window))->set_interacting(interacting_left_button || interacting_right_button);
}

void glfw_mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos)
{
	ImGuiIO& io = ImGui::GetIO();
	if (!io.WantCaptureMouse)
	{
		RenderWindow* render_window = reinterpret_cast<RenderWindow*>(glfwGetWindowUserPointer(window));

		float xposf = static_cast<float>(xpos);
		float yposf = static_cast<float>(ypos);

		std::pair<float, float> old_position = render_window->get_cursor_position();
		if (old_position.first == -1 && old_position.second == -1)
			;
		// If this is the first position of the cursor, nothing to do
		else
		{
			// Computing the difference in movement
			std::pair<float, float> difference = std::make_pair(xposf - old_position.first, yposf - old_position.second);

			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
				render_window->update_renderer_view_translation(-difference.first, difference.second);

			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
				render_window->update_renderer_view_rotation(-difference.first, -difference.second);
		}

		// Updating the position
		render_window->set_cursor_position(std::make_pair(xposf, yposf));
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
		translation.first -= 36.0f;
	if (space_pressed)
		translation.second += 36.0f;
	if (lshift_pressed)
		translation.second -= 36.0f;


	RenderWindow* render_window = reinterpret_cast<RenderWindow*>(glfwGetWindowUserPointer(window));
	render_window->update_renderer_view_translation(-translation.first, translation.second);
	render_window->update_renderer_view_zoom(-zoom);
}

void glfw_mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	ImGuiIO& io = ImGui::GetIO();
	if (!io.WantCaptureMouse)
	{
		RenderWindow* render_window = reinterpret_cast<RenderWindow*>(glfwGetWindowUserPointer(window));

		render_window->update_renderer_view_zoom(static_cast<float>(-yoffset));
	}
}

// Implementation from https://learnopengl.com/In-Practice/Debugging
void APIENTRY RenderWindow::gl_debug_output_callback(GLenum source,
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

	// The following breaks into the debugger to help pinpoint what OpenGL
	// call errored
#if defined( _WIN32 )
	__debugbreak();
#elif defined( GNUC )
	raise(SIGTRAP);
#else
	;
#endif
}

RenderWindow::RenderWindow(int width, int height) : m_viewport_width(width), m_viewport_height(height), m_render_settings(m_renderer.get_render_settings())
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
	// Setting a pointer to this instance of RenderWindow inside the m_window GLFWwindow so that
	// we can retrieve a pointer to this instance of RenderWindow in the callback functions
	// such as the window_resized_callback function for example
	glfwSetWindowUserPointer(m_window, this);
	glfwSetWindowSizeCallback(m_window, glfw_window_resized_callback);
	glfwSetCursorPosCallback(m_window, glfw_mouse_cursor_callback);
	glfwSetMouseButtonCallback(m_window, glfw_mouse_button_callback);
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
		glDebugMessageCallback(RenderWindow::gl_debug_output_callback, nullptr);
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

	create_display_programs();
	m_renderer.init_ctx(0);
	m_renderer.compile_trace_kernel(m_application_settings.kernel_files[m_application_settings.selected_kernel].c_str(),
		m_application_settings.kernel_functions[m_application_settings.selected_kernel].c_str());
	m_renderer.change_render_resolution(width, height);

	// TODO fix denoiser buffer since openGL interop
	/*m_denoiser.set_buffers(m_renderer.get_color_framebuffer().get_device_pointer(),
		m_renderer.get_denoiser_normals_buffer().get_device_pointer(), m_renderer.get_denoiser_albedo_buffer().get_device_pointer(),
		width, height);*/

	m_image_writer.set_renderer(&m_renderer);
	m_image_writer.set_render_window(this);
}

RenderWindow::~RenderWindow()
{
	glfwDestroyWindow(m_window);
	glfwTerminate();
}

void RenderWindow::resize_frame(int pixels_width, int pixels_height)
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

	// Taking resolution scaling into account
	float& resolution_scale = m_application_settings.render_resolution_scale;
	if (m_application_settings.keep_same_resolution)
		resolution_scale = m_application_settings.target_width / (float)pixels_width; // TODO what about the height changing ?

	int new_render_width = std::floor(pixels_width * resolution_scale);
	int new_render_height = std::floor(pixels_height * resolution_scale);
	m_renderer.change_render_resolution(new_render_width, new_render_height);

	// TODO fix denoiser buffer since openGL interop
	/*m_denoiser.set_buffers(m_renderer.get_color_framebuffer().get_device_pointer(),
		m_renderer.get_denoiser_normals_buffer().get_device_pointer(), m_renderer.get_denoiser_albedo_buffer().get_device_pointer(), 
		new_render_width, new_render_height);*/

	recreate_display_texture(m_display_texture_type, new_render_width, new_render_height);

	m_render_dirty = true;
}

void RenderWindow::change_resolution_scaling(float new_scaling)
{
	float new_render_width = std::floor(m_viewport_width * new_scaling);
	float new_render_height = std::floor(m_viewport_height * new_scaling);

	m_renderer.change_render_resolution(new_render_width, new_render_height);
	// TODO fix denoiser buffer since openGL interop
	/*m_denoiser.set_buffers(m_renderer.get_color_framebuffer().get_device_pointer(),
		m_renderer.get_denoiser_normals_buffer().get_device_pointer(), m_renderer.get_denoiser_albedo_buffer().get_device_pointer(),
		new_render_width, new_render_height);*/

	glActiveTexture(GL_TEXTURE0 + RenderWindow::DISPLAY_TEXTURE_UNIT);
	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, new_render_width, new_render_height, 0, GL_RGB, GL_FLOAT, nullptr);
}

int RenderWindow::get_width()
{
	return m_viewport_width;
}

int RenderWindow::get_height()
{
	return m_viewport_height;
}

void RenderWindow::set_interacting(bool is_interacting)
{
	// The user just released the camera and we were rendering at low resolution
	if (!is_interacting && m_render_settings.render_low_resolution)
		m_render_dirty = true;

	m_render_settings.render_low_resolution = is_interacting;
}

ApplicationSettings& RenderWindow::get_application_settings()
{
	return m_application_settings;
}

const ApplicationSettings& RenderWindow::get_application_settings() const
{
	return m_application_settings;
}

void RenderWindow::create_display_programs()
{
	// Creating the texture that will contain the path traced data to be displayed
	// by the shader.
	glGenTextures(1, &m_display_texture);
	glActiveTexture(GL_TEXTURE0 + RenderWindow::DISPLAY_TEXTURE_UNIT);
	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	// The texture is initially of the size of the viewport because there is no resolution scaling involved
	// at startup
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_viewport_width, m_viewport_height, 0, GL_RGB, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// This empty VAO is necessary on NVIDIA drivers even though
	// we're hardcoding our full screen quad in the vertex shader
	glCreateVertexArrays(1, &m_vao);

	OpenGLShader fullscreen_quad_vertex_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/fullscreen_quad.vert", OpenGLShader::VERTEX_SHADER);
	OpenGLShader default_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/default_display.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader normal_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/normal_display.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader albedo_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/albedo_display.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader adaptative_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/heatmap_int.frag", OpenGLShader::FRAGMENT_SHADER);

	m_default_display_program.attach(fullscreen_quad_vertex_shader);
	m_default_display_program.attach(default_display_fragment_shader);
	m_default_display_program.link();

	m_normal_display_program.attach(fullscreen_quad_vertex_shader);
	m_normal_display_program.attach(normal_display_fragment_shader);
	m_normal_display_program.link();

	m_albedo_display_program.attach(fullscreen_quad_vertex_shader);
	m_albedo_display_program.attach(albedo_display_fragment_shader);
	m_albedo_display_program.link();

	m_adaptative_sampling_display_program.attach(fullscreen_quad_vertex_shader);
	m_adaptative_sampling_display_program.attach(adaptative_display_fragment_shader);
	m_adaptative_sampling_display_program.link();

	select_display_program(m_application_settings.display_view);
}

void RenderWindow::select_display_program(DisplayView display_view)
{
	switch (display_view)
	{
	case DisplayView::DEFAULT:
		m_active_display_program = m_default_display_program;
		break;

	case DisplayView::DISPLAY_NORMALS:
	case DisplayView::DISPLAY_DENOISED_NORMALS:
		m_active_display_program = m_normal_display_program;
		break;

	case DisplayView::DISPLAY_ALBEDO:
	case DisplayView::DISPLAY_DENOISED_ALBEDO:
		m_active_display_program = m_albedo_display_program;
		break;

	case DisplayView::ADAPTIVE_SAMPLING_MAP:
		m_active_display_program = m_adaptative_sampling_display_program;
		break;


	default:

		break;
	}

	recreate_display_texture(display_view);
}

void RenderWindow::recreate_display_texture(DisplayView display_view)
{
	DisplayTextureType texture_type_needed;

	switch (display_view)
	{
	case DisplayView::DISPLAY_NORMALS:
	case DisplayView::DISPLAY_DENOISED_NORMALS:
	case DisplayView::DISPLAY_ALBEDO:
	case DisplayView::DISPLAY_DENOISED_ALBEDO:
		texture_type_needed = DisplayTextureType::FLOAT3;
		break;

	case DisplayView::ADAPTIVE_SAMPLING_MAP:
		texture_type_needed = DisplayTextureType::INT;
		break;

	case DisplayView::DEFAULT:
	default:
		texture_type_needed = DisplayTextureType::FLOAT3;
		break;
	}

	if (m_display_texture_type != texture_type_needed)
		recreate_display_texture(texture_type_needed, m_viewport_width, m_viewport_height);
}

void RenderWindow::recreate_display_texture(DisplayTextureType texture_type, int width, int height)
{
	GLint internal_format = texture_type.get_gl_internal_format();
	GLenum format = texture_type.get_gl_format();
	GLenum type = texture_type.get_gl_type();

	glActiveTexture(GL_TEXTURE0 + RenderWindow::DISPLAY_TEXTURE_UNIT);
	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glTexImage2D(GL_TEXTURE_2D, 0, internal_format, m_renderer.m_render_width, m_renderer.m_render_height, 0, format, type, nullptr);
	
	m_display_texture_type = texture_type;
}

void RenderWindow::upload_data_to_display_texture(const void* data, GLenum format, GLenum type)
{
	glActiveTexture(GL_TEXTURE0 + RenderWindow::DISPLAY_TEXTURE_UNIT);
	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_renderer.m_render_width, m_renderer.m_render_height, format, type, data);
}

void RenderWindow::update_active_program_uniforms()
{
	m_active_display_program.use();

	switch (m_application_settings.display_view)
	{
	case DisplayView::DEFAULT:
		m_active_display_program.set_uniform("u_texture", RenderWindow::DISPLAY_TEXTURE_UNIT);
		m_active_display_program.set_uniform("u_sample_number", m_render_settings.sample_number);
		m_active_display_program.set_uniform("u_do_tonemapping", m_application_settings.do_tonemapping);
		m_active_display_program.set_uniform("u_gamma", m_application_settings.tone_mapping_gamma);
		m_active_display_program.set_uniform("u_exposure", m_application_settings.tone_mapping_exposure);

		break;

	case DisplayView::DISPLAY_ALBEDO:
	case DisplayView::DISPLAY_DENOISED_ALBEDO:
		m_active_display_program.set_uniform("u_texture", RenderWindow::DISPLAY_TEXTURE_UNIT);
		m_active_display_program.set_uniform("u_sample_number", m_render_settings.sample_number);

		break;

	case DisplayView::DISPLAY_NORMALS:
	case DisplayView::DISPLAY_DENOISED_NORMALS:
		m_active_display_program.set_uniform("u_texture", RenderWindow::DISPLAY_TEXTURE_UNIT);
		m_active_display_program.set_uniform("u_sample_number", m_render_settings.sample_number);
		m_active_display_program.set_uniform("u_do_tonemapping", m_application_settings.do_tonemapping);
		m_active_display_program.set_uniform("u_gamma", m_application_settings.tone_mapping_gamma);
		m_active_display_program.set_uniform("u_exposure", m_application_settings.tone_mapping_exposure);

		break;

	case DisplayView::ADAPTIVE_SAMPLING_MAP:
		std::vector<ColorRGB> color_stops = { ColorRGB(0.0f, 0.0f, 1.0f), ColorRGB(0.0f, 1.0f, 0.0f), ColorRGB(1.0f, 0.0f, 0.0f) };

		m_active_display_program.set_uniform("u_texture", RenderWindow::DISPLAY_TEXTURE_UNIT);
		m_active_display_program.set_uniform("u_color_stops", 3, (float*)color_stops.data());
		m_active_display_program.set_uniform("u_nb_stops", 2);
		m_active_display_program.set_uniform("u_min_val", (float)m_render_settings.adaptive_sampling_min_samples);
		m_active_display_program.set_uniform("u_max_val", (float)m_render_settings.sample_number);

		break;
	}
}

void RenderWindow::update_renderer_view_translation(float translation_x, float translation_y)
{
	if (translation_x == 0.f && translation_y == 0.0f)
		return;

	m_render_dirty = true;

	glm::vec3 translation = glm::vec3(translation_x / m_application_settings.view_translation_sldwn_x, translation_y / m_application_settings.view_translation_sldwn_y, 0.0f);
	m_renderer.translate_camera_view(translation);
}

void RenderWindow::update_renderer_view_rotation(float offset_x, float offset_y)
{
	m_render_dirty = true;

	float rotation_x, rotation_y;

	rotation_x = offset_x / m_viewport_width * 2.0f * M_PI / m_application_settings.view_rotation_sldwn_x;
	rotation_y = offset_y / m_viewport_height * 2.0f * M_PI / m_application_settings.view_rotation_sldwn_y;

	m_renderer.rotate_camera_view(glm::vec3(rotation_x, rotation_y, 0.0f));
}

void RenderWindow::update_renderer_view_zoom(float offset)
{
	if (offset == 0.0f)
		return;

	m_render_dirty = true;

	m_renderer.zoom_camera_view(offset / m_application_settings.view_zoom_sldwn);
}

void RenderWindow::increment_sample_number()
{
	if (m_render_settings.render_low_resolution)
		m_render_settings.sample_number++; // Only doing 1 SPP when moving the cameras
	else
		m_render_settings.sample_number += m_render_settings.samples_per_frame;

	m_active_display_program.use();
	m_active_display_program.set_uniform("u_sample_number", m_render_settings.sample_number);
}

void RenderWindow::reset_render()
{
	m_startRenderTime = std::chrono::high_resolution_clock::now();
	m_renderer.set_sample_number(0);
	m_render_settings.frame_number = 0;

	m_active_display_program.use();
	m_active_display_program.set_uniform("u_sample_number", 0);

	m_render_dirty = false;
}

Renderer& RenderWindow::get_renderer()
{
	return m_renderer;
}

std::pair<float, float> RenderWindow::get_cursor_position()
{
	return m_cursor_position;
}

void RenderWindow::set_cursor_position(std::pair<float, float> new_position)
{
	m_cursor_position = new_position;
}

void RenderWindow::run()
{
	while (!glfwWindowShouldClose(m_window))
	{
		if (m_render_dirty)
			reset_render();

		glfwPollEvents();
		glClear(GL_COLOR_BUFFER_BIT);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		bool image_rendered = render();

		if (m_application_settings.enable_denoising)
		{
			if (m_application_settings.denoise_at_target_sample_count)
			{
				if (m_render_settings.sample_number == m_application_settings.max_sample_count)
				{
					m_denoiser.denoise();
					display(m_denoiser.get_denoised_data_pointer());
				}
				else
					display(m_renderer.get_color_framebuffer());
			}
			else
			{
				if ((m_render_settings.sample_number % m_application_settings.denoiser_sample_skip) == 0)
				{
					m_denoiser.denoise();
					m_application_settings.last_denoised_sample_count = m_render_settings.sample_number;
				}
			
				display(m_denoiser.get_denoised_data_pointer());
			}
		}
		else
		{
			// TODO
			// NOTE that we're not using any OpenGL interop here yet and we're going through the
			// CPU to display the various buffers because Orochi doesn't support OpenGL Interop for
			// NVIDIA yet and we don't want to have a lot of dirty expection cases. We'll just wait
			// for OpenGL interop to be supported on NVIDIA
			switch (m_application_settings.display_view)
			{
			case DisplayView::DISPLAY_NORMALS:
				display(m_renderer.get_denoiser_normals_buffer().download_data().data());
				break;

			case DisplayView::DISPLAY_DENOISED_NORMALS:
				m_denoiser.denoise_normals();
				display(m_denoiser.get_denoised_normals_pointer());
				break;

			case DisplayView::DISPLAY_ALBEDO:
				display(m_renderer.get_denoiser_albedo_buffer().download_data().data());
				break;

			case DisplayView::DISPLAY_DENOISED_ALBEDO:
				m_denoiser.denoise_albedo();
				display(m_denoiser.get_denoised_albedo_pointer());
				break;

			case DisplayView::ADAPTIVE_SAMPLING_MAP:
				display(m_renderer.get_pixels_sample_count_buffer().download_data().data());
				break;

			case DisplayView::ADAPTIVE_SAMPLING_ACTIVE_PIXELS:
				display(m_renderer.get_debug_pixel_active_buffer().download_data().data());

			case DisplayView::DEFAULT:
			default:
				display(m_renderer.get_color_framebuffer());
				break;
			}
		}
		
		draw_imgui();

		glfwSwapBuffers(m_window);
	}

	quit();
}

bool RenderWindow::render()
{
	if (!(m_application_settings.max_sample_count != 0 && m_render_settings.sample_number + 1 > m_application_settings.max_sample_count))
	{
		m_renderer.render();
		increment_sample_number();
		m_render_settings.frame_number++;

		return true;
	}

	return false;
}

void RenderWindow::display(const void* data)
{
	GLenum format = m_display_texture_type.get_gl_format();
	GLenum type = m_display_texture_type.get_gl_type();
	upload_data_to_display_texture(data, format, type);

	update_active_program_uniforms();

	// Binding an empty VAO here (empty because we're hardcoding our full-screen quad vertices
	// in our vertex shader) because this is required on NVIDIA drivers
	glBindVertexArray(m_vao);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

void RenderWindow::show_render_settings_panel()
{
	if (!ImGui::CollapsingHeader("Render Settings"))
		return;
	ImGui::TreePush("Render settings tree");

	if (ImGui::Combo("Render Kernel", &m_application_settings.selected_kernel, "Full Path Tracer\0Normals Visualisation\0\0"))
	{
		m_renderer.compile_trace_kernel(m_application_settings.kernel_files[m_application_settings.selected_kernel].c_str(), m_application_settings.kernel_functions[m_application_settings.selected_kernel].c_str());
		m_render_dirty = true;
	}

	const char* items[] = { "Default", "Denoiser - Normals", "Denoiser - Denoised normals", "Denoiser - Albedo", "Denoiser - Denoised albedo", "Adaptative sampling map"};
	if (ImGui::Combo("Display View", (int*)(&m_application_settings.display_view), items, IM_ARRAYSIZE(items)))
		select_display_program(m_application_settings.display_view);

	if (m_application_settings.keep_same_resolution)
		ImGui::BeginDisabled();
	float resolution_scaling_backup = m_application_settings.render_resolution_scale;
	if (ImGui::InputFloat("Resolution scale", &m_application_settings.render_resolution_scale))
	{
		float& resolution_scale = m_application_settings.render_resolution_scale;
		if (resolution_scale <= 0)
			resolution_scale = resolution_scaling_backup;

		change_resolution_scaling(resolution_scale);
		m_render_dirty = true;
	}
	if (m_application_settings.keep_same_resolution)
		ImGui::EndDisabled();

	if (ImGui::Checkbox("Keep same render resolution", &m_application_settings.keep_same_resolution))
	{
		if (m_application_settings.keep_same_resolution)
		{
			// Remembering the width and height we need to target
			m_application_settings.target_width = m_renderer.m_render_width;
			m_application_settings.target_height = m_renderer.m_render_height;
		}
	}

	ImGui::Separator();

	if (ImGui::InputInt("Stop render at sample count", &m_application_settings.max_sample_count))
		m_application_settings.max_sample_count = std::max(m_application_settings.max_sample_count, 0);
	ImGui::InputInt("Samples per frame", &m_render_settings.samples_per_frame);
	if (ImGui::InputInt("Max bounces", &m_render_settings.nb_bounces))
	{
		// Clamping to 0 in case the user input a negative number of bounces	
		m_render_settings.nb_bounces = std::max(m_render_settings.nb_bounces, 0); 
		m_render_dirty = true;
	}

	if (ImGui::CollapsingHeader("Adaptive sampling"))
	{
		ImGui::TreePush("Adaptive sampling tree");

		m_render_dirty |= ImGui::Checkbox("Enable adaptive sampling", &m_render_settings.enable_adaptive_sampling);
		m_render_dirty |= ImGui::InputInt("Adaptive sampling minimum samples", &m_render_settings.adaptive_sampling_min_samples);
		if (ImGui::InputFloat("Adaptive sampling noise threshold", &m_render_settings.adaptive_sampling_noise_threshold))
		{
			m_render_settings.adaptive_sampling_noise_threshold = std::max(0.0f, m_render_settings.adaptive_sampling_noise_threshold);
			m_render_dirty = true;
		}

		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Lighting"))
	{
		ImGui::TreePush("Lighting tree");

		m_render_dirty |= ImGui::Checkbox("Use ambient light", &m_renderer.get_world_settings().use_ambient_light);
		m_render_dirty |= ImGui::ColorEdit3("Ambient light color", (float*)&m_renderer.get_world_settings().ambient_light_color, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);

		ImGui::TreePop();
	}

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void RenderWindow::show_objects_panel()
{
	if (!ImGui::CollapsingHeader("Objects"))
		return;
	ImGui::TreePush("Objects tree");

	std::vector<RendererMaterial> materials = m_renderer.get_materials();

	int material_modfied_id = -1;
	int material_counter = 0;
	bool some_material_changed = false;

	ImGui::PushItemWidth(384);
	for (RendererMaterial& material : materials)
	{
		// Multiple ImGui widgets cannot have the same label
		// If all our materials use the same "Base color", "Subsurface", ... labels for
		// the slider, there is a chance that the slider will be linked together
		// and that multiple materials will be modified when only touching one slider
		// One solution to that is to avoid using the same label for multiple sliders
		// by naming them "material 1 Base color", "material 2 Base color" for example
		// This is however not very practical so ImGui provides us with the PushID function which
		// essentially differentiate the widgets without having to change the labels 
		ImGui::PushID(material_counter);

		some_material_changed |= ImGui::ColorEdit3("Base color", (float*)&material.base_color);
		some_material_changed |= ImGui::SliderFloat("Subsurface", &material.subsurface, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Metallic", &material.metallic, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Specular", &material.specular, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Specular tint strength", &material.specular_tint, 0.0f, 1.0f);
		some_material_changed |= ImGui::ColorEdit3("Specular color", (float*)&material.specular_color);
		some_material_changed |= ImGui::SliderFloat("Roughness", &material.roughness, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Anisotropic", &material.anisotropic, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Anisotropic rotation", &material.anisotropic_rotation, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Sheen", &material.sheen, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Sheen tint strength", &material.sheen_tint, 0.0f, 1.0f);
		some_material_changed |= ImGui::ColorEdit3("Sheen color", (float*)&material.sheen_color);
		some_material_changed |= ImGui::SliderFloat("Clearcoat", &material.clearcoat, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Clearcoat roughness", &material.clearcoat_roughness, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Clearcoat IOR", &material.clearcoat_ior, 0.0f, 10.0f);
		some_material_changed |= ImGui::SliderFloat("IOR", &material.ior, 0.0f, 10.0f);
		some_material_changed |= ImGui::SliderFloat("Transmission", &material.specular_transmission, 0.0f, 1.0f);
		some_material_changed |= ImGui::ColorEdit3("Emission", (float*)&material.emission, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);

		ImGui::PopID();

		ImGui::Separator();

		if (some_material_changed && material_modfied_id == -1)
			material_modfied_id = material_counter;
		material_counter++;
	}
	ImGui::PopItemWidth();

	if (some_material_changed)
	{
		RendererMaterial& material = materials[material_modfied_id];
		material.make_safe();
		material.precompute_properties();

		m_renderer.update_materials(materials);
		m_render_dirty = true;
	}

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void RenderWindow::show_denoiser_panel()
{
	if (!ImGui::CollapsingHeader("Denoiser"))
		return;
	ImGui::TreePush("Denoiser tree");

	ImGui::Checkbox("Enable denoiser", &m_application_settings.enable_denoising);
	ImGui::Checkbox("Only Denoise at Target Sample Count", &m_application_settings.denoise_at_target_sample_count);
	if (m_application_settings.denoise_at_target_sample_count)
	{
		ImGui::BeginDisabled();
		ImGui::SliderInt("Denoise Sample Skip", &m_application_settings.denoiser_sample_skip, 1, 128);
		ImGui::EndDisabled();
	}
	else
		ImGui::SliderInt("Denoise Sample Skip", &m_application_settings.denoiser_sample_skip, 1, 128);

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void RenderWindow::show_post_process_panel()
{
	if (!ImGui::CollapsingHeader("Post-processing"))
		return;
	ImGui::TreePush("Post-processing tree");

	if (ImGui::Checkbox("Do tonemapping", &m_application_settings.do_tonemapping))
		m_active_display_program.set_uniform("ud_o_tonemapping", m_application_settings.do_tonemapping);
	if (ImGui::InputFloat("Gamma", &m_application_settings.tone_mapping_gamma))
		m_active_display_program.set_uniform("u_gamma", m_application_settings.tone_mapping_gamma);
	if (ImGui::InputFloat("Exposure", &m_application_settings.tone_mapping_exposure))
		m_active_display_program.set_uniform("u_exposure", m_application_settings.tone_mapping_exposure);

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void RenderWindow::draw_imgui()
{
	ImGuiIO& io = ImGui::GetIO();
	ImGui::ShowDemoWindow();

	ImGui::Begin("Settings");

	auto now_time = std::chrono::high_resolution_clock::now();
	float render_time = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - m_startRenderTime).count();
	ImGui::Text("Render time: %.3fs", render_time / 1000.0f);
	ImGui::Text("%d samples | %.2f samples/s @ %dx%d", m_render_settings.sample_number, 1.0f / io.DeltaTime * (m_render_settings.render_low_resolution ? 1 : m_render_settings.samples_per_frame), m_renderer.m_render_width, m_renderer.m_render_height);

	ImGui::Separator();

	if (ImGui::Button("Save viewport to PNG (tonemapped)"))
		m_image_writer.write_to_png();
	
	ImGui::Separator();

	ImGui::PushItemWidth(233);

	show_render_settings_panel();
	show_objects_panel();
	show_denoiser_panel();
	show_post_process_panel();

	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void RenderWindow::quit()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glDeleteVertexArrays(1, &m_vao);
	glDeleteTextures(1, &m_display_texture);

	std::exit(0);
}
