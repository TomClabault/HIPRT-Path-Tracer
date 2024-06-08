/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/RenderWindow.h"

#include <functional>
#include <iostream>
#include "Scene/SceneParser.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "Utils/Utils.h"

#include "stb_image_write.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/euler_angles.hpp"

// TODO bugs
// - kernel register count not working on AMD anymore since NVIDIA CC 7.5 fix
// - TO TEST AGAIN: something is unsafe on NVIDIA + Windows + nested-dielectrics-complex.gltf + 48 bounces minimum + nested dielectric strategy RT Gems. We get a CPU-side orochi error when downloading the framebuffer for displaying indicating that some illegal memory was accessed. Is the buffer corrupted by something?
// - normals AOV not converging correctly ?
//		- for the denoiser normals convergence issue, is it an error at the end of the Path Tracer kernel where we're accumulating ? Should we have
//		render_data.aux_buffers.denoiser_albedo[index] * render_data.render_settings.sample_number 
//		instead of 
//		render_data.aux_buffers.denoiser_albedo[index] * render_data.render_settings.frame_number
//		?
// - denoiser AOVs not accounting for tranmission correctly since Disney 



// TODO Code Organization:
// - investigate why kernel compiling was so much faster in the past (commit db34b23 seems to be a good candidate)
// - refactor the usage of strings in the compile kernel functions
// - fork all submodules in case they "disappear"
// - multiple GLTF, one GLB for different point of views per model
// - cleanup orochi gl interop buffer #ifdef everywhere
// - do we need OpenGL Lib/bin in thirdparties?
// - fork HIPRT and remove the encryption thingy that slows down kernel compilation on NVIDIA
// - A good way to automatically find MSBuild with CMake? Build HIPRT with make instead of VS maybe?
// - refactor HIPCC compiler options
// - Destroy buffers when disabling adaptive sampling to save VRAM
// - uniform #ifndef in Device headers
// - Refactor material editor
// - Device/ or HostDeviceCommon. Not both
// - reorganize methods order in RenderWindow
// - imgui controller to put all the imgui code in one class
// - put mouse / keyboard code in an interactor
//		- Have the is_interacting boolean in this interactor class and poll it from the main loop to check whether we need to render the frame at a lower resolution or not
// - check for level of abstractions in functions



// TODO Features:
// - build BVHs one by one to avoid big memory spike? but what about BLAS performance cost?
// - ray statistics with filter functions
// - filter function for base color alpha / alpha transparency + better performabce ?
// - alpha transparency
// - look at blender cycles "medium contrast", "medium low constract", "medium high", ...
// - normal mapping strength
// - blackbody light emitters
// - ACES mapping
// - better post processing: contrast, low, medium, high exposure curve
// - bloom post processing
// - hold shift for faster camera
// - hold CTRL for slower camera
// - BRDF swapper ImGui : Disney, Lambertian, Oren Nayar, Cook Torrance, Perfect fresnel dielectric reflect/transmit
// - choose disney diffuse model (disney, lambertian, oren nayar)
// - Cool colored thread-safe logger singleton class --> loguru lib
// - portal envmap sampling --> choose portals with ImGui
// - recursive trace through transmissive / reflective materials for caustics
// - find a way to not fill the texcoords buffer for meshes that don't have textures
// - pack RendererMaterial informations such as texture indices (we can probably use 16 bit for a texture index --> 2 texture indices in one 32 bit register)
// - use 8 bit textures for material properties instead of float
// - log size of buffers used: vertices, indices, normals, ...
// - display active pixels adaptive sampling
// - able / disable normal mapping
// - use only one channel for material property texture to save VRAM
// - Scene parsing is pretty slow and seems to be CPU bound in our code, not ASSIMP so have a look at that
// - Remove vertex normals for meshes that have normal maps and save VRAM
// - texture compression
// - float compression for render buffers?
// - Exporter (just serialize the scene to binary file I guess)
// - Allow material parameters textures manipulation with ImGui
// - Disable material parameters in ImGui that have a texture associated (since the ImGui slider in this case has no effect)
// - Upload grayscale as one channel to the GPU instead of memory costly RGBA
// - Emissive textures sampling: how to sample an object that has an emissive texture? How to know which triangles of the mesh are covered by the emissive parts of the texture?
// - stream compaction / active thread compaction (ingo wald 2011)
// - sample regeneration
// - structure of arrays instead of arrays of struct relevant for global buffers in terms of performance?
// - data packing in buffer --> use one 32 bit buffer to store multiple information if not using all 32 bits
//		- pack active pixel in same buffer as pixel sample count
// - pack two texture indices in one int for register saving, 65536 (16 bit per index when packed) textures is enough
// - hint shadow rays for better traversal perf
// - benchmarker to measure frame times precisely (avg, std dev, ...) + fixed random seed for reproducible results
// - alias table for sampling env map instead of log(n) binary search
// - image comparator slider (to have adaptive sampling view + default view on the same viewport for example)
// - auto adaptive sample per frame with adaptive sampling to keep GPU busy
// - Maybe look at better Disney sampling (luminance?)
// - thin materials
// - Look at what Orochi & HIPCC can do in terms of displaying registers used / options to specify shared stack size / block size (-DBLOCK_SIZE, -DSHARED_STACK_SIZE)
// - Have the UI run at its own framerate to avoid having the UI come to a crawl when the path tracing is expensive
// - Denoiser blend to allow blending the original noisy image and the perfect denoised result by a given factor
// - When modifying the emission of a material with the material editor, it should be reflected in the scene and allow the direct sampling of the geometry so the emissive triangles buffer should be updated
// - Ray differentials for texture mipampping (better bandwidth utilization since sampling potentially smaller texture --> fit better in cache)
// - Ray reordering for performance
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
// - Flakes BRDF (maybe look at OSPRay implementation for a reference ?)
// - ImGuizmo for moving objects in the scene
// - Paths roughness regularization
// - choose denoiser quality in imgui
// - try async buffer copy for the denoiser (maybe run a kernel to generate normals and another to generate albedo buffer before the path tracing kernel to be able to async copy while the path tracing kernel is running?)
// - enable denoising with all combinations of beauty/normal/albedo via imgui
// - cutout filters
// - write scene details to imgui (nb vertices, triangles, ...)
// - ImGui to choose the BVH flags at runtime and be able to compare the performance
// - ImGui widgets for SBVH / LBVH
// - BVH compaction + imgui checkbox
// - shader cache (write our own or wait for HIPRT to fix it?)
// - indirect / direct lighting clamping
// - choose env map at runtime imgui
// - choose scene file at runtime imgui
// - lock camera checkbox to avoid messing up when big render in progress
// - use defines insead of IFs in the kernel code and recompile kernel everytime (for some options at least to reduce register pressure)
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
static bool just_pressed = false;
void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	bool imgui_wants_mouse = ImGui::GetIO().WantCaptureMouse;

	switch (button)
	{
	case GLFW_MOUSE_BUTTON_LEFT:
		interacting_left_button = (action == GLFW_PRESS && !imgui_wants_mouse);

		break;

	case GLFW_MOUSE_BUTTON_RIGHT:
		interacting_right_button = (action == GLFW_PRESS && !imgui_wants_mouse);

		break;
	}

	RenderWindow* render_window = reinterpret_cast<RenderWindow*>(glfwGetWindowUserPointer(window));

	bool is_mouse_pressed = interacting_left_button || interacting_right_button;
	if (is_mouse_pressed)
	{
		double current_x, current_y;
		glfwGetCursorPos(window, &current_x, &current_y);
		render_window->set_grab_cursor_position(std::make_pair(static_cast<float>(current_x), static_cast<float>(current_y)));

		just_pressed = true;
	}
	else
		just_pressed = false;

	render_window->set_interacting(is_mouse_pressed);
}

void glfw_mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos)
{
	ImGuiIO& io = ImGui::GetIO();
	if (!io.WantCaptureMouse)
	{
		if (just_pressed)
		{
			// We want to skip the frame where the mouse is being repositioned to
			// the center of the screen because if the cursor wasn't at the center,
			// we're going to consider to delta from the old position to the center as
			// the moving having moved but it's not the case. The user didn't move the
			// mouse, it's us forcing it in the center of the viewport
			just_pressed = false;

			return;
		}

		RenderWindow* render_window = reinterpret_cast<RenderWindow*>(glfwGetWindowUserPointer(window));

		float xposf = static_cast<float>(xpos);
		float yposf = static_cast<float>(ypos);

		std::pair<float, float> old_position = render_window->get_grab_cursor_position();
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
		if (render_window->is_interacting())
			// Locking the cursor in place as long as we're moving the camera
			glfwSetCursorPos(window, old_position.first, old_position.second);
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
	Utils::debugbreak();
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
	glfwSwapInterval(0);

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

	m_renderer.initialize(0);
	ThreadManager::start_thread(ThreadManager::COMPILE_KERNEL_THREAD_KEY, ThreadFunctions::compile_kernel, std::ref(m_renderer), m_application_settings.kernel_files[m_application_settings.selected_kernel].c_str(), m_application_settings.kernel_functions[m_application_settings.selected_kernel].c_str());
	//m_renderer.compile_trace_kernel(DEVICE_KERNELS_DIRECTORY "/RegisterTestKernel.h", "TestFunction");
	m_renderer.change_render_resolution(width, height);
	create_display_programs();

	// TODO fix denoiser buffer since openGL interop
	/*m_denoiser.set_buffers(m_renderer.get_color_framebuffer().get_device_pointer(),
		m_renderer.get_denoiser_normals_buffer().get_device_pointer(), m_renderer.get_denoiser_albedo_buffer().get_device_pointer(),
		width, height);*/

	m_screenshoter.set_renderer(&m_renderer);
	m_screenshoter.set_render_window(this);

	// Making the render dirty to force a cleanup at startup
	m_render_dirty = true;
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

	recreate_display_texture(m_display_texture_type, new_render_width, new_render_height);
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

bool RenderWindow::is_interacting()
{
	return m_render_settings.render_low_resolution;
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
	// 
	glGenTextures(1, &m_display_texture);
	recreate_display_texture_from_display_view(m_application_settings.display_view);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// This empty VAO is necessary on NVIDIA drivers even though
	// we're hardcoding our full screen quad in the vertex shader
	glCreateVertexArrays(1, &m_vao);

	OpenGLShader fullscreen_quad_vertex_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/fullscreen_quad.vert", OpenGLShader::VERTEX_SHADER);
	OpenGLShader default_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/default_display.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader normal_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/normal_display.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader albedo_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/albedo_display.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader adaptive_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/heatmap_int.frag", OpenGLShader::FRAGMENT_SHADER);

	m_default_display_program.attach(fullscreen_quad_vertex_shader);
	m_default_display_program.attach(default_display_fragment_shader);
	m_default_display_program.link();

	m_normal_display_program.attach(fullscreen_quad_vertex_shader);
	m_normal_display_program.attach(normal_display_fragment_shader);
	m_normal_display_program.link();

	m_albedo_display_program.attach(fullscreen_quad_vertex_shader);
	m_albedo_display_program.attach(albedo_display_fragment_shader);
	m_albedo_display_program.link();

	m_adaptive_sampling_display_program.attach(fullscreen_quad_vertex_shader);
	m_adaptive_sampling_display_program.attach(adaptive_display_fragment_shader);
	m_adaptive_sampling_display_program.link();

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
		m_active_display_program = m_adaptive_sampling_display_program;
		break;

	default:
		break;
	}

	recreate_display_texture_from_display_view(display_view);
}

void RenderWindow::recreate_display_texture_from_display_view(DisplayView display_view)
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
		recreate_display_texture(texture_type_needed, m_renderer.m_render_width, m_renderer.m_render_height);
}

void RenderWindow::recreate_display_texture(DisplayTextureType texture_type, int width, int height)
{
	GLint internal_format = texture_type.get_gl_internal_format();
	GLenum format = texture_type.get_gl_format();
	GLenum type = texture_type.get_gl_type();

	// Making sure the buffer isn't bound
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glActiveTexture(GL_TEXTURE0 + RenderWindow::DISPLAY_TEXTURE_UNIT);
	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, type, nullptr);

	m_display_texture_type = texture_type;
}

void RenderWindow::upload_data_to_display_texture(const void* data, GLenum format, GLenum type)
{
	glActiveTexture(GL_TEXTURE0 + RenderWindow::DISPLAY_TEXTURE_UNIT);
	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_renderer.m_render_width, m_renderer.m_render_height, format, type, data);
}

void RenderWindow::update_program_uniforms(OpenGLProgram& program)
{
	program.use();

	int resolution_scaling = (m_render_settings.render_low_resolution) ? m_render_settings.render_low_resolution_scaling : 1;

	switch (m_application_settings.display_view)
	{
	case DisplayView::DEFAULT:
		program.set_uniform("u_texture", RenderWindow::DISPLAY_TEXTURE_UNIT);
		program.set_uniform("u_sample_number", m_render_settings.sample_number);
		program.set_uniform("u_do_tonemapping", m_application_settings.do_tonemapping);
		program.set_uniform("u_resolution_scaling", resolution_scaling);
		program.set_uniform("u_gamma", m_application_settings.tone_mapping_gamma);
		program.set_uniform("u_exposure", m_application_settings.tone_mapping_exposure);

		break;

	case DisplayView::DISPLAY_ALBEDO:
	case DisplayView::DISPLAY_DENOISED_ALBEDO:
		program.set_uniform("u_texture", RenderWindow::DISPLAY_TEXTURE_UNIT);
		program.set_uniform("u_sample_number", m_render_settings.sample_number);
		program.set_uniform("u_resolution_scaling", resolution_scaling);

		break;

	case DisplayView::DISPLAY_NORMALS:
	case DisplayView::DISPLAY_DENOISED_NORMALS:
		program.set_uniform("u_texture", RenderWindow::DISPLAY_TEXTURE_UNIT);
		program.set_uniform("u_sample_number", m_render_settings.sample_number);
		program.set_uniform("u_resolution_scaling", resolution_scaling);
		program.set_uniform("u_do_tonemapping", m_application_settings.do_tonemapping);
		program.set_uniform("u_gamma", m_application_settings.tone_mapping_gamma);
		program.set_uniform("u_exposure", m_application_settings.tone_mapping_exposure);

		break;

	case DisplayView::ADAPTIVE_SAMPLING_MAP:
		std::vector<ColorRGB> color_stops = { ColorRGB(0.0f, 0.0f, 1.0f), ColorRGB(0.0f, 1.0f, 0.0f), ColorRGB(1.0f, 0.0f, 0.0f) };

		float min_val = (float)m_render_settings.adaptive_sampling_min_samples;
		float max_val = std::max((float)m_render_settings.sample_number, min_val);

		program.set_uniform("u_texture", RenderWindow::DISPLAY_TEXTURE_UNIT);
		program.set_uniform("u_resolution_scaling", resolution_scaling);
		program.set_uniform("u_color_stops", 3, (float*)color_stops.data());
		program.set_uniform("u_nb_stops", 3);
		program.set_uniform("u_min_val", min_val);
		program.set_uniform("u_max_val", max_val);

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
		m_render_settings.sample_number++; // Only doing 1 SPP when moving the camera
	else
		m_render_settings.sample_number += m_render_settings.samples_per_frame;
}

bool RenderWindow::is_rendering_done()
{
	bool rendering_done = false;

	rendering_done |= m_renderer.get_ray_active_buffer().download_data()[0] == 0;
	rendering_done |= m_renderer.get_stop_noise_threshold_buffer().download_data()[0] == m_renderer.m_render_width * m_renderer.m_render_height;
	rendering_done |= (m_application_settings.max_sample_count != 0 && m_render_settings.sample_number + 1 > m_application_settings.max_sample_count);

	return rendering_done;
}

void RenderWindow::reset_render()
{
	unsigned char true_data = 1;
	unsigned int zero_data = 0;

	m_render_settings.frame_number = 0;
	if (m_application_settings.auto_sample_per_frame)
	{
		// Resetting the number of samples per frame to be sure we're not
		// going to timeout the GPU driver
		m_render_settings.samples_per_frame = 1;
		// Samples per second manually reset to 0.0f so that samples_per_frame
		// isn't automatically recalculated from the samples per second of last
		// frame (before the render reset) which would basically fail the whole
		// point of resetting the number of samples per frame.
		m_samples_per_second = 0.0f;
	}
	m_current_render_time = 0.0f;
	m_renderer.set_sample_number(0);
	m_renderer.get_ray_active_buffer().upload_data(&true_data);
	m_renderer.get_stop_noise_threshold_buffer().upload_data(&zero_data);

	m_render_dirty = false;
}

GPURenderer& RenderWindow::get_renderer()
{
	return m_renderer;
}

std::pair<float, float> RenderWindow::get_grab_cursor_position()
{
	return m_grab_cursor_position;
}

void RenderWindow::set_grab_cursor_position(std::pair<float, float> new_position)
{
	m_grab_cursor_position = new_position;
}

void RenderWindow::run()
{
	while (!glfwWindowShouldClose(m_window))
	{
		m_start_cpu_frame_time = std::chrono::high_resolution_clock::now();

		glfwPollEvents();
		glClear(GL_COLOR_BUFFER_BIT);

		update_keyboard_inputs();

		// We're resetting the render each frame if rendering at low resolution
		m_render_dirty |= m_render_settings.render_low_resolution == 1;
		if (m_render_dirty)
			reset_render();

		if (m_application_settings.auto_sample_per_frame && m_samples_per_second > 0)
			m_render_settings.samples_per_frame = std::min(std::max(1, static_cast<int>(m_samples_per_second / 20.0f)), 10000);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		render();

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
			// for OpenGL interop to be supported on NVIDIA by Orochi
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

			/*case DisplayView::ADAPTIVE_SAMPLING_ACTIVE_PIXELS:
				display(m_renderer.get_debug_pixel_active_buffer().download_data().data());
				break;*/

			case DisplayView::DEFAULT:
			default:
				display(m_renderer.get_color_framebuffer());
				break;
			}
		}
		
		m_stop_cpu_frame_time = std::chrono::high_resolution_clock::now();

		draw_imgui();

		glfwSwapBuffers(m_window);
	}

	quit();
}

void RenderWindow::update_keyboard_inputs()
{
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

	update_renderer_view_translation(-translation.first, translation.second);
	update_renderer_view_zoom(-zoom);
}

void RenderWindow::render()
{
	if (!is_rendering_done())
	{
		m_renderer.render();
		increment_sample_number();
		m_render_settings.frame_number++;
	}
	else
		// Sleeping so that we don't burn the CPU (and GPU because it'll have to do the display)
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

void RenderWindow::display(const void* data)
{
	GLenum format = m_display_texture_type.get_gl_format();
	GLenum type = m_display_texture_type.get_gl_type();
	upload_data_to_display_texture(data, format, type);

	update_program_uniforms(m_active_display_program);

	// Binding an empty VAO here (empty because we're hardcoding our full-screen quad vertices
	// in our vertex shader) because this is required on NVIDIA drivers
	glBindVertexArray(m_vao);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

void RenderWindow::draw_render_settings_panel()
{
	if (!ImGui::CollapsingHeader("Render Settings"))
		return;
	ImGui::TreePush("Render settings tree");

	if (ImGui::Combo("Render Kernel", &m_application_settings.selected_kernel, "Full Path Tracer\0Normals Visualisation\0\0"))
	{
		m_renderer.compile_trace_kernel(m_application_settings.kernel_files[m_application_settings.selected_kernel].c_str(), m_application_settings.kernel_functions[m_application_settings.selected_kernel].c_str());
		m_render_dirty = true;
	}

	const char* items[] = { "Default", "Denoiser - Normals", "Denoiser - Denoised normals", "Denoiser - Albedo", "Denoiser - Denoised albedo", "Adaptive sampling map"};
	if (ImGui::Combo("Display View", (int*)(&m_application_settings.display_view), items, IM_ARRAYSIZE(items)))
		select_display_program(m_application_settings.display_view);

	ImGui::BeginDisabled(m_application_settings.keep_same_resolution);
	float resolution_scaling_backup = m_application_settings.render_resolution_scale;
	if (ImGui::InputFloat("Resolution scale", &m_application_settings.render_resolution_scale))
	{
		float& resolution_scale = m_application_settings.render_resolution_scale;
		if (resolution_scale <= 0)
			resolution_scale = resolution_scaling_backup;

		change_resolution_scaling(resolution_scale);
		m_render_dirty = true;
	}
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

	unsigned int converged_count;
	unsigned int total_pixel_count;
	ImGui::BeginDisabled(m_render_settings.enable_adaptive_sampling);
	if (ImGui::InputFloat("Stop render at noise threshold", &m_render_settings.stop_noise_threshold))
	{
		unsigned int zero_data = 0;
		m_render_settings.stop_noise_threshold = std::max(0.0f, m_render_settings.stop_noise_threshold);
		m_renderer.get_stop_noise_threshold_buffer().upload_data(&zero_data);
	}

	ImGui::TreePush("Tree stop noise threshold");
	{
		converged_count = m_renderer.get_stop_noise_threshold_buffer().download_data()[0] * (!m_render_settings.enable_adaptive_sampling);
		total_pixel_count = m_renderer.m_render_width * m_renderer.m_render_height;
		ImGui::Text("Pixels converged: %d / %d - %.4f%%", converged_count, total_pixel_count, static_cast<float>(converged_count) / total_pixel_count * 100.0f);
	}
	ImGui::TreePop();
	ImGui::EndDisabled();

	ImGui::BeginDisabled(m_application_settings.auto_sample_per_frame);
	ImGui::InputInt("Samples per frame", &m_render_settings.samples_per_frame); 
	ImGui::EndDisabled();
	ImGui::SameLine();
	ImGui::Checkbox("Auto", &m_application_settings.auto_sample_per_frame);
	if (ImGui::InputInt("Max bounces", &m_render_settings.nb_bounces))
	{
		// Clamping to 0 in case the user input a negative number of bounces	
		m_render_settings.nb_bounces = std::max(m_render_settings.nb_bounces, 0); 
		m_render_dirty = true;
	}

	ImGui::Separator();
	if (ImGui::CollapsingHeader("Adaptive sampling"))
	{
		ImGui::TreePush("Adaptive sampling tree");

		m_render_dirty |= ImGui::Checkbox("Enable adaptive sampling", (bool*)&m_render_settings.enable_adaptive_sampling);

		ImGui::BeginDisabled(!m_render_settings.enable_adaptive_sampling);
		m_render_dirty |= ImGui::InputInt("Adaptive sampling minimum samples", &m_render_settings.adaptive_sampling_min_samples);
		if (ImGui::InputFloat("Adaptive sampling noise threshold", &m_render_settings.adaptive_sampling_noise_threshold))
		{
			m_render_settings.adaptive_sampling_noise_threshold = std::max(0.0f, m_render_settings.adaptive_sampling_noise_threshold);
			m_render_dirty = true;
		}
		ImGui::EndDisabled();

		ImGui::TreePop();
	}

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void RenderWindow::draw_lighting_panel()
{
	if (ImGui::CollapsingHeader("Lighting"))
	{
		ImGui::TreePush("Lighting tree");

		m_render_dirty |= ImGui::RadioButton("None", ((int*)&m_renderer.get_world_settings().ambient_light_type), 0); ImGui::SameLine();
		m_render_dirty |= ImGui::RadioButton("Use uniform lighting", ((int*)&m_renderer.get_world_settings().ambient_light_type), 1); ImGui::SameLine();
		m_render_dirty |= ImGui::RadioButton("Use envmap lighting", ((int*)&m_renderer.get_world_settings().ambient_light_type), 2);

		if (m_renderer.get_world_settings().ambient_light_type == AmbientLightType::UNIFORM)
		{
			m_render_dirty |= ImGui::ColorEdit3("Uniform light color", (float*)&m_renderer.get_world_settings().uniform_light_color, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
		}
		else if (m_renderer.get_world_settings().ambient_light_type == AmbientLightType::ENVMAP)
		{
			static float rota_X = 0.0f, rota_Y = 0.0f, rota_Z = 0.0f;
			bool rotation_changed;

			rotation_changed = false;
			rotation_changed |= ImGui::SliderFloat("Envmap rotation X", &rota_X, 0.0f, 1.0f);
			rotation_changed |= ImGui::SliderFloat("Envmap rotation Y", &rota_Y, 0.0f, 1.0f);
			rotation_changed |= ImGui::SliderFloat("Envmap rotation Z", &rota_Z, 0.0f, 1.0f);

			if (rotation_changed)
			{
				glm::mat4x4 rotation_matrix;
				
				// glm::orientate3 interprets the X, Y and Z angles we give it as a yaw/pitch/roll semantic.
				// 
				// The standard yaw/pitch/roll interpretation is:
				//	- Yaw for rotation around Z
				//	- Pitch for rotation around Y
				//	- Roll for rotation around X
				// 
				// but with a Z-up coordinate system. We want a Y-up coordinate system so
				// we want our Yaw to rotate around Y instead of Z (and our Pitch to rotate around Z).
				// 
				// This means that we need to reverse Y and Z.
				// 
				// See this picture for a visual aid on what we **don't** want (the z-up):
				// https://www.researchgate.net/figure/xyz-and-pitch-roll-and-yaw-systems_fig4_253569466
				rotation_matrix = glm::orientate3(glm::vec3(rota_X * 2.0f * M_PI, rota_Z * 2.0f * M_PI, rota_Y * 2.0f * M_PI));
				m_renderer.get_world_settings().envmap_rotation_matrix = *reinterpret_cast<float4x4*>(&rotation_matrix);
			}

			m_render_dirty |= rotation_changed;
			m_render_dirty |= ImGui::SliderFloat("Envmap intensity", (float*)&m_renderer.get_world_settings().envmap_intensity, 0.0f, 10.0f);
			ImGui::TreePush("Envmap intensity tree");
			m_render_dirty |= ImGui::Checkbox("Scale background intensity", (bool*)&m_renderer.get_world_settings().envmap_scale_background_intensity);
			ImGui::TreePop();
		}

		// Ensuring no negative light color
		m_renderer.get_world_settings().uniform_light_color.clamp(0.0f, 1.0e38f);

		ImGui::TreePop();

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
	}
}

void RenderWindow::draw_objects_panel()
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
		some_material_changed |= ImGui::SliderFloat("Clearcoat IOR", &material.clearcoat_ior, 0.0f, 5.0f);
		some_material_changed |= ImGui::SliderFloat("IOR", &material.ior, 0.0f, 5.0f);
		ImGui::Separator();
		some_material_changed |= ImGui::SliderFloat("Transmission", &material.specular_transmission, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Absorption distance", &material.absorption_at_distance, 0.0f, 20.0f);
		some_material_changed |= ImGui::ColorEdit3("Absorption color", (float*)&material.absorption_color);
		unsigned short int zero = 0, eight = 8;
		ImGui::BeginDisabled(material.specular_transmission == 0.0f);
		some_material_changed |= ImGui::SliderScalar("Dielectric priority", ImGuiDataType_U16, &material.dielectric_priority, &zero, &eight);
		ImGui::EndDisabled();
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
	ImGui::BeginDisabled(m_application_settings.denoise_at_target_sample_count);
	ImGui::SliderInt("Denoise Sample Skip", &m_application_settings.denoiser_sample_skip, 1, 128);
	ImGui::EndDisabled();

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void RenderWindow::draw_post_process_panel()
{
	if (!ImGui::CollapsingHeader("Post-processing"))
		return;
	ImGui::TreePush("Post-processing tree");

	ImGui::Checkbox("Do tonemapping", &m_application_settings.do_tonemapping);
	ImGui::InputFloat("Gamma", &m_application_settings.tone_mapping_gamma);
	ImGui::InputFloat("Exposure", &m_application_settings.tone_mapping_exposure);

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void RenderWindow::draw_performance_panel()
{
	if (!ImGui::CollapsingHeader("Performance"))
		return;

	ImGui::TreePush("Performance tree");

	ImGui::Text("Device: %s", m_renderer.get_device_properties().name);
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	if (ImGui::Button("Apply benchmark settings"))
	{
		m_render_settings.freeze_random = true;
		m_render_settings.enable_adaptive_sampling = false;
		m_application_settings.auto_sample_per_frame = false;

		reset_render();
	}
	if (ImGui::Checkbox("Freeze random", (bool*)&m_render_settings.freeze_random))
		reset_render();

	bool rolling_window_size_changed = false;
	int rolling_window_size = m_perf_metrics.get_window_size();
	rolling_window_size_changed |= ImGui::RadioButton("25", &rolling_window_size, 25); ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("100", &rolling_window_size, 100); ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("1000", &rolling_window_size, 1000);

	if (rolling_window_size_changed)
		m_perf_metrics.resize_window(rolling_window_size);

	float variance, min, max;
	variance = m_perf_metrics.get_variance(PerformanceMetricsComputer::SAMPLE_TIME_KEY);
	min = m_perf_metrics.get_min(PerformanceMetricsComputer::SAMPLE_TIME_KEY);
	max = m_perf_metrics.get_max(PerformanceMetricsComputer::SAMPLE_TIME_KEY);

	static float scale_min = min, scale_max = max;
	scale_min = m_perf_metrics.get_data_index(PerformanceMetricsComputer::SAMPLE_TIME_KEY) == 0 ? min : scale_min;
	scale_max = m_perf_metrics.get_data_index(PerformanceMetricsComputer::SAMPLE_TIME_KEY) == 0 ? max : scale_max;

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::PlotHistogram("", 
						PerformanceMetricsComputer::data_getter, 
						m_perf_metrics.get_data(PerformanceMetricsComputer::SAMPLE_TIME_KEY).data(), 
						m_perf_metrics.get_value_count(PerformanceMetricsComputer::SAMPLE_TIME_KEY), 
						/* value offset */0, 
						"Sample time", 
						scale_min, scale_max, 
						/* size */ ImVec2(0, 80));
	ImGui::SameLine();
	if (ImGui::Button("Rescale"))
	{
		scale_min = min;
		scale_max = max;
	}
	ImGui::Text("Sample time (avg)      : %.3fms (%.1f FPS)", m_perf_metrics.get_average(PerformanceMetricsComputer::SAMPLE_TIME_KEY), 1000.0f / m_perf_metrics.get_average(PerformanceMetricsComputer::SAMPLE_TIME_KEY));
	ImGui::Text("Sample time (var)      : %.3fms", variance);
	ImGui::Text("Sample time (std dev)  : %.3fms", std::sqrt(variance));
	ImGui::Text("Sample time (min / max): %.3fms / %.3fms", min, max);

	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	ImGui::TreePop();
}

void RenderWindow::draw_imgui()
{
	ImGuiIO& io = ImGui::GetIO();
	ImGui::ShowDemoWindow();

	ImGui::Begin("Settings");

	auto now_time = std::chrono::high_resolution_clock::now();
	if (!is_rendering_done())
	{
		float sample_time = m_renderer.get_frame_time() / ((m_render_settings.render_low_resolution ? 1 : m_render_settings.samples_per_frame));

		m_current_render_time += std::chrono::duration_cast<std::chrono::milliseconds>(m_stop_cpu_frame_time - m_start_cpu_frame_time).count();
		m_samples_per_second = 1000.0f / sample_time;

		if (!m_render_settings.render_low_resolution)
			// Not adding the frame time if we're rendering low resolution, not relevant
			m_perf_metrics.add_value(PerformanceMetricsComputer::SAMPLE_TIME_KEY, sample_time);
	}

	ImGui::Text("Render time: %.3fs", m_current_render_time/ 1000.0f);
	ImGui::Text("%d samples | %.2f samples/s @ %dx%d", m_render_settings.sample_number, m_samples_per_second, m_renderer.m_render_width, m_renderer.m_render_height);

	ImGui::Separator();

	if (ImGui::Button("Save viewport to PNG (tonemapped)"))
		m_screenshoter.write_to_png();
	
	ImGui::Separator();

	ImGui::PushItemWidth(233);

	draw_render_settings_panel();
	draw_lighting_panel();
	draw_objects_panel();
	//show_denoiser_panel();
	draw_post_process_panel();
	draw_performance_panel();

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
