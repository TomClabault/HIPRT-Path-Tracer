/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Scene/SceneParser.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "UI/RenderWindow.h"
#include "UI/LinuxRenderWindowMouseInteractor.h"
#include "UI/WindowsRenderWindowMouseInteractor.h"
#include "Utils/Utils.h"

#include <functional>
#include <iostream>

#include "stb_image_write.h"

// TODO bugs
// - denoiser albedo display not properly scaled
// - TO TEST AGAIN: something is unsafe on NVIDIA + Windows + nested-dielectrics-complex.gltf + 48 bounces minimum + nested dielectric strategy RT Gems. We get a CPU-side orochi error when downloading the framebuffer for displaying indicating that some illegal memory was accessed. Is the buffer corrupted by something?
// - denoiser AOVs not accounting for transmission correctly since Disney 
//	  - same with perfect reflection
// - Issue with screenshoter + denoiser, PNG output is way darker than viewport (sometimes?)



// TODO Code Organization:
// - rename target sample count to max sample count
// - update OIDN to 2.3.0 non beta
// - get grab / set grab cursor position needs to be in windows interactor, not render window, get_cursor_position? set_interacting?
// - investigate why kernel compiling was so much faster in the past (commit db34b23 seems to be a good candidate)
// - refactor the usage of strings in the compile kernel functions
// - multiple GLTF, one GLB for different point of views per model
// - cleanup orochi gl interop buffer #ifdef everywhere
// - do we need OpenGL Lib/bin in thirdparties?
// - fork HIPRT and remove the encryption thingy that slows down kernel compilation on NVIDIA
// - A good way to automatically find MSBuild with CMake? Build HIPRT with make instead of VS maybe?
// - uniform #ifndef in Device headers
// - Refactor material editor
// - Device/ or HostDeviceCommon. Not both
// - reorganize methods order in RenderWindow
// - imgui controller to put all the imgui code in one class
// - check for level of abstractions in functions



// TODO Features:
// - can we do direct lighting + take emissive at all bounces but divide by 2 to avoid double taking into account emissive lights? this would solve missing caustics
// - improve performance by only intersecting the selected emissive triangle with the BSDF ray when multiple importance sampling, we don't need a full BVH traversal at all
// - If could not load given scene file, fallback to cornell box instead of not continuing
// - CTRL + mouse wheel for zoom in viewport, CTRL click reset zoom
// - Kahan summation for weighted reservoir sampling?
// - hardware triangle intersection can be disabled in HIPRT Compiler.cpp so that's good for comparing performance (__USE_HWI__ define)
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
// - Spectral rendering / look at gemstone rendering because they quite a lot of interesting lighting effect to take into account (pleochroism, birefringent, dispersion, ...)
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
// - Minimum contribution to speed things up as in OSPRay ?
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

RenderWindow::RenderWindow(int width, int height) : m_viewport_width(width), m_viewport_height(height)
{
	init_glfw(width, height);
	init_gl(width, height);
	init_imgui();

	m_application_settings = std::make_shared<ApplicationSettings>();

	m_renderer = std::make_shared<GPURenderer>();
	m_renderer->initialize(0);
	ThreadManager::start_thread(ThreadManager::COMPILE_KERNEL_THREAD_KEY, ThreadFunctions::compile_kernel, std::ref(m_renderer), m_application_settings->kernel_files[m_application_settings->selected_kernel].c_str(), m_application_settings->kernel_functions[m_application_settings->selected_kernel].c_str());
	m_renderer->change_render_resolution(width, height);

	m_denoiser = std::make_shared<OpenImageDenoiser>();
	m_denoiser->initialize();
	m_denoiser->resize(width, height);
	m_denoiser->set_use_albedo(m_application_settings->denoiser_use_albedo);
	m_denoiser->set_use_normals(m_application_settings->denoiser_use_normals);
	m_denoiser->finalize();

	m_screenshoter = std::make_shared<Screenshoter>();
	m_screenshoter->set_renderer(m_renderer);
	m_screenshoter->set_render_window(this);

	m_perf_metrics = std::make_shared<PerformanceMetricsComputer>();

	m_imgui_renderer.set_render_window(this);

	create_display_programs();

	// Making the render dirty to force a cleanup at startup
	m_render_dirty = true;
}

RenderWindow::~RenderWindow()
{
	glfwDestroyWindow(m_glfw_window);
	glfwTerminate();
}

void RenderWindow::init_glfw(int width, int height)
{
	if (!glfwInit())
		wait_and_exit("Could not initialize GLFW...");

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);

#ifdef __unix__         
	m_mouse_interactor = std::make_unique<LinuxRenderWindowMouseInteractor>();
#elif defined(_WIN32) || defined(WIN32) 
	m_mouse_interactor = std::make_unique<WindowsRenderWindowMouseInteractor>();
#endif
	m_keyboard_interactor.set_render_window(this);

	m_glfw_window = glfwCreateWindow(width, height, "HIPRT Path Tracer", NULL, NULL);
	if (!m_glfw_window)
		wait_and_exit("Could not initialize the GLFW window...");

	glfwMakeContextCurrent(m_glfw_window);
	// Setting a pointer to this instance of RenderWindow inside the m_window GLFWwindow so that
	// we can retrieve a pointer to this instance of RenderWindow in the callback functions
	// such as the window_resized_callback function for example
	glfwSetWindowUserPointer(m_glfw_window, this);
	glfwSwapInterval(0);
	glfwSetWindowSizeCallback(m_glfw_window, glfw_window_resized_callback);
	m_mouse_interactor->set_callbacks(m_glfw_window);
	m_keyboard_interactor.set_callbacks(m_glfw_window);

	glewInit();
}

void RenderWindow::init_gl(int width, int height)
{
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
}

void RenderWindow::init_imgui()
{
	// Setting ImGui up
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	ImGui_ImplGlfw_InitForOpenGL(m_glfw_window, true);
	ImGui_ImplOpenGL3_Init();
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
	float& resolution_scale = m_application_settings->render_resolution_scale;
	if (m_application_settings->keep_same_resolution)
		resolution_scale = m_application_settings->target_width / (float)pixels_width; // TODO what about the height changing ?

	int new_render_width = std::floor(pixels_width * resolution_scale);
	int new_render_height = std::floor(pixels_height * resolution_scale);

	if (new_render_height == 0 || new_render_width == 0)
		// Can happen if resizing the window to a 1 pixel width/height while having a resolution scaling < 1. 
		// Integer maths will round it down to 0
		return;
	
	m_renderer->change_render_resolution(new_render_width, new_render_height);

	m_denoiser->resize(new_render_width, new_render_height);
	m_denoiser->finalize();

	internal_recreate_display_texture(m_display_texture_1, RenderWindow::DISPLAY_TEXTURE_UNIT_1, m_display_texture_1.second, new_render_width, new_render_height);
	internal_recreate_display_texture(m_display_texture_2, RenderWindow::DISPLAY_TEXTURE_UNIT_2, m_display_texture_2.second, new_render_width, new_render_height);

	m_render_dirty = true;
}

void RenderWindow::change_resolution_scaling(float new_scaling)
{
	float new_render_width = std::floor(m_viewport_width * new_scaling);
	float new_render_height = std::floor(m_viewport_height * new_scaling);

	m_renderer->change_render_resolution(new_render_width, new_render_height);
	m_denoiser->resize(new_render_width, new_render_height);
	m_denoiser->finalize();

	internal_recreate_display_texture(m_display_texture_1, RenderWindow::DISPLAY_TEXTURE_UNIT_1, m_display_texture_1.second, new_render_width, new_render_height);
	internal_recreate_display_texture(m_display_texture_2, RenderWindow::DISPLAY_TEXTURE_UNIT_2, m_display_texture_2.second, new_render_width, new_render_height);
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
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	// The user just released the camera and we were rendering at low resolution
	if (!is_interacting && render_settings.render_low_resolution)
		m_render_dirty = true;

	render_settings.render_low_resolution = is_interacting;
}

bool RenderWindow::is_interacting()
{
	return m_renderer->get_render_settings().render_low_resolution;
}

std::shared_ptr<ApplicationSettings> RenderWindow::get_application_settings()
{
	return m_application_settings;
}

void RenderWindow::create_display_programs()
{
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

	m_default_display_program.attach(fullscreen_quad_vertex_shader);
	m_default_display_program.attach(default_display_fragment_shader);
	m_default_display_program.link();

	m_blend_2_display_program.attach(fullscreen_quad_vertex_shader);
	m_blend_2_display_program.attach(blend_2_display_fragment_shader);
	m_blend_2_display_program.link();

	m_normal_display_program.attach(fullscreen_quad_vertex_shader);
	m_normal_display_program.attach(normal_display_fragment_shader);
	m_normal_display_program.link();

	m_albedo_display_program.attach(fullscreen_quad_vertex_shader);
	m_albedo_display_program.attach(albedo_display_fragment_shader);
	m_albedo_display_program.link();

	m_adaptive_sampling_display_program.attach(fullscreen_quad_vertex_shader);
	m_adaptive_sampling_display_program.attach(adaptive_display_fragment_shader);
	m_adaptive_sampling_display_program.link();

	change_display_view(m_application_settings->display_view);
}

void RenderWindow::change_display_view(DisplayView display_view)
{
	m_application_settings->display_view = display_view;

	// Adjusting the denoiser setting according to the selected view
	m_application_settings->enable_denoising = display_view == DisplayView::DENOISED_BLEND;

	switch (display_view)
	{
	case DisplayView::DEFAULT:
		m_active_display_program = m_default_display_program;
		break;

	case DisplayView::DENOISED_BLEND:
		m_active_display_program = m_blend_2_display_program;
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

	internal_recreate_display_textures_from_display_view(display_view);
}

void RenderWindow::internal_recreate_display_textures_from_display_view(DisplayView display_view)
{
	DisplayTextureType texture_1_type_needed = DisplayTextureType::UNINITIALIZED;
	DisplayTextureType texture_2_type_needed = DisplayTextureType::UNINITIALIZED;

	switch (display_view)
	{
	case DisplayView::DISPLAY_NORMALS:
	case DisplayView::DISPLAY_DENOISED_NORMALS:
	case DisplayView::DISPLAY_ALBEDO:
	case DisplayView::DISPLAY_DENOISED_ALBEDO:
		texture_1_type_needed = DisplayTextureType::FLOAT3;
		break;

	case DisplayView::ADAPTIVE_SAMPLING_MAP:
		texture_1_type_needed = DisplayTextureType::INT;
		break;

	case DisplayView::DENOISED_BLEND:
		texture_1_type_needed = DisplayTextureType::FLOAT3;
		texture_2_type_needed = DisplayTextureType::FLOAT3;
		break;

	case DisplayView::DEFAULT:
	default:
		texture_1_type_needed = DisplayTextureType::FLOAT3;
		break;
	}

	if (m_display_texture_1.second != texture_1_type_needed)
		internal_recreate_display_texture(m_display_texture_1, RenderWindow::DISPLAY_TEXTURE_UNIT_1, texture_1_type_needed, m_renderer->m_render_width, m_renderer->m_render_height);

	if (m_display_texture_2.second != texture_2_type_needed)
		internal_recreate_display_texture(m_display_texture_2, RenderWindow::DISPLAY_TEXTURE_UNIT_2, texture_2_type_needed, m_renderer->m_render_width, m_renderer->m_render_height);
}

void RenderWindow::internal_recreate_display_texture(std::pair<GLuint, DisplayTextureType>& display_texture, GLenum display_texture_unit, DisplayTextureType new_texture_type, int width, int height)
{
	bool freeing = false;
	if (new_texture_type == DisplayTextureType::UNINITIALIZED)
	{
		if (display_texture.second != DisplayTextureType::UNINITIALIZED)
		{
			// If the texture was valid before and we've given UNINITIALIZED as the new type, this means
			// that we're not using the texture anymore. We're going to resize the texture to 1x1,
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

void RenderWindow::upload_data_to_display_texture(GLuint display_texture, const void* data, GLenum format, GLenum type)
{
	glActiveTexture(GL_TEXTURE0 + RenderWindow::DISPLAY_TEXTURE_UNIT_1);
	glBindTexture(GL_TEXTURE_2D, display_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_renderer->m_render_width, m_renderer->m_render_height, format, type, data);
}

void RenderWindow::update_program_uniforms(OpenGLProgram& program)
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
	int resolution_scaling = (render_settings.render_low_resolution) ? render_settings.render_low_resolution_scaling : 1;

	program.use();

	switch (m_application_settings->display_view)
	{
	case DisplayView::DEFAULT:
		int sample_number;
		if (m_application_settings->enable_denoising && m_application_settings->last_denoised_sample_count != -1)
			sample_number = m_application_settings->last_denoised_sample_count;
		else
			sample_number = render_settings.sample_number;

		program.set_uniform("u_texture", RenderWindow::DISPLAY_TEXTURE_UNIT_1);
		program.set_uniform("u_sample_number", sample_number);
		program.set_uniform("u_do_tonemapping", m_application_settings->do_tonemapping);
		program.set_uniform("u_resolution_scaling", resolution_scaling);
		program.set_uniform("u_gamma", m_application_settings->tone_mapping_gamma);
		program.set_uniform("u_exposure", m_application_settings->tone_mapping_exposure);

		break;

	case DisplayView::DENOISED_BLEND:
		int noisy_sample_number;
		int denoised_sample_number;

		noisy_sample_number = render_settings.sample_number;
		denoised_sample_number = m_application_settings->last_denoised_sample_count;

		program.set_uniform("u_blend_factor", m_application_settings->denoiser_blend);
		program.set_uniform("u_texture_1", RenderWindow::DISPLAY_TEXTURE_UNIT_1);
		program.set_uniform("u_texture_2", RenderWindow::DISPLAY_TEXTURE_UNIT_2);
		program.set_uniform("u_sample_number_1", noisy_sample_number);
		program.set_uniform("u_sample_number_2", denoised_sample_number);
		program.set_uniform("u_do_tonemapping", m_application_settings->do_tonemapping);
		program.set_uniform("u_resolution_scaling", resolution_scaling);
		program.set_uniform("u_gamma", m_application_settings->tone_mapping_gamma);
		program.set_uniform("u_exposure", m_application_settings->tone_mapping_exposure);

		break;

	case DisplayView::DISPLAY_ALBEDO:
	case DisplayView::DISPLAY_DENOISED_ALBEDO:
		program.set_uniform("u_texture", RenderWindow::DISPLAY_TEXTURE_UNIT_1);
		program.set_uniform("u_resolution_scaling", resolution_scaling);

		break;

	case DisplayView::DISPLAY_NORMALS:
	case DisplayView::DISPLAY_DENOISED_NORMALS:
		program.set_uniform("u_texture", RenderWindow::DISPLAY_TEXTURE_UNIT_1);
		program.set_uniform("u_resolution_scaling", resolution_scaling);
		program.set_uniform("u_do_tonemapping", m_application_settings->do_tonemapping);
		program.set_uniform("u_gamma", m_application_settings->tone_mapping_gamma);
		program.set_uniform("u_exposure", m_application_settings->tone_mapping_exposure);

		break;

	case DisplayView::ADAPTIVE_SAMPLING_MAP:
		std::vector<ColorRGB> color_stops = { ColorRGB(0.0f, 0.0f, 1.0f), ColorRGB(0.0f, 1.0f, 0.0f), ColorRGB(1.0f, 0.0f, 0.0f) };

		float min_val = (float)render_settings.adaptive_sampling_min_samples;
		float max_val = std::max((float)render_settings.sample_number, min_val);

		program.set_uniform("u_texture", RenderWindow::DISPLAY_TEXTURE_UNIT_1);
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

	glm::vec3 translation = glm::vec3(translation_x / m_application_settings->view_translation_sldwn_x, translation_y / m_application_settings->view_translation_sldwn_y, 0.0f);
	m_renderer->translate_camera_view(translation);
}

void RenderWindow::update_renderer_view_rotation(float offset_x, float offset_y)
{
	m_render_dirty = true;

	float rotation_x, rotation_y;

	rotation_x = offset_x / m_viewport_width * 2.0f * M_PI / m_application_settings->view_rotation_sldwn_x;
	rotation_y = offset_y / m_viewport_height * 2.0f * M_PI / m_application_settings->view_rotation_sldwn_y;

	m_renderer->rotate_camera_view(glm::vec3(rotation_x, rotation_y, 0.0f));
}

void RenderWindow::update_renderer_view_zoom(float offset)
{
	if (offset == 0.0f)
		return;

	m_render_dirty = true;

	m_renderer->zoom_camera_view(offset / m_application_settings->view_zoom_sldwn);
}

void RenderWindow::increment_sample_number()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	if (render_settings.render_low_resolution)
		render_settings.sample_number++; // Only doing 1 SPP when moving the camera
	else
		render_settings.sample_number += render_settings.samples_per_frame;
}

bool RenderWindow::is_rendering_done()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	bool rendering_done = false;

	rendering_done |= m_renderer->get_ray_active_buffer().download_data()[0] == 0;
	rendering_done |= m_renderer->get_stop_noise_threshold_buffer().download_data()[0] == m_renderer->m_render_width * m_renderer->m_render_height;
	rendering_done |= (m_application_settings->max_sample_count != 0 && render_settings.sample_number + 1 > m_application_settings->max_sample_count);

	return rendering_done;
}

void RenderWindow::reset_render()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	unsigned char true_data = 1;
	unsigned int zero_data = 0;

	render_settings.frame_number = 0;
	if (m_application_settings->auto_sample_per_frame)
	{
		// Resetting the number of samples per frame to be sure we're not
		// going to timeout the GPU driver
		render_settings.samples_per_frame = 1;
		// Samples per second manually reset to 0.0f so that samples_per_frame
		// isn't automatically recalculated from the samples per second of last
		// frame (before the render reset) which would basically fail the whole
		// point of resetting the number of samples per frame.
		m_samples_per_second = 0.0f;
	}
	m_current_render_time = 0.0f;
	render_settings.sample_number = 0;
	m_renderer->get_ray_active_buffer().upload_data(&true_data);
	m_renderer->get_stop_noise_threshold_buffer().upload_data(&zero_data);
	m_application_settings->last_denoised_sample_count = -1;

	m_render_dirty = false;
}

void RenderWindow::set_render_dirty(bool render_dirty)
{
	m_render_dirty = render_dirty;
}

float RenderWindow::get_current_render_time()
{
	return m_current_render_time;
}

float RenderWindow::get_samples_per_second()
{
	return m_samples_per_second;
}

std::shared_ptr<OpenImageDenoiser> RenderWindow::get_denoiser()
{
	return m_denoiser;
}

std::shared_ptr<GPURenderer> RenderWindow::get_renderer()
{
	return m_renderer;
}

std::shared_ptr<PerformanceMetricsComputer> RenderWindow::get_performance_metrics()
{
	return m_perf_metrics;
}

std::shared_ptr<Screenshoter> RenderWindow::get_screenshoter()
{
	return m_screenshoter;
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
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	while (!glfwWindowShouldClose(m_glfw_window))
	{
		m_start_cpu_frame_time = std::chrono::high_resolution_clock::now();

		glfwPollEvents();
		glClear(GL_COLOR_BUFFER_BIT);

		m_keyboard_interactor.poll_keyboard_inputs();

		// We're resetting the render each frame if rendering at low resolution
		m_render_dirty |= render_settings.render_low_resolution == 1;
		if (m_render_dirty)
			reset_render();

		if (m_application_settings->auto_sample_per_frame && m_samples_per_second > 0)
			render_settings.samples_per_frame = std::min(std::max(1, static_cast<int>(m_samples_per_second / 20.0f)), 10000);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		render();

		float blend_override = -1.0f;
		if (m_application_settings->enable_denoising)
		{
			// Evaluating all the conditions for whether or not we want to denoise
			// the current color framebuffer and whether or not we want to display
			// the denoised framebuffer to the viewport (we may want NOT to display
			// the denoised framebuffer if we're only denoising at the target
			// sample count and we haven't reached the target sample count yet. That's
			// just one example)

			// Do we want to denoise only when reaching the target sample count?
			bool denoise_at_target = m_application_settings->denoise_at_target_sample_count;
			// Have we reached the target sample count? This is going to evaluate to true
			// if the sample target 'max_sample_count' is 0 but that's fine I guess
			bool target_sample_reached = denoise_at_target && render_settings.sample_number >= m_application_settings->max_sample_count;
			// Have we not reached the target sample count while only wanting denoising at target sample count?
			bool target_sample_not_reached = denoise_at_target && render_settings.sample_number < m_application_settings->max_sample_count;
			// Have we rendered enough samples since last time we denoised that we need to denoise?
			bool sample_skip_threshold_reached = !denoise_at_target && (render_settings.sample_number - std::max(0, m_application_settings->last_denoised_sample_count) >= m_application_settings->denoiser_sample_skip);

			bool need_denoising = false;
			bool display_noisy = false;

			// Denoise if:
			//	- We have reached the target sample count
			//	- We have rendered enough samples since the last denoise step that we need to denoise again
			//	- We're not denoising if we're interacting (moving the camera)
			need_denoising |= target_sample_reached;
			need_denoising |= sample_skip_threshold_reached;
			need_denoising &= !is_interacting();

			// Display the noisy framebuffer if: 
			//	- We want to denoise only at target sample count but haven't reached it yet
			//	- We want to denoise every m_application_settings->denoiser_sample_skip samples
			//		but we haven't even reached that number yet. We're displaying the noisy framebuffer in the meantime
			//	- We're moving the camera
			display_noisy |= target_sample_not_reached;
			display_noisy |= !sample_skip_threshold_reached && m_application_settings->last_denoised_sample_count == -1;
			display_noisy |= is_interacting();

			if (need_denoising)
			{
				std::shared_ptr<OpenGLInteropBuffer<float3>> normals_buffer = nullptr;
				std::shared_ptr<OpenGLInteropBuffer<ColorRGB>> albedo_buffer = nullptr;

				if (m_application_settings->denoiser_use_normals)
					normals_buffer = m_renderer->get_denoiser_normals_AOV_buffer();

				if (m_application_settings->denoiser_use_albedo)
					albedo_buffer = m_renderer->get_denoiser_albedo_AOV_buffer();

				m_denoiser->denoise(m_renderer->get_color_framebuffer(), normals_buffer, albedo_buffer);
				m_denoiser->copy_denoised_data_to_buffer(m_renderer->get_denoised_framebuffer());

				m_application_settings->last_denoised_sample_count = render_settings.sample_number;
			}

			if (display_noisy)
				// We need to display the noisy framebuffer so we're forcing the blending factor to 0.0f to only
				// choose the first view out of the two that are going to be blend (and the first view is the noisy view)
				blend_override = 0.0f;
		}
		
		switch (m_application_settings->display_view)
		{
		case DisplayView::DENOISED_BLEND:
			display_blend(m_renderer->get_color_framebuffer(), m_renderer->get_denoised_framebuffer(), blend_override);
			break;

		case DisplayView::DISPLAY_NORMALS:
			display(m_renderer->get_denoiser_normals_AOV_buffer());
			break;

			/*case DisplayView::DISPLAY_DENOISED_NORMALS:
				m_denoiser->denoise_normals();
				display(m_denoiser->get_denoised_normals_pointer());
				break;*/

		case DisplayView::DISPLAY_ALBEDO:
			display(m_renderer->get_denoiser_albedo_AOV_buffer());
			break;

			/*case DisplayView::DISPLAY_DENOISED_ALBEDO:
				m_denoiser->denoise_albedo();
				display(m_denoiser->get_denoised_albedo_pointer());
				break;*/

		case DisplayView::ADAPTIVE_SAMPLING_MAP:
			display(m_renderer->get_pixels_sample_count_buffer());
			break;

			/*case DisplayView::ADAPTIVE_SAMPLING_ACTIVE_PIXELS:
				display(m_renderer->get_debug_pixel_active_buffer().download_data().data());
				break;*/

		case DisplayView::DEFAULT:
		default:
			display(m_renderer->get_color_framebuffer());
			break;
		}
		
		m_imgui_renderer.draw_imgui_interface();

		m_stop_cpu_frame_time = std::chrono::high_resolution_clock::now();

		if (!is_rendering_done())
		{
			m_current_render_time += std::chrono::duration_cast<std::chrono::milliseconds>(m_stop_cpu_frame_time - m_start_cpu_frame_time).count();
			m_samples_per_second = 1000.0f / m_renderer->get_sample_time();
		}

		glfwSwapBuffers(m_glfw_window);
	}

	quit();
}

void RenderWindow::render()
{
	if (!is_rendering_done())
	{
		m_renderer->render();
		increment_sample_number();
		m_renderer->get_render_settings().frame_number++;
	}
	else
		// Sleeping so that we don't burn the CPU (and GPU because it'll have to do the display)
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

void RenderWindow::display(const void* data)
{
	DisplayTextureType texture_1_type = m_display_texture_1.second;
	GLenum format = texture_1_type.get_gl_format();
	GLenum type = texture_1_type.get_gl_type();

	upload_data_to_display_texture(m_display_texture_1.first, data, format, type);
	update_program_uniforms(m_active_display_program);

	// Binding an empty VAO here (empty because we're hardcoding our full-screen quad vertices
	// in our vertex shader) because this is required on NVIDIA drivers
	glBindVertexArray(m_vao);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

void RenderWindow::quit()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glDeleteVertexArrays(1, &m_vao);
	glDeleteTextures(1, &m_display_texture_1.first);
	glDeleteTextures(1, &m_display_texture_2.first);

	std::exit(0);
}
