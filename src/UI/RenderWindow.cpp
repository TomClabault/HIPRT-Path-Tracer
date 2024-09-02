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

// TODOs ReSTIR DI
// - add envmap sampling to light samples with a probability (refactor envmap sampling in eval, sample and PDF functions)
// - add second bounce direct light sampling strategy in imgui
// - add sample rotation in spatial reuse to imgui
// - add hammersley usage or not imgui for spatial reuse
// - neighbor similiraty tests in spatial reuse, roughness, normal, depth
// - pairwise mis
// - allocate / deallocate restir reservoirs if using / not using restir
// - feature to disable ReSTIR after a certain percentage of convergence --> we don't want to pay the full price of resampling and everything only for a few difficult isolated pixels (especially true with adaptive sampling where neighbors don't get sampled --> no new samples added to their reservoir --> no need to resample)
// - test/fix sampling lights inside dielectrics with ReSTIR DI
// - do not do initial candidates / spatial reuse and everything if the pixel is inactive (adaptive sampling)
// - driver crash on the white room love
// - camera ray jittering causes dark lines and darkens glossy reflections
// - multiple spatial reuse passes destroy glossy reflections
// - multiple spatial reuse passes + accumulate = black
// - refactor temporal reuse to avoid hardcoded for loop on 2 iterations --> just unloop and make things clean by pre-reading the temporal neighbor reservoir instead of re-reading it multiple times
// - m cap at 0 in ImGui breaks the render because of infinite M growth --> hardcap M to something like ~1000000 or something
// - temporal reprojection at grazing angles broken
// - different M cap for glossy surfaces ?
// - possibility to use visibility reuse at the end of each spatial pass
// - temporal reservoirs reset button in ImGui

// TODO bugs:
// - memory leak with OpenGL when resizing the window?
// - memory leak when launching kernels? Does that happen also on Linux or is it only due to AMD Windows drivers?
// - playing with the pixel noise threshold eventually leaves it at 4000/2000000 for example, the counter doesn't reset properly
// - pixels converged count sometimes goes above 100%
// - take transmission color into account when direct sampling a light source that is inside a volume
// - denoiser AOVs not accounting for transmission correctly since Disney 
//	  - same with perfect reflection
// - heatmap with adaptive sampling and only pixel stopnoise threshold not displaying the same heatmap (particularly in shadows in the white room)
// - Start render without adaptive sampling --> enable pixel noise threshold --> the convergence counter is broken and starts from when we enabled pixel noise threshold instead of taking all pixels that have converged into account
// - no accumulation + denoiser + 1 SPP max = denoiser running full blow but shouldn't since the input image doesn't change because of the 1SPP max. More generally, this happens also with accumulation and it just seems that the denoiser still runs even when the renderer has reached the maximum amount of SPP
// - GPU limiter not working when interacting?



// TODO Code Organization:
// - Use ShowHelpMarker() for tooltips in ImGui
// - refactor SimplifiedRendererMaterial class
// - fork HIPRT and remove the encryption thingy that slows down kernel compilation on NVIDIA
// - cleanup RIS reservoir with all the BSDF stuff
// - denoiser albedo and normals still useful now that we have the GBuffer?
// - make a function get_camera_ray that handles pixel jittering
// - use simplified material everywhere in the BSDF etc... because we don't need the texture indices of the full material at this point
// - we don't need the full HitInfo 'closest_hit_info' structure everywhere, only the inter point and the two normals for the most part so maybe have a simplified structure 
// - only the material index can be stored in the pixel states ofthe wavefront path tracer, don't need to store the whole material
// - refactor envmap to have a sampling & eval function
// - Use HIPRT with CMake as a subdirectory (available soon)
// - free denoiser buffers if not using denoising



// TODO Features:
// - Reuse MIS BSDF sample as path next bounce if the ray didn't hit anything
// - RIS: do no use BSDF samples for rough surfaces (have a BSDF ray roughness treshold basically
//		We may have to do something with the lobes of the BSDF specifically for this one. A coated diffuse cannot always ignore light samples for example because the diffuse lobe benefits from light samples even if the surface is not smooth (coating) 
// - Whole scene BSDF overrider
// - support stochastic alpha transparency
// - enable samples per frame even when not accumulating
// - maybe allow not resetting ReSTIR buffers while accumulation is on and camera has moved? Probably gives bad results but why not allow it for testing purposes?
// - shadow terminator issue on sphere low smooth scene
// - use HIP/CUDA graphs to reduce launch overhead
// - keep compiling kernels in the background after application has started to cache the most common kernel options on disk
// - linear interpolation function for the parameters of the BSDF
// - compensated importance sampling of envmap
// - have pixel jittering disablable
// - have accumulation disablable
// - have render low resolution when moving disablable
// - multiple GLTF, one GLB for different point of views per model
// - can we do direct lighting + take emissive at all bounces but divide by 2 to avoid double taking into account emissive lights? this would solve missing caustics
// - improve performance by only intersecting the selected emissive triangle with the BSDF ray when multiple importance sampling, we don't need a full BVH traversal at all
// - If could not load given scene file, fallback to cornell box instead of not continuing
// - CTRL + mouse wheel for zoom in viewport, CTRL click reset zoom
// - add clear shader cache in ImGui
// - adapt number of light samples in light sampling routines based on roughness of the material --> no need to sample 8 lights in RIS for perfectly specular material + use ray ballot for that because we don't want to reduce light rays unecessarily if one thread of the warp is going to slow everyone down anyways
// - UI scaling in ImGui
// - clay render
// - Scale all emissive in the scene in the material editor
// - Kahan summation for weighted reservoir sampling?
// - build BVHs one by one to avoid big memory spike? but what about BLAS performance cost?
// - play with SBVH building parameters alpha/beta for memory/performance tradeoff + ImGui for that
// - ability to change the color of the heatmap shader in ImGui
// - ray statistics with filter functions
// - filter function for base color alpha / alpha transparency = better performance
// - do not store alpha from envmap
// - fixed point 18b RGB for envmap? 70% size reduction compared to full size. Can't use texture sampler though. Is not using a sampler ok performance-wise? --> it probably is since we're probably memory lantency bound, not memory bandwidth
// - look at blender cycles "medium contrast", "medium low constract", "medium high", ...
// - normal mapping strength
// - blackbody light emitters
// - ACES mapping
// - better post processing: contrast, low, medium, high exposure curve
// - bloom post processing
// - BRDF swapper ImGui : Disney, Lambertian, Oren Nayar, Cook Torrance, Perfect fresnel dielectric reflect/transmit
// - choose disney diffuse model (disney, lambertian, oren nayar)
// - Cool colored thread-safe logger singleton class --> loguru lib
// - portal envmap sampling --> choose portals with ImGui
// - recursive trace through transmissive / reflective materials for caustics
// - find a way to not fill the texcoords buffer for meshes that don't have textures
// - pack RendererMaterial informations such as texture indices (we can probably use 16 bit for a texture index --> 2 texture indices in one 32 bit register)
// - use 8 bit textures for material properties instead of float
// - use fixed point 8 bit for materials parameters in [0, 1], should be good enough
// - log size of buffers used: vertices, indices, normals, ...
// - log memory size of buffers used: vertices, indices, normals, ...
// - display active pixels adaptive sampling
// - able / disable normal mapping
// - use only one channel for material property texture to save VRAM
// - Remove vertex normals for meshes that have normal maps and save VRAM
// - texture compression
// - float compression for render buffers?
// - Exporter (just serialize the scene to binary file and have a look at how to do backward compatibility)
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
// - Better ray origin offset to avoid self intersections
// - Realistic Camera Model
// - Focus blur
// - Flakes BRDF (maybe look at OSPRay implementation for a reference ?)
// - ImGuizmo for moving objects in the scene
// - Paths roughness regularization
// - choose denoiser quality in imgui
// - try async buffer copy for the denoiser (maybe run a kernel to generate normals and another to generate albedo buffer before the path tracing kernel to be able to async copy while the path tracing kernel is running?)
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

void glfw_window_resized_callback(GLFWwindow* window, int width, int height)
{
	int new_width_pixels, new_height_pixels;
	glfwGetFramebufferSize(window, &new_width_pixels, &new_height_pixels);


	if (new_width_pixels == 0 || new_height_pixels == 0)
		// This probably means that the application has been minimized, we're not doing anything then
		return;
	else
	{
		// We've stored a pointer to the RenderWindow in the "WindowUserPointer" of glfw
		RenderWindow* render_window = reinterpret_cast<RenderWindow*>(glfwGetWindowUserPointer(window));
		render_window->resize_frame(width, height);
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

RenderWindow::RenderWindow(int width, int height, std::shared_ptr<HIPRTOrochiCtx> hiprt_oro_ctx) : m_viewport_width(width), m_viewport_height(height)
{
	init_glfw(width, height);
	init_gl(width, height);
	init_imgui();

	m_renderer = std::make_shared<GPURenderer>(hiprt_oro_ctx);
	m_renderer->resize(width, height);

	m_application_settings = std::make_shared<ApplicationSettings>();
	// Disabling auto samples per frame is accumulation is OFF
	m_application_settings->auto_sample_per_frame = m_renderer->get_render_settings().accumulate ? m_application_settings->auto_sample_per_frame : false;
	m_application_state = std::make_shared<ApplicationState>();

	m_display_view_system = std::make_shared<DisplayViewSystem>(m_renderer, this);

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

	m_imgui_renderer = std::make_shared<ImGuiRenderer>();
	m_imgui_renderer->set_render_window(this);

	// Making the render dirty to force a cleanup at startup
	m_application_state->render_dirty = true;
}

RenderWindow::~RenderWindow()
{
	glfwDestroyWindow(m_glfw_window);
	glfwTerminate();
}

void RenderWindow::init_glfw(int width, int height)
{
	if (!glfwInit())
	{
		std::cerr << "Could not initialize GLFW..." << std::endl;
		int trash = std::getchar();

		std::exit(1);
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);

#ifdef __unix__         
	m_mouse_interactor = std::make_shared<LinuxRenderWindowMouseInteractor>();
#elif defined(_WIN32) || defined(WIN32) 
	m_mouse_interactor = std::make_shared<WindowsRenderWindowMouseInteractor>();
#endif
	m_keyboard_interactor.set_render_window(this);

	m_glfw_window = glfwCreateWindow(width, height, "HIPRT-Path-Tracer", NULL, NULL);
	if (!m_glfw_window)
	{
		std::cerr << "Could not initialize the GLFW window..." << std::endl;
		int trash = std::getchar();

		std::exit(1);
	}

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
		// in the minimized state triggers a queue_resize event with a new size
		// of (0, 0) and getting out of the minimized state triggers a queue_resize
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
		// TODO what about the height changing ?
		resolution_scale = m_application_settings->target_width / (float)pixels_width;

	int new_render_width = std::floor(pixels_width * resolution_scale);
	int new_render_height = std::floor(pixels_height * resolution_scale);

	if (new_render_height == 0 || new_render_width == 0)
		// Can happen if resizing the window to a 1 pixel width/height while having a resolution scaling < 1. 
		// Integer maths will round it down to 0
		return;
	
	m_renderer->synchronize_kernel();
	m_renderer->resize(new_render_width, new_render_height);
	m_denoiser->resize(new_render_width, new_render_height);
	m_denoiser->finalize();
	m_display_view_system->resize(new_render_width, new_render_height);

	m_application_state->render_dirty = true;
}

void RenderWindow::change_resolution_scaling(float new_scaling)
{
	float new_render_width = std::floor(m_viewport_width * new_scaling);
	float new_render_height = std::floor(m_viewport_height * new_scaling);

	m_renderer->synchronize_kernel();
	m_renderer->resize(new_render_width, new_render_height);
	m_denoiser->resize(new_render_width, new_render_height);
	m_denoiser->finalize();
	m_display_view_system->resize(new_render_width, new_render_height);
}

int RenderWindow::get_width()
{
	return m_viewport_width;
}

int RenderWindow::get_height()
{
	return m_viewport_height;
}

bool RenderWindow::is_interacting()
{
	return m_mouse_interactor->is_interacting() || m_keyboard_interactor.is_interacting();
}

RenderWindowKeyboardInteractor& RenderWindow::get_keyboard_interactor()
{
	return m_keyboard_interactor;
}

std::shared_ptr<RenderWindowMouseInteractor> RenderWindow::get_mouse_interactor()
{
	return m_mouse_interactor;
}

std::shared_ptr<ApplicationSettings> RenderWindow::get_application_settings()
{
	return m_application_settings;
}

std::shared_ptr<DisplayViewSystem> RenderWindow::get_display_view_system()
{
	return m_display_view_system;
}

void RenderWindow::update_renderer_view_translation(float translation_x, float translation_y, bool scale_translation)
{
	if (scale_translation)
	{
		translation_x *= m_application_state->last_delta_time_ms / 1000.0f;
		translation_y *= m_application_state->last_delta_time_ms / 1000.0f;

		translation_x *= m_renderer->get_camera().camera_movement_speed;
		translation_y *= m_renderer->get_camera().camera_movement_speed;
	}

	if (translation_x == 0.0f && translation_y == 0.0f)
		return;

	m_application_state->render_dirty = true;

	glm::vec3 translation = glm::vec3(translation_x, translation_y, 0.0f);
	m_renderer->translate_camera_view(translation);
}

void RenderWindow::update_renderer_view_rotation(float offset_x, float offset_y)
{
	m_application_state->render_dirty = true;

	float rotation_x, rotation_y;

	rotation_x = offset_x / m_viewport_width * 2.0f * M_PI / m_application_settings->view_rotation_sldwn_x;
	rotation_y = offset_y / m_viewport_height * 2.0f * M_PI / m_application_settings->view_rotation_sldwn_y;

	m_renderer->rotate_camera_view(glm::vec3(rotation_x, rotation_y, 0.0f));
}

void RenderWindow::update_renderer_view_zoom(float offset, bool scale_delta_time)
{
	if (scale_delta_time)
		offset *= m_application_state->last_delta_time_ms / 1000.0f;
	offset *= m_renderer->get_camera().camera_movement_speed;

	if (offset == 0.0f)
		return;

	m_application_state->render_dirty = true;

	m_renderer->zoom_camera_view(offset);
}

bool RenderWindow::is_rendering_done()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	bool rendering_done = false;

	// No more active pixels (in the case of adaptive sampling for example)
	rendering_done |= !m_renderer->get_status_buffer_values().one_ray_active;

	// All pixels have converged to the noise threshold given
	float proportion_converged;
	proportion_converged = m_renderer->get_status_buffer_values().pixel_converged_count / static_cast<float>(m_renderer->m_render_resolution.x * m_renderer->m_render_resolution.y);
	proportion_converged *= 100.0f; // To human-readable percentage as used in the ImGui interface
	rendering_done |= proportion_converged > render_settings.stop_pixel_percentage_converged && render_settings.stop_pixel_noise_threshold > 0.0f;

	// Max sample count
	rendering_done |= (m_application_settings->max_sample_count != 0 && render_settings.sample_number + 1 > m_application_settings->max_sample_count);

	// Max render time
	float render_time_ms = m_application_state->current_render_time_ms / 1000.0f;
	rendering_done |= (m_application_settings->max_render_time != 0.0f && render_time_ms >= m_application_settings->max_render_time);

	// If we are at 0 samples, this means that the render got resetted and so
	// the render is not done
	rendering_done &= render_settings.sample_number > 0;

	return rendering_done;
}

void RenderWindow::reset_render()
{
	m_application_settings->last_denoised_sample_count = -1;

	m_application_state->samples_per_second = 0.0f;
	m_application_state->current_render_time_ms = 0.0f;
	m_application_state->render_dirty = false;

	m_renderer->reset();
}

void RenderWindow::set_render_dirty(bool render_dirty)
{
	m_application_state->render_dirty = render_dirty;
}

float RenderWindow::get_current_render_time()
{
	return m_application_state->current_render_time_ms;
}

float RenderWindow::get_samples_per_second()
{
	return m_application_state->samples_per_second;
}

float RenderWindow::compute_samples_per_second()
{
	float samples_per_frame = m_renderer->get_render_settings().do_render_low_resolution() ? 1.0f : m_renderer->get_render_settings().samples_per_frame;

	// Frame time divided by the number of samples per frame
	// 1 sample per frame assumed if rendering at low resolution
	if (m_application_state->last_GPU_submit_time > 0)
	{
		uint64_t current_time = glfwGetTimerValue();
		float difference_ms = (current_time - m_application_state->last_GPU_submit_time) / static_cast<float>(glfwGetTimerFrequency()) * 1000.0f;

		return 1000.0f / (difference_ms / samples_per_frame);
	}
	else
		return 0.0f;
}

float RenderWindow::compute_GPU_stall_duration()
{
	if (m_application_settings->GPU_stall_percentage > 0.0f)
	{
		float last_frame_time = m_renderer->get_last_frame_time();
		float stall_duration = last_frame_time * (1.0f / (1.0f - m_application_settings->GPU_stall_percentage / 100.0f)) - last_frame_time;

		return stall_duration;
	}

	return 0.0f;
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

std::shared_ptr<ImGuiRenderer> RenderWindow::get_imgui_renderer()
{
	return m_imgui_renderer;
}

void RenderWindow::run()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	uint64_t last_second_start = glfwGetTimerValue();

	uint64_t time_frequency = glfwGetTimerFrequency();
	uint64_t frame_start_time = 0;
	while (!glfwWindowShouldClose(m_glfw_window))
	{
		frame_start_time = glfwGetTimerValue();

		glfwPollEvents();
		glClear(GL_COLOR_BUFFER_BIT);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		m_application_state->render_dirty |= is_interacting();
		m_application_state->render_dirty |= m_application_state->interacting_last_frame != is_interacting();

		render();
		m_display_view_system->display();
		
		m_imgui_renderer->rescale_ui();
		m_imgui_renderer->draw_imgui_interface();

		glfwSwapBuffers(m_glfw_window);

		float delta_time_ms = (glfwGetTimerValue() - frame_start_time) / static_cast<float>(time_frequency) * 1000.0f;
		m_application_state->last_delta_time_ms = delta_time_ms;

		if (!is_rendering_done())
			m_application_state->current_render_time_ms += delta_time_ms;
		m_keyboard_interactor.poll_keyboard_inputs();
	}

	quit();
}

void RenderWindow::render()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	// Boolean local to this function to remember whether or not we need to upload
	// the frame result to OpenGL for displaying
	static bool buffer_upload_necessary = true;

	if (m_renderer->frame_render_done())
	{
		// ------
		// Everything that is in there is synchronous with the renderer
		// ------
		m_renderer->copy_status_buffers();

		if (m_application_state->GPU_stall_duration_left > 0 && !is_rendering_done())
		{
			// If we're stalling the GPU.
			// We're whether or not the rendering is done because we don't need to
			// stall the GPU if the rendering is done

			if (m_application_state->GPU_stall_duration_left > 0.0f)
				// Updating the duration left to stall the GPU.
				m_application_state->GPU_stall_duration_left -= m_application_state->last_delta_time_ms;
		}
		else if (!is_rendering_done() || m_application_state->render_dirty)
		{
			// We can unmap the renderer's buffers so that OpenGL can use them when displaying
			m_renderer->unmap_buffers();
			// Update the display view system so that the display view is changed to the
			// one that we want to use (in the DisplayViewSystem's queue)
			m_display_view_system->update_selected_display_view();
			
			// Denoising to fill the buffers with denoised data (if denoising is enabled)
			denoise();

			// We upload the data to the OpenGL textures for displaying
			m_display_view_system->upload_relevant_buffers_to_texture();
			// We want the next frame to be displayed with the same wants_render_low_resolution setting
			// as it was queued with. This is only useful for first frames when getting in low resolution
			// (when we start moving the camera for example) or first frames when getting out of low resolution
			// (when we stop moving the camera). In such situations, the last kernel launch in the GPU queue is
			// a "first frame" that was queued with the corresponding wants_render_low_resolution (getting in or out of low resolution).
			// and so we want to display it the same way.
			m_display_view_system->set_render_low_resolution(m_renderer->was_last_frame_low_resolution());
			// Updating the uniforms so that next time we display, we display correctly
			m_display_view_system->update_current_display_program_uniforms();

			// We got a frame rendered --> We can compute the samples per second
			m_application_state->samples_per_second = compute_samples_per_second();

			// Adding the time for *one* sample to the performance metrics counter
			if (!m_renderer->was_last_frame_low_resolution() && m_application_state->samples_per_second > 0.0f)
				// Not adding the frame time if we're rendering at low resolution, not relevant
				m_perf_metrics->add_value(PerformanceMetricsComputer::SAMPLE_TIME_KEY, 1000.0f / m_application_state->samples_per_second);

			render_settings.wants_render_low_resolution = is_interacting();
			if (m_application_settings->auto_sample_per_frame && (render_settings.do_render_low_resolution() || m_renderer->was_last_frame_low_resolution()))
				// Only one sample when low resolution rendering
				render_settings.samples_per_frame = 1;
			else if (m_application_settings->auto_sample_per_frame)
				render_settings.samples_per_frame = std::min(std::max(1, static_cast<int>(m_application_state->samples_per_second / m_application_settings->target_GPU_framerate)), 65536);

			if (m_application_state->render_dirty)
				reset_render();
			m_application_state->interacting_last_frame = is_interacting();
			m_application_state->GPU_stall_duration_left = compute_GPU_stall_duration();
			
			// Otherwise, if we're not stalling, queuing a new frame for the GPU to render
			m_application_state->last_GPU_submit_time = glfwGetTimerValue();
			m_renderer->update();
			m_renderer->render();

			buffer_upload_necessary = true;
		}
		else
		{
			// The rendering is done

			buffer_upload_necessary |= m_display_view_system->update_selected_display_view();

			if (m_application_settings->enable_denoising)
				// We may still want to denoise on the final frame
				if (denoise())
					buffer_upload_necessary = true;

			if (buffer_upload_necessary)
			{
				// Re-uploading only if necessary
				m_display_view_system->upload_relevant_buffers_to_texture();

				buffer_upload_necessary = false;
			}

			m_display_view_system->set_render_low_resolution(m_renderer->was_last_frame_low_resolution());
			// Updating the uniforms if the user touches the post processing parameters
			// or something else (denoiser blend, ...)
			m_display_view_system->update_current_display_program_uniforms();

			// Sleeping so that we don't burn the CPU and GPU
			std::this_thread::sleep_for(std::chrono::milliseconds(3));
		}
	}
}

bool RenderWindow::denoise()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
	m_application_settings->blend_override = -1.0f;

	if (m_application_settings->enable_denoising)
	{
		// Evaluating all the conditions for whether or not we want to denoise
		// the current color framebuffer and whether or not we want to display
		// the denoised framebuffer to the viewport (we may want NOT to display
		// the denoised framebuffer if we're only denoising when the render is done
		// but the render isn't done yet. That's just one example)



		// ---- Utility variables ----
		// Do we want to denoise only when reaching the rendering is done?
		bool denoise_when_done = m_application_settings->denoise_when_rendering_done;
		// Is the rendering done?
		bool rendering_done = is_rendering_done();
		// Whether or not we've already denoise the framebuffer after the rendering is done.
		// This is to avoid denoising again and again the framebuffer when the rendering is done (because that would just be using the machine for nothing)
		bool final_frame_denoised_already = !m_application_settings->denoiser_settings_changed && rendering_done && m_application_settings->last_denoised_sample_count == render_settings.sample_number;



		// ---- Conditions for denoising / displaying noisy ----
		// - Is the rendering done 
		// - And we only want to denoise when the rendering is done
		// - And we haven't alraedy denoised the final frame
		bool denoise_rendering_done = rendering_done && denoise_when_done && !final_frame_denoised_already;
		// Have we rendered enough samples since last time we denoised that we need to denoise again?
		bool sample_skip_threshold_reached = !denoise_when_done && (render_settings.sample_number - std::max(0, m_application_settings->last_denoised_sample_count) >= m_application_settings->denoiser_sample_skip);
		// We're also going to denoise if we changed the denoiser settings
		// (because we need to denoise to reflect the new settings)
		bool denoiser_settings_changed = m_application_settings->denoiser_settings_changed;




		bool need_denoising = false;
		bool display_noisy = false;

		// Denoise if:
		//	- The render is done and we're denoising when the render 
		//	- We have rendered enough samples since the last denoise step that we need to denoise again
		//	- We're not denoising if we're interacting (moving the camera)
		need_denoising |= denoise_rendering_done;
		need_denoising |= sample_skip_threshold_reached;
		need_denoising |= denoiser_settings_changed;
		need_denoising &= !is_interacting();

		// Display the noisy framebuffer if: 
		//	- We only denoise when the rendering is done but it isn't done yet
		//	- We want to denoise every m_application_settings->denoiser_sample_skip samples
		//		but we haven't even reached that number yet. We're displaying the noisy framebuffer in the meantime
		//	- We're moving the camera
		display_noisy |= !rendering_done && denoise_when_done;
		display_noisy |= !sample_skip_threshold_reached && m_application_settings->last_denoised_sample_count == -1 && !rendering_done;
		display_noisy |= is_interacting();

		if (need_denoising)
		{
			std::shared_ptr<OpenGLInteropBuffer<float3>> normals_buffer = nullptr;
			std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> albedo_buffer = nullptr;

			if (m_application_settings->denoiser_use_normals)
				normals_buffer = m_renderer->get_denoiser_normals_AOV_buffer();

			if (m_application_settings->denoiser_use_albedo)
				albedo_buffer = m_renderer->get_denoiser_albedo_AOV_buffer();

			auto start = std::chrono::high_resolution_clock::now();
			m_denoiser->denoise(m_renderer->get_color_framebuffer(), normals_buffer, albedo_buffer);
			auto stop = std::chrono::high_resolution_clock::now();

			m_denoiser->copy_denoised_data_to_buffer(m_renderer->get_denoised_framebuffer());

			m_application_settings->last_denoised_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
			m_application_settings->last_denoised_sample_count = render_settings.sample_number;
		}

		if (display_noisy)
			// We need to display the noisy framebuffer so we're forcing the blending factor to 0.0f to only
			// choose the first view out of the two that are going to be blend (and the first view is the noisy view)
			m_application_settings->blend_override = 0.0f;

		m_application_settings->denoiser_settings_changed = false;

		return need_denoising && !display_noisy;
	}

	return false;
}

void RenderWindow::quit()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}
