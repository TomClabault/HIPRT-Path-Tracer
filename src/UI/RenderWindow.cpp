/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompiler.h"
#include "Scene/SceneParser.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "UI/RenderWindow.h"
#include "UI/Interaction/LinuxRenderWindowMouseInteractor.h"
#include "UI/Interaction/WindowsRenderWindowMouseInteractor.h"
#include "Utils/Utils.h"

#include <functional>
#include <iostream>

#include "stb_image_write.h"

// GPUKernelCompiler for waiting on threads currently reading files on disk
extern GPUKernelCompiler g_gpu_kernel_compiler;
extern ImGuiLogger g_imgui_logger;

// TODOs  performance improvements branch:
// - texcoords packing --> bring everyone in [0, 1] on the CPU and then encode as uchar?
// - nested dielectrics in shared mem or global memory, it's pretty slow  in intersect.h
// - pack ray payload for register usage reduction
// - pack envmap?
// - check that sheen still works with  new texture handling
// - reuse MIS bounce
// - texture compression
// - wavefront path tracing
// - investigate where the big register usage comes from --> split shaders there?
// - split shaders for material specifics?
// - use wavefront path tracing to evaluate direct  lighting, envmap and BSDF sample in parallel
// - start shooting camera rays for frame N+1 during frame N?
// - use the fact that some values are already computed in bsdf_sample to pass them to bsdf_eval in a big BSDFStateStructure or something to avoid recomputing
// - pack ray payload and other things?
// - bsdf sampling proba do  =not use array[] for CDF
// - improve alpha testing
// - upload partial materials when a material is modified instead  of reuploading everything
// - NEE++
// - schlick fresnel in many places?
// - compaction -  https://github.com/microsoft/directxshadercompiler/wiki/wave-intrinsics#example
// - flags to enable energy preservation per material
// - disable energy conservation on smooth glass / smooth metal
// - per material light sampling/BSDF sampling: smooth glass / metal don't need light sampling
// - launch bounds?
// - thread group size optimization?
// - Do we need  to keep the whole code  bloat for the packed material usage since it doesn't seem to be changing anything
// - SoA instead of AoS
// - superfluous sample() call on the last bounce?
// - perfect reflection and refractions fast path
// - double buffering of frames in general to better keep the GPU occupied?
// - pack envmap (maybe not do it if the max radiance  of the envmap is too high to avoid losing precision)
// - remove unused denoiser buffers if not using the denoiser
// - 

// TODO demos:
// new oren nayar BRDF: EON
// clearcoat capabitilies

// TODOs ongoing
// - limit UI speed because it actually uses some resources (maybe Vsync or something) or does it?
// - smarter shader cache (hints to avoid using all kernel options when compiling a kernel? We know that Camera ray doesn't care about direct lighting strategy for example)
// - use self bit packing (no bitfields) for nested dielectrics because bitfields are implementation dependent in size, that's bad --> We don't get our nice packing with every compiler
// - for LTC sheen lobe, have the option to use either SGGX volumetric sheen or approximation precomputed LTC data
// - --help on the commandline
// - Search for textures next to the GLTF file location
// - Normal mapping seems broken again, light rays going under the surface... p1 env light

// TODO known bugs / incorrectness:
// - take transmission color into account when direct sampling a light source that is inside a volume: leave that for when implement volumes?
// - denoiser AOVs not accounting for transmission correctly since Disney  BSDF
//	  - same with perfect reflection
// - when using a BSDF override, transmissive materials keep their dielectric priorities and this can mess up shadow rays and intersections in general if the BSDF used for the override doesn't support transmissive materials
// - threadmanager: what if we start a thread with a dependency A on a thread that itself has a dependency B? we're going to try join dependency A even if thread with dependency on B hasn't even started yet --> joining nothing --> immediate return --> should have waited for the dependency but hasn't
// - When checking "Enable denoiser", it always denoises once immediately even if "denoise only when render done" is checked
// - Thin-film interference energy conservation/preservation is broken with "strong BSDF energy conservation" --> too bright (with transmission at 1.0f), even with film thickness == 0.0f
// - When overriding the base color for example in the global material overrider, if we then uncheck the base color override to stop overriding the base color, it returns the material to its very default base color  (the one  read from the scene file) instead of  returning it to what the user may have modified up to that point
// - Some weird interaction between the specular layer and the coat layer when both darkening are enabled. Coat 0.5f strength is brighter than coat 0.0f

// TODO Code Organization:
// - init opengl context and all that expensive stuff (compile kernels too) while the scene is being parsed
// - do not pass so many arguments to kernels everytime: make a "KernelArguments" folder in the source files with one file that contains the arguments needed for a kernel: ReSTIR_DI_InitialCandidatesArguments, ReSTIR_DI_SpatialReuseArguments, ...
// - cleanup RIS reservoir with all the BSDF stuff
// - only recompile relevant kernels in GPURenderer::recompile_kernels (i.e. not restir if not using restir for example)
// - denoiser albedo and normals still useful now that we have the GBuffer?
// - make a function get_camera_ray that handles pixel jittering
// - we don't need the full HitInfo 'closest_hit_info' structure everywhere, only the inter point and the two normals for the most part so maybe have a simplified structure 
// - only the material index can be stored in the pixel states of the wavefront path tracer, don't need to store the whole material (is that correct though? Because then we need to re-evaluate the textures at the hit point)
// - use 3x3 matrix for envmap matrices
// - free denoiser buffers if not using denoising
// - use a proper GLTF loader because ASSIMP isn't good, poor support of the GLTF spec
// - refactor ImGuiRenderer in several sub classes that each draw a panel
// - refactor closestHitTypes with something like 'hiprtGeomTraversalClosestHitType<UseSharedStackBVHTraversal>' to avoid the big #if #elif blocks



// TODO Features:
// - implement ideas of https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
// - software opacity micromaps
// - cache opacity of materials textures? --> analyze the texture when loading it from the texture and if there isn't a single transparent pixel, then we know that we won't have to fetch the material / texture in the alpha test filter function because the alpha is going to be 1.0f anyways
// - simpler BSDF for indirect bounces as a biased option for performance?
// - limit secondary bounces ray distance: objects far away won't contribute much to what the camera sees so shortening the rays should be okay?
// - limit direct lighting occlusion distance: maybe stochastically so that we get a falloff instead of a hard cut where an important may not contribute anymore
//		- for maximum ray length, limit that length even more for indirect bounces and even more so if the ray is far away from the camera (beware of mirrors in the scene which the camera can look into and see a far away part of the scene where light could be very biased)
// - only update the display every so often if accumulating because displaying is expensive (especially at high resolution) on AMD drivers at least
// - how to help with shaders combination compilation times? upload bitcodes that ** I ** compile locally to Github? Change some #if to if() where this does not increase register pressure also.
// - use bare variables for principled_bsdf_sample CDF[] because local arrays are bad on AMD GPUs
// - pack HDR as color as 9/9/9/5 RGBE? https://github.com/microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/Shaders/PixelPacking_RGBE.hlsli
// - next event estimation++? --> 2023 paper improvement
// - ideas of https://pbr-book.org/4ed/Light_Sources/Further_Reading for performance
// - envmap visibility cache? 
// - russian roulette on light sampling based on light contribution?
// - Exploiting Visibility Correlation in Direct Illumination
// - Progressive Visibility Caching for Fast Indirect Illumination
// - performance/bias tradeoff by ignoring alpha tests (either for global rays or only shadow rays) after N bounce?
// - performance/bias tradeoff by ignoring direct lighting occlusion after N bounce? --> strong bias but maybe something to do by reducing the length of shadow rays instead of just hard-disabling occlusion
// - energy conserving Oren Nayar: https://mimosa-pudica.net/improved-oren-nayar.html#images
// - GMoN estimator fireflies reduction
// - experiment with a feature that ignores really dark pixel in the variance estimation of the adaptive 
//		sampling because it seems that very dark areas in the image are always flagged as very 
//		noisy / very high variance and they take a very long time to converge (always red on the heatmap) 
//		even though they are very dark regions and we don't even noise in them. If our eyes can't see 
//		the noise, why bother? Same with very bright regions
// - pack material parameters that are between 0 and 1 into 8 bits, 1/256 is enough precision for parameters in 0-1
// - Reuse miss BSDF ray on the last bounce to sample envmap with MIS
// - Reuse MIS BSDF sample as path next bounce if the ray didn't hit anything
// - Reuse second bounce BSDF sampled direction for light sampling in MIS if we bounced in a light ?
// - RIS: do no use BSDF samples for rough surfaces (have a BSDF ray roughness treshold basically)
//		We may have to do something with the lobes of the BSDF specifically for this one. A coated diffuse cannot always ignore light samples for example because the diffuse lobe benefits from light samples even if the surface is not smooth (coating) 
// - have a light BVH for intersecting light triangles only: useful when we want to know whether or not a direction could have be sampled by the light sampler: we don't need to intersect the whole scene BVH, just the light geometry, less expensive
// - shadow terminator issue on sphere low smooth scene: [Taming the Shadow Terminator], Matt Jen-Yuan Chiang, https://github.com/aconty/aconty/blob/main/pdf/bump-terminator-nvidia2019.pdf
// - use HIP/CUDA graphs to reduce launch overhead
// - linear interpolation (spatial, object space, world space) function for the parameters of the BSDF
// - compensated importance sampling of envmap
// - Product importance sampling envmap: https://github.com/aconty/aconty/blob/main/pdf/fast-product-importance-abstract.pdf
// - multiple GLTF, one GLB for different point of views per model
// - improve performance by only intersecting the selected emissive triangle with the BSDF ray when multiple importance sampling, we don't need a full BVH traversal at all
// - CTRL + mouse wheel for zoom in viewport, CTRL click reset zoom
// - add clear shader cache in ImGui
// - adapt number of light samples in light sampling routines based on roughness of the material --> no need to sample 8 lights in RIS for perfectly specular material + use __any() intrinsic for that because we don't want to reduce light rays unecessarily if one thread of the warp is going to slow everyone down anyways
// - UI scaling in ImGui
// - clay render
// - build BVHs one by one to avoid big memory spike? but what about BLAS performance cost?
// - play with SBVH building parameters alpha/beta for memory/performance tradeoff + ImGui for that
// - ability to change the color of the heatmap shader in ImGui
// - do not store alpha from envmap
// - fixed point 18b RGB for envmap? 70% size reduction compared to full size. Can't use texture sampler though. Is not using a sampler ok performance-wise? --> it probably is since we're probably memory lantency bound, not memory bandwidth
// - look at blender cycles "medium contrast", "medium low constract", "medium high", ...
// - normal mapping strength
// - blackbody light emitters
// - ACES mapping
// - better post processing: contrast, low, medium, high exposure curve
// - bloom post processing
// - BRDF swapper ImGui : Disney, Lambertian, Oren Nayar, Cook Torrance, Perfect fresnel dielectric reflect/transmit
// - Directional albedo sampling weights for the principled BSDF importance sampling. Also, can we do "perfect importance" sampling where we sample each relevant lobe, evaluate them (because we have to evaluate them anyways in eval()) and choose which one is sampled proportionally to its contribution or is it exactly the idea of sampling based on directional albedo?
// - choose principled BSDF diffuse model (disney, lambertian, oren nayar)
// - portal envmap sampling --> choose portals with ImGui
// - find a way to not fill the texcoords buffer for meshes that don't have textures
// - pack CPUMaterial informations such as texture indices (we can probably use 16 bit for a texture index --> 2 texture indices in one 32 bit register)
// - use 8 bit textures for material properties instead of float
// - use fixed point 8 bit for materials parameters in [0, 1], should be good enough
// - log size of buffers used: vertices, indices, normals, ...
// - log memory size of buffers used: vertices, indices, normals, ...
// - able / disable normal mapping
// - use only one channel for material property texture to save VRAM
// - Remove vertex normals for meshes that have normal maps and save VRAM
// - texture compression
// - WUFFS for image loading?
// - float compression for render buffers?
// - Exporter (just serialize the scene to binary file and have a look at how to do backward compatibility)
// - Allow material parameters textures manipulation with ImGui
// - Disable material parameters in ImGui that have a texture associated (since the ImGui slider in this case has no effect)
// - Upload grayscale texture (roughness, specular and other BSDF parameters basically) as one channel to the GPU instead of memory costly RGBA
// - Emissive textures sampling: how to sample an object that has an emissive texture? How to know which triangles of the mesh are covered by the emissive parts of the texture?
// - stream compaction / active thread compaction (ingo wald 2011)
// - sample regeneration
// - Spectral rendering / look at gemstone rendering because they quite a lot of interesting lighting effect to take into account (pleochroism, birefringent, dispersion, ...)
// - structure of arrays instead of arrays of struct relevant for global buffers in terms of performance?
// - data packing in buffer --> use one 32 bit buffer to store multiple information if not using all 32 bits
//		- pack active pixel in same buffer as pixel sample count
// - pack two texture indices in one int for register saving, 65536 (16 bit per index when packed) textures is enough
// - hint shadow rays for better traversal perf on RDNA3?
// - benchmarker to measure frame times precisely (avg, std dev, ...) + fixed random seed for reproducible results
// - alias table for sampling env map instead of log(n) binary search
// - image comparator slider (to have adaptive sampling view + default view on the same viewport for example)
// - thin materials
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
// - feature to disable ReSTIR after a certain percentage of convergence --> we don't want to pay the full price of resampling and everything only for a few difficult isolated pixels (especially true with adaptive sampling where neighbors don't get sampled --> no new samples added to their reservoir --> no need to resample)
// - Better ray origin offset to avoid self intersections --> Use ray TMin
// - Realistic Camera Model
// - Focus blur
// - Flakes BRDF (maybe look at OSPRay implementation for a reference ?)
// - ImGuizmo for moving objects in the scene
// - Paths roughness regularization
// - choose denoiser quality in imgui
// - try async buffer copy for the denoiser (maybe run a kernel to generate normals and another to generate albedo buffer before the path tracing kernel to be able to async copy while the path tracing kernel is running?)
// - write scene details to imgui (nb vertices, triangles, ...)
// - ImGui to choose the BVH flags at runtime and be able to compare the performance
// - ImGui widgets for SBVH / LBVH
// - BVH compaction + imgui checkbox
// - choose env map at runtime imgui
// - choose scene file at runtime imgui
// - lock camera checkbox to avoid messing up when big render in progress
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
		render_window->resize(width, height);
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
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204) 
		return;

	if (id == 131154)
		// NVIDIA specific warning
		// Pixel-path performance warning: Pixel transfer is synchronized with 3D rendering.
		// 
		// Mainly happens when we take a screenshot
		return;

	if (id == 131154)
		// NVIDIA specific warning
		// Pixel-path performance warning: Pixel transfer is synchronized with 3D rendering.
		// 
		// Mainly happens when we take a screenshot
		return;

	std::string source_str;
	std::string type_str;
	std::string severity_str;

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:             source_str = "Source: API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   source_str = "Source: Window System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: source_str = "Source: Shader Compiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     source_str = "Source: Third Party"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     source_str = "Source: Application"; break;
	case GL_DEBUG_SOURCE_OTHER:           source_str = "Source: Other"; break;
	}

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:               type_str = "Type: Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: type_str = "Type: Deprecated Behaviour"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  type_str = "Type: Undefined Behaviour"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         type_str = "Type: Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         type_str = "Type: Performance"; break;
	case GL_DEBUG_TYPE_MARKER:              type_str = "Type: Marker"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          type_str = "Type: Push Group"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           type_str = "Type: Pop Group"; break;
	case GL_DEBUG_TYPE_OTHER:               type_str = "Type: Other"; break;
	}

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         severity_str = "Severity: high"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:       severity_str = "Severity: medium"; break;
	case GL_DEBUG_SEVERITY_LOW:          severity_str = "Severity: low"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: severity_str = "Severity: notification"; break;
	}

	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, 
		"---------------\n"
		"Debug message (%d): %s\n"
		"%s\n%s\n%s\n\n", id, message, source_str.c_str(), type_str.c_str(), severity_str.c_str());

	// The following breaks into the debugger to help pinpoint what OpenGL
	// call errored
	Utils::debugbreak();
}

RenderWindow::RenderWindow(int renderer_width, int renderer_height, std::shared_ptr<HIPRTOrochiCtx> hiprt_oro_ctx) : m_viewport_width(renderer_width), m_viewport_height(renderer_height)
{
	// Adding the size of the windows around the viewport such that these windows
	// have their base size and the viewport has the size the the user has asked for
	// (through the commandline)
	int window_width = renderer_width + ImGuiSettingsWindow::BASE_SIZE;
	int window_height = renderer_height + ImGuiLogWindow::BASE_SIZE;

	init_glfw(window_width, window_height);
	init_gl(renderer_width, renderer_height);
	ImGuiRenderer::init_imgui(m_glfw_window);

	m_renderer = std::make_shared<GPURenderer>(hiprt_oro_ctx);
	m_gpu_baker = std::make_shared<GPUBaker>(m_renderer);

	ThreadManager::add_dependency(ThreadManager::RENDER_WINDOW_RENDERER_INITIAL_RESIZE, ThreadManager::RENDERER_STREAM_CREATE);
	ThreadManager::start_thread(ThreadManager::RENDER_WINDOW_RENDERER_INITIAL_RESIZE, [this, renderer_width, renderer_height]() {
		m_renderer->resize(renderer_width, renderer_height, /* resize interop buffers */ false);
	});
	// We need to resize OpenGL interop buffers on the main thread becaues they
	// need the OpenGL context which is only available to the main thread
	m_renderer->resize_interop_buffers(renderer_width, renderer_height);

	m_application_settings = std::make_shared<ApplicationSettings>();
	// Disabling auto samples per frame is accumulation is OFF
	m_application_settings->auto_sample_per_frame = m_renderer->get_render_settings().accumulate ? m_application_settings->auto_sample_per_frame : false;
	m_application_state = std::make_shared<ApplicationState>();

	ThreadManager::start_thread(ThreadManager::RENDER_WINDOW_CONSTRUCTOR, [this, renderer_width, renderer_height]() {
		m_denoiser = std::make_shared<OpenImageDenoiser>();
		m_denoiser->initialize();
		m_denoiser->resize(renderer_width, renderer_height);
		m_denoiser->set_use_albedo(m_application_settings->denoiser_use_albedo);
		m_denoiser->set_use_normals(m_application_settings->denoiser_use_normals);
		m_denoiser->finalize();

		m_perf_metrics = std::make_shared<PerformanceMetricsComputer>();

		m_imgui_renderer = std::make_shared<ImGuiRenderer>();
		m_imgui_renderer->set_render_window(this);

		// Making the render dirty to force a cleanup at startup
		m_application_state->render_dirty = true;
	});


	// Cannot create that on a thread since it compiles OpenGL shaders
	// which the OpenGL context which is only available to the thread it was created on (the main thread)
	m_display_view_system = std::make_shared<DisplayViewSystem>(m_renderer, this);

	// Same for the screenshoter
	m_screenshoter = std::make_shared<Screenshoter>();
	m_screenshoter->set_renderer(m_renderer);
	m_screenshoter->set_render_window(this);
}

RenderWindow::~RenderWindow()
{
	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Exiting...");

	// Hiding the window to show the user that the app has exited. This is basically only useful if the
	// wait function call below hangs for a while: we don't want the user to see the application
	// frozen in this case. Note that we're *hiding* the window and not *destroying* it because
	// destroying the window also destroys the GL context which may cause crashes is some
	// other part of the app is still using buffers or whatnot
	glfwHideWindow(m_glfw_window);

	// Waiting for all threads that are currently reading from the disk (for compiling kernels in the background)
	// to finish the reading to avoid SEGFAULTING
	g_gpu_kernel_compiler.wait_compiler_file_operations();

	// Waiting for the renderer to finish its frame otherwise
	// we're probably going to close the window / destroy the
	// GL context / etc... while the renderer might still be
	// using so OpenGL Interop buffers --> segfault
	m_renderer->synchronize_kernel();
	// Manually destroying the renderer now before we destroy the GL context
	// glfwDestroyWindow()
	m_renderer = nullptr;
	// Same for the screenshoter
	m_screenshoter = nullptr;
	// Same for the baker
	m_gpu_baker = nullptr;
	// Same for the display view system
	m_display_view_system = nullptr;
	// Same for the imgui renderer
	m_imgui_renderer = nullptr;

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(m_glfw_window);
}

void RenderWindow::init_glfw(int window_width, int window_height)
{
	if (!glfwInit())
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Could not initialize GLFW...");

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

	const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

	m_glfw_window = glfwCreateWindow(window_width, window_height, "HIPRT-Path-Tracer", NULL, NULL);
	if (!m_glfw_window)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Could not initialize the GLFW window...");

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

void RenderWindow::resize(int pixels_width, int pixels_height)
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

		translation_x *= m_renderer->get_camera().camera_movement_speed * m_renderer->get_camera().user_movement_speed_multiplier;
		translation_y *= m_renderer->get_camera().camera_movement_speed * m_renderer->get_camera().user_movement_speed_multiplier;
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

	rotation_x = offset_x / m_viewport_width * M_TWO_PI / m_application_settings->view_rotation_sldwn_x;
	rotation_y = offset_y / m_viewport_height * M_TWO_PI / m_application_settings->view_rotation_sldwn_y;

	// Inverting X and Y here because moving your mouse to the right actually means
	// rotating the camera around the Y axis
	m_renderer->rotate_camera_view(glm::vec3(rotation_y, rotation_x, 0.0f));
}

void RenderWindow::update_renderer_view_zoom(float offset, bool scale_delta_time)
{
	if (scale_delta_time)
		offset *= m_application_state->last_delta_time_ms / 1000.0f;
	offset *= m_renderer->get_camera().camera_movement_speed * m_renderer->get_camera().user_movement_speed_multiplier;

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
	proportion_converged *= 100.0f; // To percentage as used in the ImGui interface

	// We're allowed to stop the render after the given proportion of pixel of the image converged if we're actually
	// using the pixel stop noise threshold feature (enabled + threshold > 0.0f) or if we're using the
	// stop noise threshold but only for the proportion stopping condition (we're not using the threshold of the pixel
	// stop noise threshold feature) --> (enabled & adaptive sampling enabled)
	bool use_proportion_stopping_condition = (render_settings.stop_pixel_noise_threshold > 0.0f && render_settings.enable_pixel_stop_noise_threshold) 
		|| (render_settings.enable_pixel_stop_noise_threshold && render_settings.enable_adaptive_sampling);
	rendering_done |= proportion_converged > render_settings.stop_pixel_percentage_converged && use_proportion_stopping_condition;

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

	m_application_state->current_render_time_ms = 0.0f;
	m_application_state->render_dirty = false;

	m_renderer->reset(m_application_settings);
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

float RenderWindow::get_UI_delta_time()
{
	return m_application_state->last_delta_time_ms;
}

std::shared_ptr<OpenImageDenoiser> RenderWindow::get_denoiser()
{
	return m_denoiser;
}

std::shared_ptr<GPURenderer> RenderWindow::get_renderer()
{
	return m_renderer;
}

std::shared_ptr<GPUBaker> RenderWindow::get_baker()
{
	return m_gpu_baker;
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

	uint64_t time_frequency = glfwGetTimerFrequency();
	uint64_t frame_start_time = 0;
	while (!glfwWindowShouldClose(m_glfw_window))
	{
		frame_start_time = glfwGetTimerValue();

		glfwPollEvents();
		glClear(GL_COLOR_BUFFER_BIT);

		m_application_state->render_dirty |= is_interacting();
		m_application_state->render_dirty |= m_application_state->interacting_last_frame != is_interacting();

		render();
		m_display_view_system->display();
		
		m_imgui_renderer->draw_interface();

		glfwSwapBuffers(m_glfw_window);

		float delta_time_ms = (glfwGetTimerValue() - frame_start_time) / static_cast<float>(time_frequency) * 1000.0f;
		m_application_state->last_delta_time_ms = delta_time_ms;

		if (!is_rendering_done())
			m_application_state->current_render_time_ms += delta_time_ms;
		m_keyboard_interactor.poll_keyboard_inputs();
	}
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
			//// We can unmap the renderer's buffers so that OpenGL can use them for displaying
			m_renderer->unmap_buffers();

			// Update the display view system so that the display view is changed to the
			// one that we want to use (in the DisplayViewSystem's queue)
			m_display_view_system->update_selected_display_view();
			
			// Denoising to fill the buffers with denoised data (if denoising is enabled)
			denoise();

			//// We upload the data to the OpenGL textures for displaying
			m_display_view_system->upload_relevant_buffers_to_texture();

			// We want the next frame to be displayed with the same 'wants_render_low_resolution' setting
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
				update_perf_metrics();

			render_settings.wants_render_low_resolution = is_interacting();
			if (m_application_settings->auto_sample_per_frame && (render_settings.do_render_low_resolution() || m_renderer->was_last_frame_low_resolution()) && render_settings.accumulate)
				// Only one sample when low resolution rendering.
				// Also, we only want to apply this if we're accumulating. If we're not accumulating, 
				// (so we the renderer in "interactive mode" we may want more than 1 sample per frame
				// to experiment
				render_settings.samples_per_frame = 1;
			else if (m_application_settings->auto_sample_per_frame)
				render_settings.samples_per_frame = std::min(std::max(1, static_cast<int>(m_application_state->samples_per_second / m_application_settings->target_GPU_framerate)), 65536);

			m_application_state->interacting_last_frame = is_interacting();
			m_application_state->GPU_stall_duration_left = compute_GPU_stall_duration();
			if (m_application_state->render_dirty)
				reset_render();
			
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

			RendererAnimationState& renderer_animation_state = m_renderer->get_animation_state();
			if (renderer_animation_state.is_rendering_frame_sequence && renderer_animation_state.frames_rendered_so_far < renderer_animation_state.number_of_animation_frames)
			{
				// If we're rendering an animation and the frame just converged
				renderer_animation_state.ensure_output_folder_exists();
				m_screenshoter->write_to_png(renderer_animation_state.get_frame_filepath());
				// Indicating that the animations can step forward since we're done
				// with this frame
				renderer_animation_state.frames_rendered_so_far++;
				if (renderer_animation_state.frames_rendered_so_far == renderer_animation_state.number_of_animation_frames)
					// We just rendered the last frame, deactivating rendering frame sequence state
					renderer_animation_state.is_rendering_frame_sequence = false;
				else
				{
					// Not the last frame
					renderer_animation_state.can_step_animation = true;

					set_render_dirty(true);
				}

			}

			// Sleeping so that we don't burn the CPU and GPU
			std::this_thread::sleep_for(std::chrono::milliseconds(3));
		}
	}
}

void RenderWindow::update_perf_metrics()
{
	m_renderer->compute_render_pass_times();

	// Not adding the frame time if we're rendering at low resolution, not relevant
	m_perf_metrics->add_value(GPURenderer::FULL_FRAME_TIME_KEY, 1000.0f / m_application_state->samples_per_second);

	m_renderer->update_perf_metrics(m_perf_metrics);
}

bool RenderWindow::denoise()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
	DisplaySettings& display_settings = m_display_view_system->get_display_settings();

	display_settings.blend_override = -1.0f;

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
			display_settings.blend_override = 0.0f;

		m_application_settings->denoiser_settings_changed = false;

		return need_denoising && !display_noisy;
	}

	return false;
}
