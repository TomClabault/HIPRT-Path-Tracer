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
// - Test scene with full red sphere metallic & full green emission plane (white diffuse color of the material) 
//		+ white ambient light should probably show the reflection of the emission quad 
//		in the sphere as red even if the quad is only emitting pure green (and the sphere is pure red) 
//		because the white ambient light reflected off the white diffuse emissive quad and that's what we 
//		should see in the reflection of the sphere
// - Why is the rough dragon having black fringes even with normal flipping ?
// - why is the view direction below the geometric normal sometimes with clearcoat ?
// - normals AOV not converging correctly ?
// - denoiser not accounting for tranmission correctly since Disney 
// - aspect ratio issue on CPU or GPU ?



// TODO Code Organization:
// - reset_sample_number() is a bad name, I couldn't remember the name when looking for it and was expecting soemthing like 'dirty render' or something
// - Can we have access to HoL when calling disey_metallic_fresnel to avoid passing the two vectors and recomputing the dot product in the return statement ?
// - rename HIPRT kernel files without the hiprt prefix
// - Clean the Git of all the HIP/Orochi binary files. Try to download them automatically in the CMake or write installation instructions
// - DO THE DISNEY SHADING IN SHADING SPACE. WHAT THE H IS THIS CODE BUILDING ONB IN EVERY FUNCTION HUH?
// - Check for light and view direction in the same hemisphere in the disney eval function, not just in the clearcoat eval
// - Check for sampled light direction not under the surface in disney sample, before eval, not just in the metallic/clearcoat sample
// - reorganize methods order in RenderWindow
// - use constructors instead of struct {} syntax in gpu code
// - separate path tracer kernel functions in header files
// - do not duplicate render functions. Make a common h file that uses the float3 type (cosine_weighted_direction_around_normal, hiprt_lambertian.h:7)
// - instead of duplicating structures (RendererMaterial + RendererMaterial, BRDF + BRDF, ...), it would be better to create a folder
//		HostDeviceCommon containing the structures that are used both by the GPU and CPU renderer
// - imgui controller to put all the imgui code in one class
// - put mouse / keyboard code in an interactor
// - when the mouse / keyvoard code will be in an interactor class, have the is_interacting boloean in this interactor class
//		and poll it from the main loop to check whether we need to render the frame at a lower resolution or not
// - check for level of abstractions in functions



// TODO Features:
// - Look at what Orochi & HIPCC can do in terms of displaying registers used / options to specify shared stack size / block size (-DBLOCK_SIZE, -DSHARED_STACK_SIZE)
// - Have the UI run at its own framerate to avoid having the UI come to a crawl when the path tracing is expensive
// - Denoiser blend to allow blending the original noisy image and the perfect denoised result by a given factor
// - When modifying the emission of a material with the material editor, it should be reflected in the scene and allow the direct sampling of the geometry
// - Color fallof (change of material base base_color based on the angle with the view direction and the normal
// - Transmission color
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
// - Flakes BRDF (maybe look at OSPRay implementation ?)
// - ImGuizmo
// - Orochi 2.0 --> Textures and OpenGL Interop 
// - Paths roughness regularization
// - choose disney base_color model (disney, lambertian, oren nayar)
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
// - shader cache
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

	setup_display_program();
	m_renderer.init_ctx(0);
	m_renderer.compile_trace_kernel(m_application_settings.kernel_files[m_application_settings.selected_kernel].c_str(),
		m_application_settings.kernel_functions[m_application_settings.selected_kernel].c_str());
	m_renderer.change_render_resolution(width, height);

	m_denoiser.set_buffers(m_renderer.get_color_framebuffer().get_pointer(),
		m_renderer.get_denoiser_normals_buffer().get_pointer(), m_renderer.get_denoiser_albedo_buffer().get_pointer(),
		width, height);

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
	if (m_render_settings.keep_same_resolution)
		resolution_scale = m_render_settings.target_width / (float)pixels_width; // TODO what about the height changing ?

	int new_render_width = std::floor(pixels_width * resolution_scale);
	int new_render_height = std::floor(pixels_height * resolution_scale);
	m_renderer.change_render_resolution(new_render_width, new_render_height);
	m_denoiser.set_buffers(m_renderer.get_color_framebuffer().get_pointer(),
		m_renderer.get_denoiser_normals_buffer().get_pointer(), m_renderer.get_denoiser_albedo_buffer().get_pointer(), 
		new_render_width, new_render_height);

	// Recreating the OpenGL display texture
	glActiveTexture(GL_TEXTURE0 + RenderWindow::DISPLAY_TEXTURE_UNIT);
	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, new_render_width, new_render_height, 0, GL_RGB, GL_FLOAT, nullptr);

	reset_sample_number();
}

void RenderWindow::change_resolution_scaling(float new_scaling)
{
	float new_render_width = std::floor(m_viewport_width * new_scaling);
	float new_render_height = std::floor(m_viewport_height * new_scaling);

	m_renderer.change_render_resolution(new_render_width, new_render_height);
	m_denoiser.set_buffers(m_renderer.get_color_framebuffer().get_pointer(),
		m_renderer.get_denoiser_normals_buffer().get_pointer(), m_renderer.get_denoiser_albedo_buffer().get_pointer(),
		new_render_width, new_render_height);

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
		reset_sample_number();

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

void RenderWindow::setup_display_program()
{
	// Creating the shaders for displaying the path traced render
	m_display_program = OpenGLUtils::compile_shader_program("Shaders/fullscreen_quad.vert", "Shaders/display.frag");

	// Creating the texture that will contain the path traced data to be displayed
	// by the shader.
	glGenTextures(1, &m_display_texture);
	glActiveTexture(GL_TEXTURE0 + RenderWindow::DISPLAY_TEXTURE_UNIT);
	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_viewport_width, m_viewport_height, 0, GL_RGB, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glUseProgram(m_display_program);
	glUniform1i(glGetUniformLocation(m_display_program, "u_texture"), RenderWindow::DISPLAY_TEXTURE_UNIT);
	glUniform1i(glGetUniformLocation(m_display_program, "u_sample_number"), 0);
	glUniform1f(glGetUniformLocation(m_display_program, "u_gamma"), m_application_settings.tone_mapping_gamma);
	glUniform1f(glGetUniformLocation(m_display_program, "u_exposure"), m_application_settings.tone_mapping_exposure);
}

void RenderWindow::update_renderer_view_translation(float translation_x, float translation_y)
{
	if (translation_x == 0.f && translation_y == 0.0f)
		return;

	reset_sample_number();

	glm::vec3 translation = glm::vec3(translation_x / m_application_settings.view_translation_sldwn_x, translation_y / m_application_settings.view_translation_sldwn_y, 0.0f);
	m_renderer.translate_camera_view(translation);
}

void RenderWindow::update_renderer_view_rotation(float offset_x, float offset_y)
{
	reset_sample_number();

	float rotation_x, rotation_y;

	rotation_x = offset_x / m_viewport_width * 2.0f * M_PI / m_application_settings.view_rotation_sldwn_x;
	rotation_y = offset_y / m_viewport_height * 2.0f * M_PI / m_application_settings.view_rotation_sldwn_y;

	m_renderer.rotate_camera_view(glm::vec3(rotation_x, rotation_y, 0.0f));
}

void RenderWindow::update_renderer_view_zoom(float offset)
{
	if (offset == 0.0f)
		return;

	reset_sample_number();

	m_renderer.zoom_camera_view(offset / m_application_settings.view_zoom_sldwn);
}

void RenderWindow::increment_sample_number()
{
	if (m_render_settings.render_low_resolution)
		m_render_settings.sample_number++; // Only doing 1 SPP when moving the cameras
	else
		m_render_settings.sample_number += m_render_settings.samples_per_frame;

	glUseProgram(m_display_program);
	glUniform1i(glGetUniformLocation(m_display_program, "u_sample_number"), m_render_settings.sample_number);
}

void RenderWindow::reset_sample_number()
{
	m_startRenderTime = std::chrono::high_resolution_clock::now();
	m_renderer.set_sample_number(0);

	glUseProgram(m_display_program);
	glUniform1i(glGetUniformLocation(m_display_program, "u_sample_number"), 0);

	reset_frame_number();
}

void RenderWindow::reset_frame_number()
{
	m_render_settings.frame_number = 0;
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
		glfwPollEvents();
		glClear(GL_COLOR_BUFFER_BIT);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		bool image_rendered = false;
		if (!(m_application_settings.stop_render_at != 0 && m_render_settings.sample_number + 1 > m_application_settings.stop_render_at))
		{
			m_renderer.render(m_denoiser);
			increment_sample_number();
			m_render_settings.frame_number++;

			image_rendered = true;
		}

		if (m_render_settings.enable_denoising)
		{
			if (m_application_settings.denoise_at_target_sample_count)
			{
				if (m_render_settings.sample_number == m_application_settings.stop_render_at)
				{
					m_denoiser.denoise();
					display(m_denoiser.get_denoised_data_pointer());
				}
				else
					display(m_renderer.get_color_framebuffer());
			}
			else
			{
				if ((m_render_settings.sample_number % m_render_settings.denoiser_sample_skip) == 0)
				{
					m_denoiser.denoise();
					m_application_settings.last_denoised_sample_count = m_render_settings.sample_number;
				}
			
				DisplaySettings settings;
				settings.display_normals = false;
				settings.do_tonemapping = true;
				settings.scale_by_frame_number = true;
				settings.sample_count_override = m_application_settings.last_denoised_sample_count;

				set_display_settings(settings);

				display(m_denoiser.get_denoised_data_pointer());
			}
		}
		else
		{
			switch (m_application_settings.debug_display_denoiser)
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

			case DisplayView::NONE:
			default:
				display(m_renderer.get_color_framebuffer());
				break;
			}
		}
		
		display_imgui();

		glfwSwapBuffers(m_window);
	}

	quit();
}

DisplaySettings RenderWindow::get_display_settings()
{
	return m_display_settings;
}

void RenderWindow::set_display_settings(DisplaySettings settings)
{
	m_display_settings = settings;
}

void RenderWindow::setup_display_uniforms(GLuint program)
{
	glUseProgram(program);
	glUniform1i(glGetUniformLocation(program, "u_display_normals"), m_display_settings.display_normals);
	glUniform1i(glGetUniformLocation(program, "u_scale_by_frame_number"), m_display_settings.scale_by_frame_number);
	glUniform1i(glGetUniformLocation(program, "u_do_tonemapping"), m_display_settings.do_tonemapping);
	glUniform1i(glGetUniformLocation(program, "u_sample_count_override"), m_display_settings.sample_count_override);
}

void RenderWindow::display(const void* data)
{
	setup_display_uniforms(m_display_program);

	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_renderer.m_render_width, m_renderer.m_render_height, GL_RGB, GL_FLOAT, data);

	glDrawArrays(GL_TRIANGLES, 0, 6);
}

void RenderWindow::show_render_settings_panel()
{
	if (!ImGui::CollapsingHeader("Render Settings"))
		return;

	bool render_dirty = false;

	if (ImGui::Combo("Render Kernel", &m_application_settings.selected_kernel, "Full Path Tracer\0Normals Visualisation\0\0"))
	{
		m_renderer.compile_trace_kernel(m_application_settings.kernel_files[m_application_settings.selected_kernel].c_str(), m_application_settings.kernel_functions[m_application_settings.selected_kernel].c_str());
		render_dirty = true;
	}

	const char* items[] = { "Default", "Denoiser - Normals", "Denoiser - Denoised normals", "Denoiser - Albedo", "Denoiser - Denoised albedo" };
	if (ImGui::Combo("Display View", (int*)(&m_application_settings.debug_display_denoiser), items, IM_ARRAYSIZE(items)))
	{
		DisplaySettings display_settings;

		switch (m_application_settings.debug_display_denoiser)
		{
		case DisplayView::DISPLAY_NORMALS:
		case DisplayView::DISPLAY_DENOISED_NORMALS:
			display_settings.display_normals = true;
			display_settings.do_tonemapping = false;
			display_settings.scale_by_frame_number = false;
			display_settings.sample_count_override = -1;

			break;

		case DisplayView::DISPLAY_ALBEDO:
		case DisplayView::DISPLAY_DENOISED_ALBEDO:
			display_settings.display_normals = false;
			display_settings.do_tonemapping = false;
			display_settings.scale_by_frame_number = false;
			display_settings.sample_count_override = -1;

			break;

		case DisplayView::NONE:
		default:
			display_settings.display_normals = false;
			display_settings.do_tonemapping = true;
			display_settings.scale_by_frame_number = true;
			display_settings.sample_count_override = -1;

			break;
		}

		set_display_settings(display_settings);
	}

	if (m_render_settings.keep_same_resolution)
		ImGui::BeginDisabled();
	float resolution_scaling_backup = m_application_settings.render_resolution_scale;
	if (ImGui::InputFloat("Resolution scale", &m_application_settings.render_resolution_scale))
	{
		float& resolution_scale = m_application_settings.render_resolution_scale;
		if (resolution_scale <= 0)
			resolution_scale = resolution_scaling_backup;

		change_resolution_scaling(resolution_scale);
		render_dirty = true;
	}
	if (m_render_settings.keep_same_resolution)
		ImGui::EndDisabled();

	// TODO for the denoising with normals / albedo, add imgui buttons to display normals / albedo buffer
	if (ImGui::Checkbox("Keep same render resolution", &m_render_settings.keep_same_resolution))
	{
		if (m_render_settings.keep_same_resolution)
		{
			// Remembering the width and height we need to target
			m_render_settings.target_width = m_renderer.m_render_width;
			m_render_settings.target_height = m_renderer.m_render_height;
		}
	}

	ImGui::Separator();

	if (ImGui::InputInt("Stop render at sample count", &m_application_settings.stop_render_at))
		m_application_settings.stop_render_at = std::max(m_application_settings.stop_render_at, 0);
	ImGui::InputInt("Samples per frame", &m_render_settings.samples_per_frame);
	if (ImGui::InputInt("Max bounces", &m_render_settings.nb_bounces))
	{
		// Clamping to 0 in case the user input a negative number of bounces	
		m_render_settings.nb_bounces = std::max(m_render_settings.nb_bounces, 0); 
		render_dirty = true;
	}

	render_dirty |= ImGui::Checkbox("Use ambient light", &m_renderer.get_world_settings().use_ambient_light);
	render_dirty |= ImGui::ColorEdit3("Ambient light color", (float*)&m_renderer.get_world_settings().ambient_light_color, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);

	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	if (render_dirty)
		reset_sample_number();
}

void RenderWindow::show_objects_panel()
{
	if (!ImGui::CollapsingHeader("Objects"))
		return;

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
		reset_sample_number();
	}

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void RenderWindow::show_denoiser_panel()
{
	if (!ImGui::CollapsingHeader("Denoiser"))
		return;

	if (ImGui::Checkbox("Enable denoiser", &m_render_settings.enable_denoising))
		if (!m_render_settings.enable_denoising) // Denoising unchecked
			// Making sure to reset the sample count override that may have been
			// set when the denoising checkbox was checked
			m_display_settings.sample_count_override = -1;

	ImGui::Checkbox("Only Denoise at Target Sample Count", &m_application_settings.denoise_at_target_sample_count);
	if (m_application_settings.denoise_at_target_sample_count)
	{
		ImGui::BeginDisabled();
		ImGui::SliderInt("Denoise Sample Skip", &m_render_settings.denoiser_sample_skip, 1, 128);
		ImGui::EndDisabled();
	}
	else
		ImGui::SliderInt("Denoise Sample Skip", &m_render_settings.denoiser_sample_skip, 1, 128);

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void RenderWindow::show_post_process_panel()
{
	if (!ImGui::CollapsingHeader("Post-processing"))
		return;

	if (ImGui::InputFloat("Gamma", &m_application_settings.tone_mapping_gamma))
		glUniform1f(glGetUniformLocation(m_display_program, "u_gamma"), m_application_settings.tone_mapping_gamma);
	if (ImGui::InputFloat("Exposure", &m_application_settings.tone_mapping_exposure))
		glUniform1f(glGetUniformLocation(m_display_program, "u_exposure"), m_application_settings.tone_mapping_exposure);

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void RenderWindow::display_imgui()
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

	glDeleteTextures(1, &m_display_texture);
	glDeleteProgram(m_display_program);

	std::exit(0);
}
