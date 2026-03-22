#include "UI/ImGui/ImGuiConvergenceGraphWidget.h"
#include "UI/ImGui/ImGuiConvergenceGraphWidgetScreenshotter.h"
#include "UI/RenderWindow.h"
#include "Utils/Utils.h"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

ImGuiConvergenceGraphWidgetScreenshotter::ImGuiConvergenceGraphWidgetScreenshotter(ImGuiConvergenceGraphWidget* graph_widget) : m_graph_widget(graph_widget) {}

void ImGuiConvergenceGraphWidgetScreenshotter::init()
{
	ImFontAtlas* shared_fonts = ImGui::GetIO().Fonts;

	m_captureImguiCtx  = ImGui::CreateContext(shared_fonts);
	m_captureImPlotCtx = ImPlot::CreateContext();

	ImGuiContext* previous_imgui_ctx = ImGui::GetCurrentContext();

	// OpenGL 3 Init
	ImGui::SetCurrentContext(m_captureImguiCtx);
	ImGui_ImplOpenGL3_Init();

	ImGui::SetCurrentContext(previous_imgui_ctx);
}

std::vector<unsigned char> ImGuiConvergenceGraphWidgetScreenshotter::screenshot_graph_to_memory(int screenshot_width, int screenshot_height, bool flip_y)
{
	regenerate_capture_fbo(screenshot_width, screenshot_height);
	if (m_capture_fbo == 0)
		return {};

	if (!m_init_done)
		init();

	// Save GL state we are about to disturb.
	GLint previous_fbo		   = 0;
	GLint previous_viewport[4] = { 0, 0, 0, 0 };
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previous_fbo);
	glGetIntegerv(GL_VIEWPORT, previous_viewport);

	// If you use a dedicated capture ImGui context, create it once and reuse it.
	// It should share the font atlas with the main context.
	ImGuiContext* main_ctx = ImGui::GetCurrentContext();

	ImGui::SetCurrentContext(main_ctx);

	// Switch to the capture contexts.
	ImGuiContext* previous_imgui_ctx   = ImGui::GetCurrentContext();
	ImPlotContext* previous_implot_ctx = ImPlot::GetCurrentContext();

	ImGui::SetCurrentContext(m_captureImguiCtx);
	ImPlot::SetCurrentContext(m_captureImPlotCtx);
	ImGui_ImplOpenGL3_NewFrame();

	ImGuiIO& io				   = ImGui::GetIO();
	io.DisplaySize			   = ImVec2((float)screenshot_width, (float)screenshot_height);
	io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
	io.DeltaTime			   = 1.0f / 60.0f;

	// Bind the offscreen target.
	glBindFramebuffer(GL_FRAMEBUFFER, m_capture_fbo);
	glViewport(0, 0, screenshot_width, screenshot_height);
	glClearColor(0, 0, 0, 0); // transparent background
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	// Build a tiny isolated ImGui frame containing only the plot.
	ImGui::NewFrame();

	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImVec2((float)screenshot_width, (float)screenshot_height));

	ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
							 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

	ImGui::Begin("##capture_plot_window", nullptr, flags);

	// Optional: remove window padding so the plot fills the whole image.
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

	double xmin = std::numeric_limits<double>::max();
	double xmax = std::numeric_limits<double>::lowest();
	double ymin = std::numeric_limits<double>::max();
	double ymax = std::numeric_limits<double>::lowest();

	for (size_t i = 0; i < m_graph_widget->get_recorded_xs_list().size(); ++i)
	{
		const auto& xs = m_graph_widget->get_recorded_xs_list()[i];
		const auto& ys = m_graph_widget->get_recorded_ys_list()[i];

		for (size_t j = 0; j < xs.size(); ++j)
		{
			xmin = hippt::min(xmin, (double)xs[j]);
			xmax = hippt::max(xmax, (double)xs[j]);
			ymin = hippt::min(ymin, (double)ys[j]);
			ymax = hippt::max(ymax, (double)ys[j]);
		}
	}

	if (xmin == std::numeric_limits<double>::max() || xmax == std::numeric_limits<double>::lowest() || ymin == std::numeric_limits<double>::max() ||
		ymax == std::numeric_limits<double>::lowest())
	{
		xmin = 0.0;
		xmax = 1.0;
		ymin = 0.0;
		ymax = 1.0;
	}

	ImPlot::SetNextAxesLimits(xmin, xmax, ymin, ymax, ImPlotCond_Always);
	m_graph_widget->draw(ImVec2((float)screenshot_width - SCREENSHOT_PADDING_WIDTH, (float)screenshot_height - SCREENSHOT_PADDING_HEIGHT));
	ImGui::PopStyleVar();

	ImGui::End();

	ImGui::Render();

	// Render the capture ImGui draw data into the FBO.
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	// Read back pixels.
	std::vector<unsigned char> pixels((size_t)screenshot_width * (size_t)screenshot_height * 4);

	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, screenshot_width, screenshot_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

	// Restore previous GL state.
	glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)previous_fbo);
	glViewport(previous_viewport[0], previous_viewport[1], previous_viewport[2], previous_viewport[3]);

	// Restore contexts.
	ImGui::SetCurrentContext(previous_imgui_ctx);
	ImPlot::SetCurrentContext(previous_implot_ctx);

	if (flip_y)
	{
		std::vector<unsigned char> flipped((size_t)screenshot_width * (size_t)screenshot_height * 4);
		const size_t row_bytes = (size_t)screenshot_width * 4;

		for (int row = 0; row < screenshot_height; ++row)
			std::memcpy(&flipped[(size_t)row * row_bytes], &pixels[(size_t)(screenshot_height - 1 - row) * row_bytes], row_bytes);
		return flipped;
	}

	return pixels;
}

bool ImGuiConvergenceGraphWidgetScreenshotter::screenshot_graph_to_file(int screenshot_width, int screenshot_height, const std::string_view filename)
{
	screenshot_width += SCREENSHOT_PADDING_WIDTH;
	screenshot_height += SCREENSHOT_PADDING_HEIGHT;

	std::vector<unsigned char> pixels = screenshot_graph_to_memory(screenshot_width, screenshot_height, false);

	unsigned int stride_bytes = screenshot_width * 4;
	bool ret				  = stbi_write_png(filename.data(), screenshot_width, screenshot_height, 4, pixels.data(), stride_bytes) != 0;

	restore_viewport_texture_after_screenshotting();

	return ret;
}

void ImGuiConvergenceGraphWidgetScreenshotter::screenshot_graph_to_clipboard(int screenshot_width, int screenshot_height)
{
	screenshot_width += SCREENSHOT_PADDING_WIDTH;
	screenshot_height += SCREENSHOT_PADDING_HEIGHT;

	std::vector<unsigned char> pixels = screenshot_graph_to_memory(screenshot_width, screenshot_height, true);

	Utils::copy_image_to_clipboard(Image8Bit(pixels, screenshot_width, screenshot_height, 4), false);

	restore_viewport_texture_after_screenshotting();
}

void ImGuiConvergenceGraphWidgetScreenshotter::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;
}

void ImGuiConvergenceGraphWidgetScreenshotter::regenerate_capture_fbo(int width, int height)
{
	destroy_capture_fbo();

	m_capture_width	 = width;
	m_capture_height = height;

	glGenFramebuffers(1, &m_capture_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, m_capture_fbo);

	glGenTextures(1, &m_capture_color_tex);
	glBindTexture(GL_TEXTURE_2D, m_capture_color_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_capture_width, m_capture_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_capture_color_tex, 0);

	// Optional, but safe if your rendering state expects depth/stencil.
	glGenRenderbuffers(1, &m_capture_depth_renderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, m_capture_depth_renderbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, m_capture_width, m_capture_height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_capture_depth_renderbuffer);

	const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE)
	{
		// Handle error here in your own way.
		// You may want to log `status`.
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void ImGuiConvergenceGraphWidgetScreenshotter::destroy_capture_fbo()
{
	if (m_capture_depth_renderbuffer)
	{
		glDeleteRenderbuffers(1, &m_capture_depth_renderbuffer);
		m_capture_depth_renderbuffer = 0;
	}

	if (m_capture_color_tex)
	{
		glDeleteTextures(1, &m_capture_color_tex);
		m_capture_color_tex = 0;
	}

	if (m_capture_fbo)
	{
		glDeleteFramebuffers(1, &m_capture_fbo);
		m_capture_fbo = 0;
	}
}

void ImGuiConvergenceGraphWidgetScreenshotter::restore_viewport_texture_after_screenshotting()
{
	// This is the lazy way out. For some reason screenshotting the convergence graph with the new "draw to separate FBO method" clear our texture for
	// displaying the output of the renderer to the screen. Not sure why, too lazy to figure it out so we just re-upload the buffers to the texture
	// instead if we screenshotted something
	m_render_window->get_display_view_system()->upload_relevant_buffers_to_texture();
}
