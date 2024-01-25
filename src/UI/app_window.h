#ifndef APP_WINDOW_H
#define APP_WINDOW_H

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "Utils/commandline_arguments.h"
#include "Renderer/renderer.h"

struct HIPRTOrochiCtx
{
	void init(int device_index)
	{
		OROCHI_CHECK_ERROR(static_cast<oroError>(oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0)));

		OROCHI_CHECK_ERROR(oroInit(0));
		OROCHI_CHECK_ERROR(oroDeviceGet(&orochi_device, device_index));
		OROCHI_CHECK_ERROR(oroCtxCreate(&orochi_ctx, 0, orochi_device));

		oroDeviceProp props;
		OROCHI_CHECK_ERROR(oroGetDeviceProperties(&props, orochi_device));

		std::cout << "hiprt ver." << HIPRT_VERSION_STR << std::endl;
		std::cout << "Executing on '" << props.name << "'" << std::endl;
		if (std::string(props.name).find("NVIDIA") != std::string::npos)
			hiprt_ctx_input.deviceType = hiprtDeviceNVIDIA;
		else
			hiprt_ctx_input.deviceType = hiprtDeviceAMD;

		hiprt_ctx_input.ctxt = oroGetRawCtx(orochi_ctx);
		hiprt_ctx_input.device = oroGetRawDevice(orochi_device);
		hiprtSetLogLevel(hiprtLogLevelError);

		HIPRT_CHECK_ERROR(hiprtCreateContext(HIPRT_API_VERSION, hiprt_ctx_input, hiprt_ctx));
	}

	hiprtContextCreationInput hiprt_ctx_input;
	oroCtx					  orochi_ctx;
	oroDevice				  orochi_device;

	hiprtContext hiprt_ctx;
};

class AppWindow
{
public:
	static constexpr int DISPLAY_TEXTURE_UNIT = 1;

	AppWindow(int width, int height);
	~AppWindow();

	static void APIENTRY gl_debug_output_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
	void resize(int pixels_width, int pixels_height);

	void setup_display_program();
	void setup_renderer(const CommandLineArguments& arguments);
	void set_renderer_scene(const Scene& scene);
	void display(const std::vector<Color>& image_data);
	void display(OrochiBuffer<float>& orochi_buffer);

	void run();
	void quit();

private:
	int m_width, m_height;

	Renderer m_renderer;
	HIPRTOrochiCtx m_hiprt_orochi_ctx;

	GLuint m_display_program;
	GLuint m_display_texture;
	GLFWwindow* m_window;
};

#endif