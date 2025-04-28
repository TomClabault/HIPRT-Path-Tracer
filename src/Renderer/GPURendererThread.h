/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_RENDERER_THREAD_H
#define GPU_RENDERER_THREAD_H

#include "Renderer/RenderPasses/RenderGraph.h"
#include "RenderPasses/FillGBufferRenderPass.h"
#include "RenderPasses/GMoNRenderPass.h"
#include "RenderPasses/ReGIRRenderPass.h"
#include "RenderPasses/ReSTIRDIRenderPass.h"
#include "RenderPasses/ReSTIRGIRenderPass.h"

#include "HostDeviceCommon/RenderData.h"

#include <condition_variable>
#include <memory>

class GPURenderer;
class RenderWindow;

class GPURendererThread
{
public:
	GPURendererThread() {}

	void init(GPURenderer* renderer);

	void start();
	void render_thread_function();

	/**
	 * Initializes and compiles the kernels
	 */
	void setup_render_passes();

	void request_frame();
	void request_exit();

	/**
	 * This function is in charge of updating various "dynamic attributes/properties/buffers" of the renderer before rendering a frame.
	 *
	 * These "dynamic attributes/properties/buffers" can be the adaptive sampling buffers for example.
	 *
	 * It will be checked each whether or not the adaptive sampling buffers need to be
	 * allocated or freed and action will be taken accordingly. This function basically enables a
	 * nice behavior of the application in which the renderer "automatically" reacts to changes
	 * that could be made (through the ImGui interface for example) so that it is always in the
	 * correct state. Said othrewise, this function can be seen as a centralized place for updating
	 * various stuff of the renderer instead of having to scatter these update calls everywhere
	 * in the code.
	 *
	 * The 'delta_time' parameter should be how much time passed, in milliseconds, since the last
	 * call to pre_render_update()
	 */
	void pre_render_update(float delta_time, RenderWindow* render_window);

	/**
	 * This function increments some counters (such as the number of samples rendered so far) after a
	 * sample has been rendered
	 *
	 * This function is private because it is used internally by the render() function
	 */
	void post_sample_update();

	/**
	 * Renders a frame asynchronously.
	 * Querry frame_render_done() to know whether or not the frame has completed or not.
	 */
	void render();

	/**
	 * This just renders a frame by calling all the path tracing kernels.
	 * Nothing special.
	 *
	 * This function is basically the "opposite" of 'render_debug_kernel'
	 */
	void render_path_tracing();
	/**
	 * This function launches the 'm_debug_trace_kernel' and saves its execution time
	 * in 'm_render_pass_times[GPURenderer::DEBUG_KERNEL_TIME_KEY]'
	 */
	void render_debug_kernel();
	GPUKernel& get_debug_trace_kernel();

	RenderGraph& get_render_graph();
	std::shared_ptr<GMoNRenderPass> get_gmon_render_pass();
	std::shared_ptr<ReGIRRenderPass> get_ReGIR_render_pass();
	std::shared_ptr<ReSTIRDIRenderPass> get_ReSTIR_DI_render_pass();
	std::shared_ptr<ReSTIRGIRenderPass> get_ReSTIR_GI_render_pass();

	/**
	 * Returns false if the frame queued asynchronously by a previous call to render()
	 * isn't finished yet.
	 * Returns true if the frame is completed
	 */
	bool frame_render_done();

private:
	/**
	 * Resets the value of the status buffers on the device
	 */
	void internal_pre_render_update_clear_device_status_buffers();

	/**
	 * This function evaluates whether the renderer needs the adaptive
	 * sampling buffers or not. If the buffers are needed (because the
	 * adaptive sampling or the stop noise pixel threshold is enabled for example),
	 * then the buffer will be allocated so that they can be used by the shader.
	 * If they are not needed, they will be freed to save some VRAM.
	 */
	void internal_pre_render_update_adaptive_sampling_buffers();

	/**
	 * Allocates/deallocates the data structure for NEE++ depending on whether or not
	 * NEE++ is being used
	 *
	 * The 'delta_time' parameter should be how much time passed, in milliseconds, since the last
	 * call to internal_pre_render_update_nee_plus_plus()
	 */
	void internal_pre_render_update_nee_plus_plus(float delta_time);

	/**
	 * Allocates/frees the global buffer for BVH traversal when UseSharedStackBVHTraversal is TRUE
	 */
	void internal_pre_render_update_global_stack_buffer();

	void launch_debug_kernel();	

	GPURenderer* m_renderer = nullptr;
	HIPRTRenderData* m_render_data = nullptr;
	RenderGraph m_render_graph;

	// Whether or not the frame queued on the GPU by the last call to render() 
	// is done rendering or not
	bool m_frame_rendered = true;

	// If this kernel isn't empty, then it will be used instead of all the regular path tracing
	// kernels.
	// 
	// This can be useful for debugging performance for example: write a very simple trace kernel
	// that just trace camera rays and set this kernel as the debug kernel and you'll be able to
	// see the raw ray tracing performance without any scuff
	GPUKernel m_debug_trace_kernel;

	std::thread m_render_std_thread;
	std::condition_variable m_render_condition_variable;
	std::mutex m_render_mutex;
	std::mutex m_frame_rendered_variable_mutex;

	bool m_frame_requested = false;
	bool m_exit_requested = false;
};

#endif
