/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_RENDER_PASS_H
#define RENDERER_RENDER_PASS_H

class GPURenderer;

#include "Compiler/GPUKernel.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"
#include "UI/PerformanceMetricsComputer.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

class GPURenderer;
struct HIPRTRenderData;

/**
 * Interface for a GPU Renderer render pass
 */
class RenderPass
{
public:
	RenderPass();
	RenderPass(GPURenderer* renderer);
	RenderPass(GPURenderer* renderer, const std::string& name);

	/**
	 * This will be called once when the render pass is created.
	 *
	 * After this function is called, the render pass should be ready to be
	 * launch()ed (pre_render_update() will be called before launch() though)
	 *
	 * This compile method will always be called on all render passes of a renderer.
	 * It is the responsibility of the class overriding this method to compile the kernels if necessary or not.
	 *
	 * For example: if a ReSTIRDIRenderPass implements this interface but the renderer doesn't
	 * actually use ReSTIR DI at the moment, then calling 'compile' should probably be a no-op (i.e. return directly),
	 * otherwise, this would be compiling kernels unecessarily (since the render pass is not being used
	 * 
	 * The kernels in this function may be compiled asynchronously by using the ThreadManager and launching threads
	 * with the 'COMPILE_KERNELS_THREAD_KEY' key. Look at the ReSTIR DI render pass for some examples
	 * 
	 * The default implementation does this and compiles all kernels found in the map returned by
	 * 'get_all_kernels()'. This assumes that kernels are configured in the constructor 
	 * (given their options, file path, kernel function name ,...). Have a look at the GMoNRenderPass or ReSTIRDIRenderPass
	 * for examples
	 */
	virtual void compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets = {});

	/**
	 * When some compiler options of the renderer have been changed and the render pass
	 * needs to be recompiled
	 *
	 * Same remark here as for compile(): It is the responsibility of the class overriding this method
	 * to compile the kernels if necessary or not.
	 * 
	 * Recompilation of the kernels may *not* be asynchronous without the addition of a synchronization 
	 * elsewhere (to be sure that the kernels will be compiled before the next frame starts rendering).
	 * This 'recompile' function is most likely to be called from the ImGui interface code and so 
	 * it must be blocking (or add synchronization elsewhere in the codebase) to be sure that 
	 * the kernels will be fully recompiled before the RenderWindow submits a new frame to the GPU
	 */
	virtual void recompile(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets = {}, bool silent = false, bool use_cache = true);

	/**
	 * That function is called when the host renderer is resized (i.e. when the user resizes the window)
	 * 
	 * This function should be used to resize the buffers used by this 
	 * render pass if those buffers depend on the render resolution
	 */
	virtual void resize(unsigned int new_width, unsigned int new_height) = 0;

	/**
	 * Function before 'pre_render_update()' that should compile kernels that haven't
	 * been compiled so far if necessary
	 * 
	 * For example, in a ReSTIR DI render pass, if the temporal reuse is disabled 
	 * when the application starts, the temporal reuse kernel will not be compiled 
	 * because it isn't needed. However, if the user then decides to enable temporal 
	 * reuse at runtime, the temporal reuse will now have to be compiled and this 
	 * function is in charge.
	 * 
	 * Should return true if at least one kernel was compiled/recompiled, false otherwise
	 */
	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets = {}, bool silent = false, bool use_cache = true) { return false; }

	/**
	 * Called at each frame, before launch()
	 * 
	 * Buffer allocations / deallocations depending on whether or not this render pass
	 * is necessary to the renderer can be done here
	 * 
	 * 'delta_time' is the time in milliseconds that elapsed between two calls of this method
	 * 
	 * This function should return true if the HIPRTRenderData structure of the renderer will have to
	 * be set up again. This is typically the case when some buffers of the render pass have been allocated/deallocated/resized
	 * and so we need to set the new buffer pointers in the HIPRTRenderData structure such that the GPU
	 * uses the proper buffer pointers.
	 * 
	 * Returns false otherwise
	 */
	virtual bool pre_render_update(float delta_time) = 0;

	/**
	 * This should launch the render pass kernels on the GPU
	 * 
	 * Returns true if the render pass was indeed launched
	 * Returns false otherwise (if the render pass isn't being used or if the render pass is only launched every frames or ...)
	 */
	virtual bool launch() = 0;

	/**
	 * Called at each frame, after launch()
	 * 
	 * Some counter incrementation can be done in here
	 */
	virtual void post_render_update() = 0;

	/**
	 * This function is called when the renderer that holds this render pass needs to 
	 * update its render_data structure.
	 * 
	 * For the most part, this function should modify m_renderer->get_render_data() to set
	 * up the pointers / variables that will be used by the GPU in the shaders of the render pass
	 * 
	 * the HIPRTRenderData data structure can be accessed with m_renderer->get_render_data() and it
	 * can be modified directly
	 */
	virtual void update_render_data() = 0;

	/**
	 * Called when the user resets the render (an option was changed in ImGui, the camera moved, ...)
	 */
	virtual void reset() = 0;

	/**
	 * This function is called once per frame, after all render passes have executed.
	 * 
	 * This function should get a reference to the render pass times of the renderer:
	 * std::unordered_map<std::string, float>& ms_time_per_pass = m_renderer->get_render_pass_times();
	 * 
	 * and then the execution time of this render pass should be set in the 'ms_time_per_pass' map of the renderer.
	 * 
	 * For example, for the light presampling pass of ReSTIR DI: 
	 * ms_time_per_pass[ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID] = m_kernels[ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID].get_last_execution_time();
	 * 
	 * The key used in the map can be arbitrary but should be unique. The practice used in this
	 * codebase is to define the keys in the render pass itself as "static const std::string" and
	 * use these keys to index the 'ms_time_per_pass' map.
	 * 
	 * In the example above, the key is 'ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID'
	 */
	virtual void compute_render_times();

	/**
	 * This function is called once per frame, after all render passes have executed.
	 * 
	 * This function should add the render time of the pass to the performance metrics computer.
	 * 
	 * For example:
	 * std::unordered_map<std::string, float> render_pass_times = m_renderer->get_render_pass_times();
	 * perf_metrics->add_value(ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID, render_pass_times[ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID]);
	 * 
	 * The performance metrics computer is what stores the timings of all the render passes to display
	 * the "Performance metrics" panel in ImGui
	 * 
	 * It is unlikely that you need to override the default implementation if your 'get_all_kernels()' function
	 * is properly written (i.e. only returns the kernels actually being used by the render pass)
	 */
	virtual void update_perf_metrics(std::shared_ptr<PerformanceMetricsComputer> perf_metrics);

	/**
	 * Returns a map of all the kernels of this render pass
	 *
	 * The map keys are the kernel name
	 * The map values are the kernel themselves
	 * 
	 * If this render pass isn't being used by the renderer 
	 * (for example a ReSTIR DI render pass whereas we're using RIS 
	 * for direct lighting at the first bounce, i.e. the ReSTIR DI render 
	 * pass is not in use), this function should return and empty map. 
	 * 
	 * This is such that ImGui doesn't display the GPU timings of this render pass.
	 * 
	 * This function also should not return inactive kernels of a render pass if
	 * the render pass has more than 1 kernel. For example, the ReSTIR DI render 
	 * pass has multiple kernels: spatio-temporal, spatial, temporal. 
	 * If spatiotemporal is being used, the spatial and temporal are not being used and 
	 * so they will not be in the map returned by this function. This is also to avoid 
	 * ImGui from displaying the performance metrics about kernels that are not in use 
	 * (and so we have no performance metrics on them)
	 */
	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_all_kernels();

	/**
	 * Returns a map of all the kernels of the render pass that trace rays (shadow rays, bounce rays, ...)
	 *
	 * This is used in ImGui in the performance settings panel where we can adjust the
	 * amount of shared memory used for the BVH traversal. Because this is only useful for
	 * kernels that trace rays, we want a function that returns only the kernels that trace rays
	 *
	 * The map keys are the kernel name
	 * The map values are the kernel themselves
	 * 
	 * If this render pass isn't being used by the renderer 
	 * (for example a ReSTIR DI render pass whereas we're using RIS 
	 * for direct lighting at the first bounce, i.e. the ReSTIR DI render 
	 * pass is not in use), this function should return and empty map. 
	 * 
	 * This is such that ImGui doesn't display the GPU timings of this render pass.
	 * 
	 * This function also should not return inactive kernels of a render pass if
	 * the render pass has more than 1 kernel. For example, the ReSTIR DI render 
	 * pass has multiple kernels: spatio-temporal, spatial, temporal. 
	 * If spatiotemporal is being used, the spatial and temporal are not being used and 
	 * so they will not be in the map returned by this function. This is also to avoid 
	 * ImGui from displaying the performance metrics about kernels that are not in use 
	 * (and so we have no performance metrics on them)
	 */
	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_tracing_kernels();

	/**
	 * This function may be overriden by render passes that can be enabled/disabled at runtime.
	 * 
	 * This is the case of ReSTIR render passes for example: the ReSTIR render passes are not always used for rendering.
	 * 
	 * This function is then called in the default implementation of recompile() for example such that the kernels 
	 * of the render pass will not be recompiled if the render pass is not being used
	 */
	virtual bool is_render_pass_used() const;

	/**
	 * Adds another render pass as a dependency of this render pass.
	 * The dependency render pass will then always be executed before this render pass is executed
	 */
	void add_dependency(std::shared_ptr<RenderPass> dependency);

	/**
	 * Returns a list of all the dependencies so far added to this render pass
	 */
	std::vector<std::shared_ptr<RenderPass>>& get_dependencies();

	const std::string& get_name();
	void set_name(const std::string& new_name);

protected:
	std::string m_name;

	// Access to the renderer that holds the render pass
	GPURenderer* m_renderer = nullptr;
	HIPRTRenderData* m_render_data = nullptr;

	// Other render passes which this render pass depends on.
	// They will be launched before this render pass
	std::vector<std::shared_ptr<RenderPass>> m_dependencies;

	// Name --> GPUKernel map
	std::map<std::string, std::shared_ptr<GPUKernel>> m_kernels;
};

#endif