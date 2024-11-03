/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RESTIR_DI_RENDER_PASS_H
#define RESTIR_DI_RENDER_PASS_H

#include "Device/includes/ReSTIR/DI/Reservoir.h"
#include "Device/includes/ReSTIR/DI/PresampledLight.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HostDeviceCommon/RenderData.h"
#include "UI/PerformanceMetricsComputer.h"

class GPURenderer;

class ReSTIRDIRenderPass
{
public:
	/**
	 * These constants here are used to reference kernel objects in the 'm_kernels' map
	 * or in the 'm_render_pass_times' map
	 */
	static const std::string RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID;
	static const std::string RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID;
	static const std::string RESTIR_DI_SPATIAL_REUSE_KERNEL_ID;
	static const std::string RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID;
	static const std::string RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID;

	/**
	 * This map contains constants that are the name of the main function of the kernels, their entry points.
	 * They are used when compiling the kernels.
	 *
	 * This means that if you define your camera ray kernel main function as:
	 *
	 * GLOBAL_KERNEL_SIGNATURE(void) CameraRays(HIPRTRenderData render_data, int2 res)
	 *
	 * Then KERNEL_FUNCTION_NAMES[CAMERA_RAYS_KERNEL_ID] = "CameraRays"
	 */
	static const std::unordered_map<std::string, std::string> KERNEL_FUNCTION_NAMES;

	/**
	 * Same as 'KERNELfUNCTION_NAMES' but for kernel files
	 */
	static const std::unordered_map<std::string, std::string> KERNEL_FILES;

	ReSTIRDIRenderPass() {}
	ReSTIRDIRenderPass(GPURenderer* renderer);

	void compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::unordered_set<std::string>& options_excluded_from_synchro, std::vector<hiprtFuncNameSet>& func_name_sets);
	void recompile(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent = false, bool use_cache = true);
	/**
	 * Precompiles all kernels of this render pass to fill to shader cache in advance.
	 * 
	 * Kernels will be compiled with their *current* options but with the options contained
	 * in 'partial_options' overriding the corresponding options of the kernels
	 */
	void precompile_kernels(GPUKernelCompilerOptions partial_options, std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets);

	/**
	 * Allocates/frees the ReSTIR DI buffers depending on whether or not the renderer
	 * needs them (whether or not ReSTIR DI is being used basically) respectively.
	 */
	void update();
	void update_render_data();

	void resize(int new_width, int new_height);

	void reset();

	void launch();

	LightPresamplingParameters configure_light_presampling_pass();
	void configure_initial_pass();
	void configure_temporal_pass();
	void configure_temporal_pass_for_fused_spatiotemporal();
	void configure_spatial_pass(int spatial_pass_index);
	void configure_spatial_pass_for_fused_spatiotemporal(int spatial_pass_index);
	void configure_spatiotemporal_pass();
	void configure_output_buffer();

	void launch_presampling_lights_pass();
	void launch_initial_candidates_pass();
	void launch_temporal_reuse_pass();
	void launch_spatial_reuse_passes();
	void launch_spatiotemporal_pass();

	void compute_render_times(std::unordered_map<std::string, float>& times);
	void update_perf_metrics(std::shared_ptr<PerformanceMetricsComputer> perf_metrics);

	std::map<std::string, GPUKernel> m_kernels;

private:
	// ReSTIR reservoirs for the initial candidates
	OrochiBuffer<ReSTIRDIReservoir> initial_candidates_reservoirs;
	// ReSTIR reservoirs for the output of the spatial reuse pass
	OrochiBuffer<ReSTIRDIReservoir> spatial_output_reservoirs_1;
	// ReSTIR DI final reservoirs of the frame. 
	// This the output of the spatial reuse passes.
	// Those are the reservoirs that are carried over between frames for
	// the temporal reuse pass to feed upon
	OrochiBuffer<ReSTIRDIReservoir> spatial_output_reservoirs_2;

	// Buffer that holds the presampled lights if light presampling is enabled 
	// (GPUKernelCompilerOptions::RESTIR_DI_DO_LIGHTS_PRESAMPLING)
	//
	// Implementation from the paper
	// [Rearchitecting Spatiotemporal Resampling for Production] https://research.nvidia.com/publication/2021-07_rearchitecting-spatiotemporal-resampling-production
	OrochiBuffer<ReSTIRDIPresampledLight> presampled_lights_buffer;

	// Whether or not we're currently rendering an odd frame.
	// This is used to adjust which buffers are used as input/outputs
	// and ping-pong between them
	bool odd_frame = false;

	// Events for timing the cumulated render time of all the spatial reuses passes
	oroEvent_t spatial_reuse_time_start = nullptr;
	oroEvent_t spatial_reuse_time_stop = nullptr;

	GPURenderer* m_renderer = nullptr;
	// Quick access to the renderer's render_data
	HIPRTRenderData* render_data = nullptr;
};

#endif
