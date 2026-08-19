/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_NISML_MEGA_KERNEL_RENDER_PASS_H
#define RENDERER_NISML_MEGA_KERNEL_RENDER_PASS_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeOptions.h"
#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingManyLightsOptions.h"
#include "HostDeviceCommon/NISMLMegaKernelData.h"
#include "Renderer/RenderPasses/RenderPass.h"

#include <cstddef>
#include <memory>
#include <string>

class NISMLMegaKernelRenderPass : public RenderPass
{
public:
	static const std::string NISML_MEGA_KERNEL_RENDER_PASS_NAME;
	static const std::string GENERATE_QUERIES_KERNEL;
	static const std::string INFERENCE_KERNEL;
	static const std::string RESUME_KERNEL;

	NISMLMegaKernelRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);
	~NISMLMegaKernelRenderPass() = default;

	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
											  const std::vector<hiprtFuncNameSet>& func_name_sets,
											  bool silent,
											  bool use_cache) override;

	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_sample_update(float delta_time) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override;

	virtual bool is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const override;
	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_tracing_kernels() override;

private:
	bool resize_staging_buffers();
	void free_staging_buffers();
	NISMLMegaKernelDevice get_device_data();

	int2_t m_render_resolution					  = make_int2(0, 0);
	bool m_staging_buffers_allocated			  = false;
	size_t m_allocated_ray_volume_state_byte_size = 0;

	OrochiBuffer<HIPRTRenderData> m_render_data_host_pinned;

	OrochiBuffer<NISMLMegaKernelPathData> m_path_data;
	OrochiBuffer<NISMLMegaKernelPathState> m_path_states;
	OrochiBuffer<RayVolumeState> m_path_volume_states;

	OrochiBuffer<NISQuery> m_queries;
	OrochiBuffer<float> m_residuals;
	OrochiBuffer<NISResult> m_results;

	OrochiBuffer<unsigned int> m_query_count_host_pinned;
	OrochiBuffer<unsigned int> m_query_count_device;
};

#endif
