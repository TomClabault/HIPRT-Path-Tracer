/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_NIS_ML_RENDER_PASS_H
#define RENDERER_NIS_ML_RENDER_PASS_H

#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingOptions.h"
#include "Renderer/CPUGPUCommonDataStructures/Neural/MLPDataHost.h"
#include "Renderer/RenderPasses/RenderPass.h"

#include "Renderer/CPUGPUCommonDataStructures/Neural/NISMLDataHost.h"

class NISMLRenderPass : public RenderPass
{
public:
	static const std::string NISML_RENDER_PASS_NAME;

	NISMLRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_sample_update(float delta_time) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override;

	virtual bool is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const override;

private:
	MLPDataHost<OrochiBuffer, NeuralImportanceSamplingMLP> m_mlp;
	NISMLDataHost<OrochiBuffer> m_nis_ml_data;
};

#endif
