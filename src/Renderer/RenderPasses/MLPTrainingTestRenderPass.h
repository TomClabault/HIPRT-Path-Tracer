/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef MLP_TRAINING_TEST_RENDER_PASS_H
#define MLP_TRAINING_TEST_RENDER_PASS_H

#include "Renderer/CPUGPUCommonDataStructures/Neural/MLPDataHost.h"
#include "Renderer/RenderPasses/RenderPass.h"

class MLPTrainingTestRenderPass : public RenderPass
{
public:
	static const std::string MLP_TRAINING_TEST_RENDER_PASS_NAME;
	static const std::string MLP_TRAIN;
	static const std::string MLP_OPTIMIZE;
	static const std::string MLP_PREDICT;

	MLPTrainingTestRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);
	MLPTrainingTestRenderPass(const std::string& name, GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
											  const std::vector<hiprtFuncNameSet>& func_name_sets,
											  bool silent,
											  bool use_cache) override;

	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_render_update(float delta_time) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override {};

	virtual void update_render_data() override {};
	virtual void reset(bool reset_by_camera_movement) override;

	virtual bool is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const override;

private:
	MLPDataHost<OrochiBuffer> m_mlp;
	Image8Bit m_apple;
	OrochiBuffer<unsigned char> m_texture_data;
	OrochiBuffer<unsigned char> m_out_predicted_texture;
};

#endif
