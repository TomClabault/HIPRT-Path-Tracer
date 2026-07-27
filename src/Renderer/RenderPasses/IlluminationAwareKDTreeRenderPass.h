/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_H

#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeDataHost.h"
#include "Renderer/RenderPasses/RenderPass.h"

#include <cstdint>

class IlluminationAwareKDTreeRenderPass : public RenderPass
{
public:
	static const std::string ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME;
	static const std::string INITIALIZE_ROOT_NODE_KERNEL_ID;
	static const std::string ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID;
	static const std::string ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID;
	static const std::string RESET_BATCH_STATISTICS_KERNEL_ID;
	static const std::string EXPAND_ONE_LOOKAHEAD_LEVEL_KERNEL_ID;
	static const std::string REPLAY_TRAINING_SAMPLES_KERNEL_ID;
	static const std::string INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID;
	static const std::string MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID;

	IlluminationAwareKDTreeRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	virtual void resize(unsigned int new_width, unsigned int new_height) override;
	virtual bool pre_render_update(float delta_time) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override;
	virtual bool is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const override;

private:
	static constexpr float MEAN_RADIANCE_THRESHOLD			= 0.05f;
	static constexpr float MEAN_DIRECTION_THRESHOLD_DEGREES = 3.0f;
	static constexpr double FALSE_POSITIVE_PROBABILITY		= 1.0e-4;

	IlluminationAwareKDTreeSubdivisionMode m_subdivision_mode = IlluminationAwareKDTreeSubdivisionMode::DISABLED;

	IlluminationAwareKDTreeDataHost<OrochiBuffer> m_illumination_aware_kd_tree;
	bool m_lookahead_frontier_initialized	  = false;
	bool m_current_frontier_uses_first_buffer = true;
	uint32_t m_next_creation_tag			  = 0;
};

#endif
