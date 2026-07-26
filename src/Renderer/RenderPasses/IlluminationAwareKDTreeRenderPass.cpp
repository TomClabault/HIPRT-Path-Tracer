/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/IlluminationAwareKDTreeRenderPass.h"

#include "HostDeviceCommon/RenderData.h"

const std::string IlluminationAwareKDTreeRenderPass::ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME = "Illumination-Aware KD-Tree Render Pass";

IlluminationAwareKDTreeRenderPass::IlluminationAwareKDTreeRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(IlluminationAwareKDTreeRenderPass::ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME, renderer, options)
{
}

void IlluminationAwareKDTreeRenderPass::resize(unsigned int new_width, unsigned int new_height) {}

bool IlluminationAwareKDTreeRenderPass::pre_render_update(float delta_time)
{
	return false;
}

bool IlluminationAwareKDTreeRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	return false;
}

void IlluminationAwareKDTreeRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) {}

void IlluminationAwareKDTreeRenderPass::update_render_data() {}

void IlluminationAwareKDTreeRenderPass::reset(bool reset_by_camera_movement)
{
	m_renderer->get_render_data().illumination_aware_kd_tree = {};
}

bool IlluminationAwareKDTreeRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	return true;
}
