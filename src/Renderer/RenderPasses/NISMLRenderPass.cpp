/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/NISMLRenderPass.h"

const std::string NISMLRenderPass::NISML_RENDER_PASS_NAME = "Neural Importance Sampling MLP";

NISMLRenderPass::NISMLRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(NISMLRenderPass::NISML_RENDER_PASS_NAME, renderer, options)
{
}

void NISMLRenderPass::resize(unsigned int new_width, unsigned int new_height) {}

bool NISMLRenderPass::pre_sample_update(float delta_time)
{
	return false;
}

bool NISMLRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	return false;
}

void NISMLRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) {}

void NISMLRenderPass::update_render_data() {}

void NISMLRenderPass::reset(bool reset_by_camera_movement) {}

bool NISMLRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	return false;
}
