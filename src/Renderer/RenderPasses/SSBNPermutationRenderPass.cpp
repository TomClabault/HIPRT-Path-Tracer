/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/RenderPasses/SSBNPermutationRenderPass.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "UI/RenderWindow.h"

const std::string SSBNPermutationRenderPass::SSBN_PERMUTATION_RENDER_PASS_NAME = "SSBN Permutation Render Pass";
const std::string SSBNPermutationRenderPass::SSBN_PERMUTATION_SORTING_PASS	   = "SSBN Permutation Sorting Pass";
const std::string SSBNPermutationRenderPass::SSBN_PERMUTATION_RETARGETING_PASS = "SSBN Permutation Retargeting Pass";

SSBNPermutationRenderPass::SSBNPermutationRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(renderer, options, SSBNPermutationRenderPass::SSBN_PERMUTATION_RENDER_PASS_NAME)
{
	/*m_kernels[SSBNPermutationRenderPass::COMPUTE_GMON_KERNEL] = std::make_shared<GPUKernel>();
	m_kernels[SSBNPermutationRenderPass::COMPUTE_GMON_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/GMoN/GMoNComputeMedianOfMeans.h");
	m_kernels[SSBNPermutationRenderPass::COMPUTE_GMON_KERNEL]->set_kernel_function_name("GMoNComputeMedianOfMeans");
	m_kernels[SSBNPermutationRenderPass::COMPUTE_GMON_KERNEL]->synchronize_options_with(m_compiler_options, {});*/
}

bool SSBNPermutationRenderPass::pre_render_update(float delta_time)
{
	return false;
}

bool SSBNPermutationRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	return false;
}

void SSBNPermutationRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) {}

void SSBNPermutationRenderPass::reset(bool reset_by_camera_movement) {}

void SSBNPermutationRenderPass::update_render_data() {}

bool SSBNPermutationRenderPass::is_render_pass_used() const
{
	return m_using_ssbn_permutation;
}

std::map<std::string, std::shared_ptr<GPUKernel>> SSBNPermutationRenderPass::get_tracing_kernels()
{
	return {};
}
