/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/kernels/ReSTIR/SPMIS/BuildCDFs.h"
#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/ReSTIRPTRenderPass.h"
#include "Renderer/RenderPasses/ReSTIRRenderPassCommon.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

#include <ranges>
const std::string ReSTIRPTRenderPass::RESTIR_PT_RENDER_PASS_NAME					   = "ReSTIR PT Render pass";
const std::string ReSTIRPTRenderPass::RESTIR_PT_INITIAL_CANDIDATES_KERNEL_ID		   = "ReSTIR PT Initial candidates";
const std::string ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID			   = "ReSTIR PT Temporal reuse";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID				   = "ReSTIR PT Spatial reuse";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID		   = "ReSTIR PT Spatial reuse SPMIS";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID					   = "ReSTIR PT Shading";
const std::string ReSTIRPTRenderPass::RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID	   = "ReSTIR PT Directional reuse compute";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID		   = "ReSTIR PT SPMIS Reset Buffers";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID	   = "ReSTIR PT SPMIS Reset Cells Data";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID			   = "ReSTIR PT SPMIS Count Cells";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID		   = "ReSTIR PT SPMIS Compute Offsets";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID				   = "ReSTIR PT SPMIS Sort";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID			   = "ReSTIR PT SPMIS Build CDFs";
const std::string ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID = "ReSTIR PT SPMIS Compute Per Cell Noise";

const std::unordered_map<std::string, std::string> ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES = {
	{ RESTIR_PT_INITIAL_CANDIDATES_KERNEL_ID, "ReSTIR_PT_InitialCandidates" },
	{ RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID, "ReSTIR_PT_TemporalReuse" },
	{ RESTIR_PT_SPATIAL_REUSE_KERNEL_ID, "ReSTIR_PT_SpatialReuse" },
	{ RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID, "ReSTIR_PT_SpatialReuseSPMIS" },
	{ RESTIR_PT_SHADING_KERNEL_ID, "ReSTIR_PT_Shading" },
	{ RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID, ReSTIRRenderPassCommon::DIRECTIONAL_REUSE_KERNEL_FUNCTION_NAME },
	{ RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID, "ReSTIR_SPMIS_ResetBuffers" },
	{ RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID, "ReSTIR_SPMIS_ResetCellsData" },
	{ RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID, "ReSTIR_SPMIS_CountCells" },
	{ RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID, "ReSTIR_SPMIS_ComputeOffsets" },
	{ RESTIR_PT_SPMIS_SORT_KERNEL_ID, "ReSTIR_SPMIS_Sort" },
	{ RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID, "ReSTIR_SPMIS_BuildCDFs" },
	{ RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID, "ReSTIR_SPMIS_ComputeCellsVariance" }
};

const std::unordered_map<std::string, std::string> ReSTIRPTRenderPass::KERNEL_FILES = {
	{ RESTIR_PT_INITIAL_CANDIDATES_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/InitialCandidates.h" },
	{ RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/TemporalReuse.h" },
	{ RESTIR_PT_SPATIAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/SpatialReuse.h" },
	{ RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/SpatialReuseSPMIS.h" },
	{ RESTIR_PT_SHADING_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/PT/Shading.h" },
	{ RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID, ReSTIRRenderPassCommon::DIRECTIONAL_REUSE_KERNEL_FILE },
	{ RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/ResetBuffers.h" },
	{ RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/ResetCellsData.h" },
	{ RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/CountCells.h" },
	{ RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/ComputeOffsets.h" },
	{ RESTIR_PT_SPMIS_SORT_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/Sort.h" },
	{ RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/BuildCDFs.h" },
	{ RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/SPMIS/ComputeCellsVariance.h" }
};

ReSTIRPTRenderPass::ReSTIRPTRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: MegaKernelRenderPass(ReSTIRPTRenderPass::RESTIR_PT_RENDER_PASS_NAME, renderer, options)
{
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_INITIAL_CANDIDATES_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_INITIAL_CANDIDATES_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_INITIAL_CANDIDATES_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_INITIAL_CANDIDATES_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_INITIAL_CANDIDATES_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_INITIAL_CANDIDATES_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_INITIAL_CANDIDATES_KERNEL_ID]->synchronize_options_with(m_compiler_options,
																									GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_INITIAL_CANDIDATES_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_INITIAL_CANDIDATES_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->synchronize_options_with(m_compiler_options,
																								GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->synchronize_options_with(m_compiler_options,
																							   GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->synchronize_options_with(m_compiler_options,
																									 GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID]->synchronize_options_with(m_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL,
																									 KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE,
																									 8);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->synchronize_options_with(m_compiler_options);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->get_kernel_options().set_macro_value(
		ReSTIRRenderPassCommon::DIRECTIONAL_REUSE_RESTIR_VARIANT_COMPILE_OPTION_NAME, ReSTIR_VARIANT_PT);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID]->synchronize_options_with(m_compiler_options);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID]->synchronize_options_with(m_compiler_options);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID]->synchronize_options_with(m_compiler_options);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID]->synchronize_options_with(m_compiler_options);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID]->synchronize_options_with(m_compiler_options);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID]->synchronize_options_with(m_compiler_options);

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID);
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID]->set_kernel_file_path(
		ReSTIRPTRenderPass::KERNEL_FILES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID]->set_kernel_function_name(
		ReSTIRPTRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID));
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID]->synchronize_options_with(m_compiler_options);
}

void ReSTIRPTRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	if (!is_render_pass_used(*m_compiler_options))
		return;

	m_initial_candidates_buffer.resize(new_width * new_height);
	m_temporal_buffer.resize(new_width * new_height);
	m_spatial_buffer.resize(new_width * new_height);

	m_directional_spatial_reuse_data.resize(new_width, new_height);
	m_spmis_data.resize(new_width, new_height);
}

bool ReSTIRPTRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
													  const std::vector<hiprtFuncNameSet>& func_name_sets,
													  bool silent,
													  bool use_cache)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used(*m_compiler_options))
		return false;

	bool recompiled = false;

	bool need_temporal = render_data.render_settings.restir_pt_settings.common_temporal_pass.do_temporal_reuse_pass &&
						 !m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->has_been_compiled();
	recompiled |= need_temporal;
	if (need_temporal)
		// Temporal needed
		m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool need_spatial = render_data.render_settings.restir_pt_settings.common_spatial_pass.do_spatial_reuse_pass &&
						!m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->has_been_compiled();
	recompiled |= need_spatial;
	if (need_spatial)
		// Spatial needed
		m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool need_directional_spatial_reuse = !m_kernels[ReSTIRPTRenderPass::RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->has_been_compiled() &&
										  render_data.render_settings.restir_pt_settings.common_spatial_pass.use_adaptive_directional_spatial_reuse;
	recompiled |= need_directional_spatial_reuse;
	if (need_directional_spatial_reuse)
		// Spatial needed
		m_kernels[ReSTIRPTRenderPass::RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool using_spmis = (m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) ==
							RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS ||
						m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) ==
							RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE);
	if (using_spmis)
	{
		if (!m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->has_been_compiled())
		{
			// SPMIS has never been compiled but we need it now
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

			recompiled = true;
		}
	}

	return recompiled;
}

bool ReSTIRPTRenderPass::pre_sample_update(float delta_time)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	MegaKernelRenderPass::pre_sample_update(delta_time);

	bool render_data_invalidated = false;

	int2_t render_resolution = m_renderer->m_render_resolution;

	if (is_render_pass_used(*m_compiler_options))
	{
		// ReSTIR PT enabled
		bool initial_candidates_reservoir_needs_resize	= m_initial_candidates_buffer.size() == 0;
		bool temporal_candidates_reservoir_needs_resize = m_temporal_buffer.size() == 0;
		bool spatial_candidates_reservoir_needs_resize	= m_spatial_buffer.size() == 0;

		if (initial_candidates_reservoir_needs_resize || temporal_candidates_reservoir_needs_resize || spatial_candidates_reservoir_needs_resize)
			// At least on buffer is going to be resized so buffers are invalidated
			render_data_invalidated = true;

		if (initial_candidates_reservoir_needs_resize)
			m_initial_candidates_buffer.resize(render_resolution.x * render_resolution.y);

		if (temporal_candidates_reservoir_needs_resize)
			m_temporal_buffer.resize(render_resolution.x * render_resolution.y);

		if (spatial_candidates_reservoir_needs_resize)
			m_spatial_buffer.resize(render_resolution.x * render_resolution.y);

		render_data_invalidated |= ReSTIRRenderPassCommon::pre_render_update_common_buffers<ReSTIR_VARIANT_PT>(render_data, m_directional_spatial_reuse_data);
		render_data_invalidated |= pre_render_update_spmis_buffers(render_data, *m_renderer->get_global_compiler_options());

		// Arbitrary setting this one so that we're sure it's pointing to a valid buffer when all the buffers are resized
		m_last_temporal_output_reservoirs = m_initial_candidates_buffer.get_device_pointer();
	}
	else
	{
		// ReSTIR PT disabled, we're going to free the buffers if that's not already done
		if (m_initial_candidates_buffer.size() > 0)
		{
			m_initial_candidates_buffer.free();

			render_data_invalidated = true;
		}

		if (m_temporal_buffer.size() > 0)
		{
			m_temporal_buffer.free();

			render_data_invalidated = true;
		}

		if (m_spatial_buffer.size() > 0)
		{
			m_spatial_buffer.free();

			render_data_invalidated = true;
		}

		render_data_invalidated |= ReSTIRRenderPassCommon::free_common_buffers<ReSTIR_VARIANT_PT>(m_directional_spatial_reuse_data);
		render_data_invalidated |= m_spmis_data.free();
	}

	if (render_data.render_settings.restir_pt_settings.common_spatial_pass.auto_reuse_radius)
		// A percentage of the maximum render resolution extent for automatic spatial reuse radius
		render_data.render_settings.restir_pt_settings.common_spatial_pass.reuse_radius =
			hippt::max(m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y) *
			ReSTIRRenderPassCommon::AUTO_SPATIAL_RADIUS_RESOLUTION_PERCENTAGE;

	return render_data_invalidated;
}

bool ReSTIRPTRenderPass::pre_render_update_spmis_buffers(const HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	int mis_weight_type = compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE);

	bool render_data_invalidated = false;
	if (mis_weight_type == RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS || mis_weight_type == RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE)
	{
		if (m_spmis_data.maximum_size() == 0)
		{
			m_spmis_data.resize(render_data.render_settings.render_resolution.x, render_data.render_settings.render_resolution.y);

			render_data_invalidated = true;
		}
	}
	else
	{
		if (m_spmis_data.maximum_size() > 0)
		{
			m_spmis_data.free();

			render_data_invalidated = true;
		}
	}

	return render_data_invalidated;
}

void ReSTIRPTRenderPass::compute_optimal_spatial_reuse_radii(HIPRTRenderData& render_data)
{
	bool accumulating							  = render_data.render_settings.accumulate;
	bool first_frame							  = render_data.render_settings.sample_number == 0;
	bool not_interacting						  = render_data.render_settings.wants_render_low_resolution == false;
	bool using_adaptive_directional_spatial_reuse = render_data.render_settings.restir_pt_settings.common_spatial_pass.use_adaptive_directional_spatial_reuse;

	if (accumulating && first_frame && not_interacting && using_adaptive_directional_spatial_reuse)
	{
		// If we're not accumulating, we have no guarantee that the camera isn't moving and so
		// there isn't really an "optimal" reuse radius per pixel to find
		//
		// But if the camera isn't moving, then the neighborhood of a pixel is fixed and we can optimize
		// the best spatial reuse radius
		//
		// Also, we're only doing this as a "prepass" at sample 0: we only need this once for the whole rendering

		unsigned long long int* per_pixel_spatial_reuse_direction_mask_ull =
			m_directional_spatial_reuse_data.m_spatial_reuse_data
				.get_buffer_data_ptr<ReSTIRDirectionalSpatialReuseDataHostBuffers::RESTIR_DIRECTIONAL_SPATIAL_REUSE_DIRECTION_MASK_ULL>();
		unsigned char* per_pixel_spatial_reuse_radius =
			m_directional_spatial_reuse_data.m_spatial_reuse_data
				.get_buffer_data_ptr<ReSTIRDirectionalSpatialReuseDataHostBuffers::RESTIR_DIRECTIONAL_SPATIAL_REUSE_RADIUS>();
		void* launch_args[] = { &render_data, &per_pixel_spatial_reuse_direction_mask_ull, &per_pixel_spatial_reuse_radius };

		m_kernels[ReSTIRPTRenderPass::RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->launch_asynchronous(
			KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args,
			m_renderer->get_main_stream());
	}
}

void ReSTIRPTRenderPass::configure_initial_candidates_pass(HIPRTRenderData& render_data)
{
	render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer = m_initial_candidates_buffer.get_device_pointer();
}

void ReSTIRPTRenderPass::launch_initial_candidates_pass(HIPRTRenderData& render_data)
{
	void* launch_args[] = { &render_data };

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_INITIAL_CANDIDATES_KERNEL_ID]->launch_asynchronous(
		KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args,
		m_renderer->get_main_stream());
}

void ReSTIRPTRenderPass::launch_spmis_create_reuse_cells_pass(HIPRTRenderData& render_data,
															  GPUKernelCompilerOptions& compiler_options,
															  ReSTIRPTReservoir* input_reservoirs)
{
	if (compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) != RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS &&
		compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) != RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE)
		return;

	const ReSTIRPTSPMISSettings& spmis_settings = render_data.render_settings.restir_pt_settings.spmis_settings;

	unsigned int num_cells			   = spmis_settings.pixel_hashes_count;
	void* reset_counters_launch_args[] = { &render_data, &num_cells };
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, 1, num_cells, 1,
																								   reset_counters_launch_args, m_renderer->get_main_stream());

	unsigned short int* cell_counters = m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_COUNTERS>().get_device_pointer();
	unsigned short int* cell_non_zero_reservoir_counters =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_NON_ZERO_RESERVOIR_COUNTERS>().get_device_pointer();
	unsigned int* cell_confidence_sums =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_CONFIDENCE_SUMS>().get_device_pointer();
	unsigned int* all_pixels_hashes = m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_ALL_PIXEL_HASHES>().get_device_pointer();
	unsigned int* all_pixels_index_in_cell =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_ALL_PIXEL_INDEX_IN_CELL>().get_device_pointer();
	bool count_important			= true; // We want to count the important pixels first so that they are sorted at the beginning of their cell
	void* count_cells_launch_args[] = { &all_pixels_hashes,	   &all_pixels_index_in_cell, &cell_counters, &cell_non_zero_reservoir_counters,
										&cell_confidence_sums, &input_reservoirs,		  &num_cells,	  &count_important };
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, 1, num_cells, 1, count_cells_launch_args,
																							  m_renderer->get_main_stream());

	// And then a second call counting the non-important pixels so that they are sorted after the important ones in their cell
	count_important = false;
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, 1, num_cells, 1, count_cells_launch_args,
																							  m_renderer->get_main_stream());

	unsigned int* cell_global_offset_counter =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_GLOBAL_OFFSET_COUNTER>().get_device_pointer();
	unsigned int* cell_offsets			= m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_OFFSETS>().get_device_pointer();
	void* compute_offsets_launch_args[] = { &cell_counters, &cell_global_offset_counter, &cell_offsets, &num_cells };
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, 1, num_cells, 1,
																								  compute_offsets_launch_args, m_renderer->get_main_stream());

	unsigned int* pixel_indices_sorted =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_PIXEL_INDICES_SORTED>().get_device_pointer();
	void* sort_launch_args[] = { &all_pixels_hashes, &all_pixels_index_in_cell, &cell_offsets, &pixel_indices_sorted, &num_cells };
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_SORT_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, 1, num_cells, 1, sort_launch_args,
																					   m_renderer->get_main_stream());

	float* cell_cdfs				 = m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_CDFS>().get_device_pointer();
	unsigned int* cell_alive_list	 = m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_ALIVE_LIST>().get_device_pointer();
	unsigned int cell_size			 = spmis_settings.tile_size;
	unsigned int dispatch_block_size = cell_size * cell_size;
	unsigned int cell_alive_count =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_TOTAL_COUNT_COUNTER>().download_data()[0];

	resize_cdf_luts_buffer(render_data, cell_alive_count);

	unsigned short int* cell_cdf_luts = m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_CDF_LUTS>().get_device_pointer();
	unsigned int* cell_cdf_lut_offsets =
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_CDF_LUT_OFFSETS>().get_device_pointer();
	void* build_cdfs_launch_args[] = { &cell_non_zero_reservoir_counters,
									   &cell_offsets,
									   &cell_alive_list,
									   &pixel_indices_sorted,
									   &input_reservoirs,
									   &cell_cdfs,
									   &cell_cdf_luts,
									   &cell_cdf_lut_offsets,
									   &num_cells };

	// Always dispatching 1024 sized blocks for the build cdfs kernel, since the kernel is designed to handle that many threads per cell
	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID]->launch_asynchronous(1024, 1, cell_alive_count * 1024, 1, build_cdfs_launch_args,
																							 m_renderer->get_main_stream());

	// Variance is only needed for the first frame to seed the cached cell search radius;
	// subsequent frames reuse the cached cell index and never read cell_variance.
	if (render_data.render_settings.sample_number == 0)
	{
		float* cell_variance			  = m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_VARIANCE>().get_device_pointer();
		void* compute_noise_launch_args[] = { &cell_counters,	 &cell_offsets,	 &cell_alive_list, &pixel_indices_sorted,
											  &input_reservoirs, &cell_variance, &num_cells };

		// Same dispatch as BuildCDFs: one block per alive cell, 1024 threads/block
		m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID]->launch_asynchronous(
			1024, 1, cell_alive_count * 1024, 1, compute_noise_launch_args, m_renderer->get_main_stream());
	}
}

void ReSTIRPTRenderPass::resize_cdf_luts_buffer(HIPRTRenderData& render_data, unsigned int cell_alive_count)
{
	if (m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_CDF_LUTS>().size() < cell_alive_count * ReSTIR_PT_SPMISCDFLUTSize)
	{
		// Resizing the buffer to get enough space for the CDF LUTs of all cells
		m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_CDF_LUTS>().resize(cell_alive_count * ReSTIR_PT_SPMISCDFLUTSize *
																											  1.05f);

		// Settings the pointer in both the render data and the renderer render data to the new buffer pointer
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_cdf_luts =
			m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_CDF_LUTS>().get_device_pointer();
		m_renderer->get_render_data().render_settings.restir_pt_settings.spmis_settings.cell_cdf_luts =
			m_spmis_data.m_spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_CDF_LUTS>().get_device_pointer();
	}
}

void ReSTIRPTRenderPass::configure_temporal_reuse_pass(HIPRTRenderData& render_data)
{
	render_data.render_settings.restir_pt_settings.common_temporal_pass.temporal_buffer_clear_requested = m_temporal_buffer_clear_requested;

	ReSTIRPTReservoir* temporal_input_reservoirs;
	ReSTIRPTReservoir* temporal_output_reservoirs;

	if ((render_data.render_settings.sample_number == 0 && render_data.render_settings.accumulate) || render_data.render_settings.need_to_reset)
		// First frame, using the initial candidates as the input
		temporal_input_reservoirs = render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer;
	else
		// Not the first frame, the input to the temporal pass is the output of the last frame ReSTIR
		temporal_input_reservoirs = m_last_restir_output_reservoirs;

	// For the output, using whatever buffer isn't the one we're reading from (the input buffer)
	if (temporal_input_reservoirs == m_spatial_buffer.get_device_pointer())
		temporal_output_reservoirs = m_temporal_buffer.get_device_pointer();
	else
		temporal_output_reservoirs = m_spatial_buffer.get_device_pointer();

	render_data.render_settings.restir_pt_settings.temporal_pass.input_reservoirs  = temporal_input_reservoirs;
	render_data.render_settings.restir_pt_settings.temporal_pass.output_reservoirs = temporal_output_reservoirs;

	m_last_temporal_output_reservoirs = temporal_output_reservoirs;
}

void ReSTIRPTRenderPass::launch_temporal_reuse_pass(HIPRTRenderData& render_data)
{
	void* launch_args[] = { &render_data };

	if (render_data.render_settings.restir_pt_settings.common_temporal_pass.do_temporal_reuse_pass)
		m_kernels[ReSTIRPTRenderPass::RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID]->launch_asynchronous(
			KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args,
			m_renderer->get_main_stream());
}

void ReSTIRPTRenderPass::configure_spatial_reuse_pass(HIPRTRenderData& render_data, int spatial_pass_index)
{
	render_data.render_settings.restir_pt_settings.common_spatial_pass.spatial_pass_index = spatial_pass_index;

	// The spatial reuse pass spatially reuse on the output of the temporal pass in the 'temporal buffer' and
	// stores in the 'spatial buffer'

	ReSTIRPTReservoir* input_reservoirs;
	ReSTIRPTReservoir* output_reservoirs;

	if (spatial_pass_index > 0)
		// If this is the second spatial reuse pass or more, reading from the output of the previous pass
		input_reservoirs = render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs;
	else
	{
		// This is the first spatial reuse pass, reading from the output of the temporal pass
		// or the initial candidates depending on whether or not we have a temporal reuse pass at all

		if (render_data.render_settings.restir_pt_settings.common_temporal_pass.do_temporal_reuse_pass)
			// and we have a temporal reuse pass so we're going to read from the temporal reservoirs
			input_reservoirs = render_data.render_settings.restir_pt_settings.temporal_pass.output_reservoirs;
		else
			// and we do not have a temporal reuse pass so we're just going to read from the initial candidates
			input_reservoirs = m_initial_candidates_buffer.get_device_pointer();
	}

	// Outputting to whichever reservoir we're not reading from to avoid race conditions
	if (input_reservoirs == m_temporal_buffer.get_device_pointer())
		output_reservoirs = m_spatial_buffer.get_device_pointer();
	else
		output_reservoirs = m_temporal_buffer.get_device_pointer();

	render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs  = input_reservoirs;
	render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs = output_reservoirs;
}

void ReSTIRPTRenderPass::launch_spatial_reuse_pass(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!render_data.render_settings.restir_pt_settings.common_spatial_pass.do_spatial_reuse_pass)
		return;

	void* launch_args[] = { &render_data };

	for (int pass_index = 0; pass_index < render_data.render_settings.restir_pt_settings.common_spatial_pass.number_of_passes; pass_index++)
	{
		configure_spatial_reuse_pass(render_data, pass_index);

		if (compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) == RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS ||
			compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) == RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE)
		{
			launch_spmis_create_reuse_cells_pass(render_data, compiler_options, render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs);

			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID]->launch_asynchronous(
				KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args,
				m_renderer->get_main_stream());
		}
		else
			m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPATIAL_REUSE_KERNEL_ID]->launch_asynchronous(
				KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args,
				m_renderer->get_main_stream());
	}
}

void ReSTIRPTRenderPass::configure_shading_pass(HIPRTRenderData& render_data)
{
	if (render_data.render_settings.restir_pt_settings.common_spatial_pass.do_spatial_reuse_pass)
		render_data.render_settings.restir_pt_settings.restir_output_reservoirs = render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs;
	else if (render_data.render_settings.restir_pt_settings.common_temporal_pass.do_temporal_reuse_pass)
		render_data.render_settings.restir_pt_settings.restir_output_reservoirs =
			render_data.render_settings.restir_pt_settings.temporal_pass.output_reservoirs;
	else
		render_data.render_settings.restir_pt_settings.restir_output_reservoirs =
			render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer;

	m_last_restir_output_reservoirs = render_data.render_settings.restir_pt_settings.restir_output_reservoirs;
}

void ReSTIRPTRenderPass::launch_shading_pass(HIPRTRenderData& render_data)
{
	void* launch_args[] = { &render_data };

	m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SHADING_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight,
																					m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y,
																					launch_args, m_renderer->get_main_stream());
}

bool ReSTIRPTRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))
		return false;

	// GPUKernel records every kernel launch automatically and accumulates its execution time for the frame.

	compute_optimal_spatial_reuse_radii(render_data);

	configure_initial_candidates_pass(render_data);
	launch_initial_candidates_pass(render_data);

	configure_temporal_reuse_pass(render_data);
	launch_temporal_reuse_pass(render_data);

	launch_spatial_reuse_pass(render_data, compiler_options);

	configure_shading_pass(render_data);
	launch_shading_pass(render_data);

	return true;
}

void ReSTIRPTRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))
		return;

	// If we had requested a temporal buffers clear, this has be done by this frame so we can
	// now reset the flag
	m_temporal_buffer_clear_requested = false;

	MegaKernelRenderPass::post_sample_update_async(render_data, compiler_options);

	if (compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) == RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS ||
		compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) == RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE)
	{
		void* launch_args[] = { &render_data };
		m_kernels[ReSTIRPTRenderPass::RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID]->launch_asynchronous(
			64, 1, render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y, 1, launch_args,
			m_renderer->get_main_stream());
	}
}

void ReSTIRPTRenderPass::update_render_data()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	// Setting the pointers for use in reset_render() in the camera rays kernel
	if (is_render_pass_used(*m_compiler_options))
	{
		render_data.aux_buffers.restir_pt_reservoir_buffer_1 = m_initial_candidates_buffer.get_device_pointer();
		render_data.aux_buffers.restir_pt_reservoir_buffer_2 = m_spatial_buffer.get_device_pointer();
		render_data.aux_buffers.restir_pt_reservoir_buffer_3 = m_temporal_buffer.get_device_pointer();

		ReSTIRRenderPassCommon::update_render_data_common_buffers<ReSTIR_VARIANT_PT>(render_data, m_directional_spatial_reuse_data);
		m_spmis_data.to_device(render_data);
	}
	else
	{
		// If ReSTIR PT is disabled, setting the pointers to nullptr so that the camera rays kernel
		// for example can detect that the buffers are freed and doesn't try to reset them or do
		// anything with them (which would be invalid since we would be accessing nullptr buffers)

		render_data.aux_buffers.restir_pt_reservoir_buffer_1 = nullptr;
		render_data.aux_buffers.restir_pt_reservoir_buffer_2 = nullptr;
		render_data.aux_buffers.restir_pt_reservoir_buffer_3 = nullptr;

		render_data.render_settings.restir_pt_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull = nullptr;
		render_data.render_settings.restir_pt_settings.common_spatial_pass.per_pixel_spatial_reuse_radius			   = nullptr;
	}
}

void ReSTIRPTRenderPass::reset(bool reset_by_camera_movement)
{
	ReSTIRRenderPassCommon::reset_common_buffers<ReSTIR_VARIANT_PT>(m_directional_spatial_reuse_data);
	m_spmis_data.reset();

	MegaKernelRenderPass::reset(reset_by_camera_movement);
}

std::map<std::string, std::shared_ptr<GPUKernel>> ReSTIRPTRenderPass::get_all_kernels()
{
	if (!is_render_pass_used(*m_compiler_options))
		return std::map<std::string, std::shared_ptr<GPUKernel>>();

	return MegaKernelRenderPass::get_all_kernels();
}

std::map<std::string, std::shared_ptr<GPUKernel>> ReSTIRPTRenderPass::get_tracing_kernels()
{
	if (!is_render_pass_used(*m_compiler_options))
		return std::map<std::string, std::shared_ptr<GPUKernel>>();

	return MegaKernelRenderPass::get_all_kernels();
}

bool ReSTIRPTRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	return compiler_options.get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) == PATH_SAMPLING_RESTIR_PT;
}

void ReSTIRPTRenderPass::request_temporal_bufffers_clear()
{
	m_temporal_buffer_clear_requested = true;
}

float ReSTIRPTRenderPass::get_VRAM_usage() const
{
	return (m_initial_candidates_buffer.get_byte_size() + m_temporal_buffer.get_byte_size() + m_spatial_buffer.get_byte_size() +
			m_directional_spatial_reuse_data.get_byte_size() + m_spmis_data.get_byte_size()) /
		   1000000.0f;
}
