/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_WAVEFRONT_RENDER_PASS_H
#define RENDERER_WAVEFRONT_RENDER_PASS_H

#include "Renderer/RenderPasses/RenderPass.h"
#include "Renderer/RenderPasses/WavefrontRenderPassCommon.h"

#include <map>
#include <memory>
#include <string>

class WavefrontRenderPass : public RenderPass
{
public:
	static const std::string WAVEFRONT_RENDER_PASS_NAME;
	static const std::string SHADE_PRIMARY_PATHS_KERNEL;
	static const std::string SHADE_PRIMARY_PATHS_DIFFUSE_KERNEL;
	static const std::string SHADE_PRIMARY_PATHS_GLASS_KERNEL;
	static const std::string SHADE_PRIMARY_PATHS_SINGLE_METALLIC_KERNEL;
	static const std::string SHADE_PRIMARY_PATHS_SPECULAR_DIFFUSE_KERNEL;
	static const std::string SHADE_PATHS_KERNEL;
	static const std::string SHADE_PATHS_DIFFUSE_KERNEL;
	static const std::string SHADE_PATHS_GLASS_KERNEL;
	static const std::string SHADE_PATHS_SINGLE_METALLIC_KERNEL;
	static const std::string SHADE_PATHS_SPECULAR_DIFFUSE_KERNEL;
	static const std::string COMPLETE_DEFERRED_PATHS_KERNEL;
	static const std::string COMPLETE_DEFERRED_PATHS_DIFFUSE_KERNEL;
	static const std::string COMPLETE_DEFERRED_PATHS_GLASS_KERNEL;
	static const std::string COMPLETE_DEFERRED_PATHS_SINGLE_METALLIC_KERNEL;
	static const std::string COMPLETE_DEFERRED_PATHS_SPECULAR_DIFFUSE_KERNEL;
	static const std::string TRACE_PATHS_KERNEL;
	static const std::string TRACE_SHADOW_RAYS_KERNEL;
	static const std::string NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL;
	static const std::string MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL;

	WavefrontRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);
	~WavefrontRenderPass() = default;

	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
											  const std::vector<hiprtFuncNameSet>& func_name_sets,
											  bool silent,
											  bool use_cache) override;
	virtual void recompile(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
						   const std::vector<hiprtFuncNameSet>& func_name_sets = {},
						   bool silent										   = false,
						   bool use_cache									   = true) override;

	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_frame_render_update(float delta_time) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override {};

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override;

	virtual bool is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const override;
	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_all_kernels() override;
	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_tracing_kernels() override;

	bool get_overlap_material_family_kernels() const;
	void set_overlap_material_family_kernels(bool overlap_material_family_kernels);

private:
	enum class MaterialFamilyKernelType
	{
		PRIMARY_SHADING,
		SECONDARY_SHADING,
		DEFERRED_COMPLETION
	};

	const std::string& get_primary_shading_kernel_name(unsigned int family_index) const;
	const std::string& get_secondary_shading_kernel_name(unsigned int family_index) const;
	const std::string& get_deferred_completion_kernel_name(unsigned int family_index) const;
	bool kernels_ready() const;
	const std::string& get_material_family_kernel_name(MaterialFamilyKernelType kernel_type, unsigned int family_index) const;
	WavefrontRenderPassCommon::MaterialFamilyKernels get_material_family_kernels(MaterialFamilyKernelType kernel_type) const;
	WavefrontRenderPassCommon::MaterialFamilyCountReadbackSlot get_material_family_count_readback_slot(MaterialFamilyKernelType kernel_type) const;
	void launch_material_family_kernels_async(
		MaterialFamilyKernelType kernel_type, void** launch_args, unsigned int material_family_count, unsigned int block_count, bool overlap);

	int2_t m_render_resolution = make_int2(0, 0);

	WavefrontRenderPassCommon m_wavefront_common;
};

#endif // #ifndef RENDERER_WAVEFRONT_RENDER_PASS_H
