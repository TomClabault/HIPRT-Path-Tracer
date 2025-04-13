/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerOptions.h"
#include "Device/includes/BSDFs/SheenLTCFittedParameters.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"
#include "Renderer/Baker/GPUBaker.h"
#include "Renderer/Baker/GPUBakerConstants.h"
#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/FillGBufferRenderPass.h"
#include "Renderer/RenderPasses/MegaKernelRenderPass.h"
#include "Renderer/RenderPasses/ReSTIRGIRenderPass.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "Threads/ThreadFunctions.h"
#include "UI/RenderWindow.h"

#include <Orochi/OrochiUtils.h>

#include <condition_variable>

const std::string GPURenderer::NEE_PLUS_PLUS_CACHING_PREPASS_ID = "NEE++ Caching Prepass";

// List of partials_options that will be specific to each kernel. We don't want these partials_options
	// to be synchronized between kernels
const std::unordered_set<std::string> GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED =
{
	GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL,
	GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE,
};

const std::string GPURenderer::ALL_RENDER_PASSES_TIME_KEY = "FullFrameTime";
const std::string GPURenderer::FULL_FRAME_TIME_WITH_CPU_KEY = "FullFrameTimeWithCPU";
const std::string GPURenderer::DEBUG_KERNEL_TIME_KEY = "DebugKernelTime";

GPURenderer::GPURenderer(std::shared_ptr<HIPRTOrochiCtx> hiprt_oro_ctx, std::shared_ptr<ApplicationSettings> application_settings)
{
	// Creating buffers
	m_framebuffer = std::make_shared<OpenGLInteropBuffer<ColorRGB32F>>();
	m_denoiser_buffers.m_denoised_framebuffer = std::make_shared<OpenGLInteropBuffer<ColorRGB32F>>();
	m_denoiser_buffers.m_normals_AOV_interop_buffer = std::make_shared<OpenGLInteropBuffer<float3>>();
	m_denoiser_buffers.m_normals_AOV_no_interop_buffer = std::make_shared<OrochiBuffer<float3>>();
	m_denoiser_buffers.m_albedo_AOV_interop_buffer = std::make_shared<OpenGLInteropBuffer<ColorRGB32F>>();
	m_denoiser_buffers.m_albedo_AOV_no_interop_buffer = std::make_shared<OrochiBuffer<ColorRGB32F>>();
	m_pixels_converged_sample_count_buffer = std::make_shared<OrochiBuffer<int>>();
	//m_gmon.result_framebuffer = std::make_shared<OpenGLInteropBuffer<ColorRGB32F>>();

	m_DEBUG_SUMS.resize(1024);
	m_DEBUG_SUM_COUNT.resize(1024);

	m_hiprt_orochi_ctx = hiprt_oro_ctx;	
	m_device_properties = m_hiprt_orochi_ctx->device_properties;
	m_application_settings = application_settings;

	setup_brdfs_data();
	setup_filter_functions();
	setup_render_passes();

	/*m_render_pass_times[GPURenderer::ALL_RENDER_PASSES_TIME_KEY] = 0.0f;
	for (auto& id_to_pass : GPURenderer::KERNEL_FUNCTION_NAMES)
		m_render_pass_times[id_to_pass.first] = 0.0f;*/

	OROCHI_CHECK_ERROR(oroStreamCreate(&m_main_stream));

	// Buffer that keeps track of whether at least one ray is still alive or not
	m_status_buffers.still_one_ray_active_buffer.resize(1);
	m_status_buffers.still_one_ray_active_buffer.memset_whole_buffer(1);
	m_status_buffers.pixels_converged_count_buffer.resize(1);

	OROCHI_CHECK_ERROR(oroEventCreate(&m_frame_start_event));
	OROCHI_CHECK_ERROR(oroEventCreate(&m_frame_stop_event));
}

void GPURenderer::setup_brdfs_data()
{
	init_sheen_ltc_texture();

	load_GGX_energy_compensation_textures();
	load_glossy_dielectric_energy_compensation_textures();
	load_GGX_glass_energy_compensation_textures();
}

void GPURenderer::init_sheen_ltc_texture()
{
	// CUDA/HIP do not handle 3 channels textures so we're padding it to 4 channels
	std::vector<float> padded_ltc(32 * 32 * 4);

	for (int y = 0; y < 32; y++)
	{
		for (int x = 0; x < 32; x++)
		{
			int padded_index = (y * 32 + x) * 4;
			int non_padded_index = y * 32 + x;

			padded_ltc[padded_index + 0] = ltc_parameters_table_approximation[non_padded_index].x;
			padded_ltc[padded_index + 1] = ltc_parameters_table_approximation[non_padded_index].y;
			padded_ltc[padded_index + 2] = ltc_parameters_table_approximation[non_padded_index].z;
			padded_ltc[padded_index + 3] = 0.0f;
		}
	}

	Image32Bit sheen_ltc_params_image(padded_ltc.data(), 32, 32, 4);
	m_sheen_ltc_params = OrochiTexture(sheen_ltc_params_image, hipFilterModeLinear, hipAddressModeClamp);
}

void GPURenderer::load_GGX_energy_compensation_textures(hipTextureFilterMode filtering_mode)
{
	Image32Bit GGXEss_image = Image32Bit::read_image_hdr(BRDFS_DATA_DIRECTORY "/GGX/" + GPUBakerConstants::get_GGX_conductor_directional_albedo_texture_filename(m_render_data.bsdfs_data.GGX_masking_shadowing), 1, true);
	m_GGX_conductor_directional_albedo = OrochiTexture(GGXEss_image, filtering_mode, hipAddressModeClamp);

	m_render_data_buffers_invalidated = true;
}

void GPURenderer::load_glossy_dielectric_energy_compensation_textures(hipTextureFilterMode filtering_mode)
{
	synchronize_kernel();

	std::vector<Image32Bit> images(GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR);
	for (int i = 0; i < GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR; i++)
	{
		std::string filename = std::to_string(i) + GPUBakerConstants::get_glossy_dielectric_directional_albedo_texture_filename(m_render_data.bsdfs_data.GGX_masking_shadowing);
		std::string filepath = BRDFS_DATA_DIRECTORY "/GlossyDielectrics/" + filename;
		images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
	}
	m_glossy_dielectric_directional_albedo = OrochiTexture3D(images, filtering_mode == hipFilterModeLinear ? ORO_TR_FILTER_MODE_LINEAR : ORO_TR_FILTER_MODE_POINT, ORO_TR_ADDRESS_MODE_CLAMP);

	m_render_data_buffers_invalidated = true;
}

void GPURenderer::load_GGX_glass_energy_compensation_textures(hipTextureFilterMode filtering_mode)
{
	synchronize_kernel();

	std::vector<Image32Bit> images(GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR);
	for (int i = 0; i < GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR; i++)
	{
		std::string filename = std::to_string(i) + GPUBakerConstants::get_GGX_glass_directional_albedo_texture_filename(m_render_data.bsdfs_data.GGX_masking_shadowing);
		std::string filepath = BRDFS_DATA_DIRECTORY "/GGX/Glass/" + filename;
		images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
	}
	m_GGX_glass_directional_albedo = OrochiTexture3D(images, filtering_mode == hipFilterModeLinear ? ORO_TR_FILTER_MODE_LINEAR : ORO_TR_FILTER_MODE_POINT, ORO_TR_ADDRESS_MODE_CLAMP);

	for (int i = 0; i < GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR; i++)
	{
		std::string filename = std::to_string(i) + GPUBakerConstants::get_GGX_glass_directional_albedo_inv_texture_filename(m_render_data.bsdfs_data.GGX_masking_shadowing);
		std::string filepath = BRDFS_DATA_DIRECTORY "/GGX/Glass/" + filename;
		images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
	}
	m_GGX_glass_inverse_directional_albedo = OrochiTexture3D(images, filtering_mode == hipFilterModeLinear ? ORO_TR_FILTER_MODE_LINEAR : ORO_TR_FILTER_MODE_POINT, ORO_TR_ADDRESS_MODE_CLAMP);

	images.resize(GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR);
	for (int i = 0; i < GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR; i++)
	{
		std::string filename = std::to_string(i) + GPUBakerConstants::get_GGX_thin_glass_directional_albedo_texture_filename(m_render_data.bsdfs_data.GGX_masking_shadowing);
		std::string filepath = BRDFS_DATA_DIRECTORY "/GGX/Glass/" + filename;
		images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
	}
	m_GGX_thin_glass_directional_albedo = OrochiTexture3D(images, filtering_mode == hipFilterModeLinear ? ORO_TR_FILTER_MODE_LINEAR : ORO_TR_FILTER_MODE_POINT, ORO_TR_ADDRESS_MODE_CLAMP);

	m_render_data_buffers_invalidated = true;
}

void GPURenderer::setup_nee_plus_plus_from_scene(const Scene& scene)
{
	m_nee_plus_plus.base_grid_min_point = scene.metadata.scene_bounding_box.mini;
	m_nee_plus_plus.base_grid_max_point = scene.metadata.scene_bounding_box.maxi;
}

void GPURenderer::reset_nee_plus_plus()
{
	m_render_data.nee_plus_plus.reset_visibility_map = true;
	m_render_data.nee_plus_plus.update_visibility_map = true;

	// Resetting the counters
	if (m_nee_plus_plus.total_shadow_ray_queries.get_device_pointer() != nullptr)
	{
		m_nee_plus_plus.total_shadow_ray_queries.memset_whole_buffer(1);
		m_nee_plus_plus.shadow_rays_actually_traced.memset_whole_buffer(1);
	}
	m_nee_plus_plus.total_shadow_ray_queries_cpu = 1;
	m_nee_plus_plus.shadow_rays_actually_traced_cpu = 1;

	m_nee_plus_plus.milliseconds_before_finalizing_accumulation = NEEPlusPlusGPUData::FINALIZE_ACCUMULATION_START_TIMER;
}

void GPURenderer::compute_emissives_power_area_alias_table(const Scene& scene)
{
	compute_emissives_power_area_alias_table(
		scene.emissive_triangle_indices, 
		scene.vertices_positions, 
		scene.triangle_indices, 
		scene.material_indices,
		scene.materials,
		
		m_hiprt_scene.emissive_power_area_alias_table_probas, 
		m_hiprt_scene.emissive_power_area_alias_table_alias,
		m_render_data.buffers.emissives_power_area_alias_table);

	// Not joining the thread that does the computation here because it will
	// be joined before starting the render since this method is called during
	// the initialization of the renderer
}

void GPURenderer::recompute_emissives_power_area_alias_table()
{
	if (m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BASE_STRATEGY) != LSS_BASE_POWER_AREA)
		// Not using power-area sampling, no need to compute the alias table
		return;

	synchronize_kernel();

	std::vector<int> emissive_triangle_indices = m_hiprt_scene.emissive_triangles_indices.download_data();
	std::vector<float3> vertices_positions = m_hiprt_scene.geometry.download_vertices_positions();
	std::vector<int> triangle_indices = m_hiprt_scene.geometry.download_triangle_indices();
	std::vector<int> material_indices = m_hiprt_scene.material_indices.download_data();

	compute_emissives_power_area_alias_table(
		emissive_triangle_indices,
		vertices_positions,
		triangle_indices,
		material_indices,
		m_current_materials,

		m_hiprt_scene.emissive_power_area_alias_table_probas,
		m_hiprt_scene.emissive_power_area_alias_table_alias,
		m_render_data.buffers.emissives_power_area_alias_table);

	ThreadManager::join_threads(ThreadManager::RENDERER_COMPUTE_EMISSIVES_POWER_AREA_ALIAS_TABLE);
}

void GPURenderer::compute_emissives_power_area_alias_table(
	const std::vector<int>& emissive_triangle_indices,
	const std::vector<float3>& vertices_positions,
	const std::vector<int>& triangle_indices,
	const std::vector<int>& material_indices,
	const std::vector<CPUMaterial>& materials,

	OrochiBuffer<float>& alias_table_probas_buffer,
	OrochiBuffer<int>& alias_table_alias_buffer,
	DeviceAliasTable& power_area_alias_table)
{
	if (m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BASE_STRATEGY) != LSS_BASE_POWER_AREA)
		// Not using power-area sampling, no need to compute the alias table
		return;

	ThreadManager::add_dependency(ThreadManager::RENDERER_COMPUTE_EMISSIVES_POWER_AREA_ALIAS_TABLE, ThreadManager::SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES);
	ThreadManager::start_thread(ThreadManager::RENDERER_COMPUTE_EMISSIVES_POWER_AREA_ALIAS_TABLE, [ 
		&emissive_triangle_indices, 
		&vertices_positions,
		&triangle_indices, 
		&material_indices,
		&materials,

		&alias_table_alias_buffer,
		&alias_table_probas_buffer,
		&power_area_alias_table] ()
	{
		std::vector<float> power_area_list(emissive_triangle_indices.size());
		float power_area_sum = 0.0f;

		for (int i = 0; i < emissive_triangle_indices.size(); i++)
		{
			int emissive_triangle_index = emissive_triangle_indices[i];

			// Computing the area of the triangle
			float3 vertex_A = vertices_positions[triangle_indices[emissive_triangle_index * 3 + 0]];
			float3 vertex_B = vertices_positions[triangle_indices[emissive_triangle_index * 3 + 1]];
			float3 vertex_C = vertices_positions[triangle_indices[emissive_triangle_index * 3 + 2]];

			float3 AB = vertex_B - vertex_A;
			float3 AC = vertex_C - vertex_A;

			float3 normal = hippt::cross(AB, AC);
			float length_normal = hippt::length(normal);
			float triangle_area = 0.5f * length_normal;

			int mat_index = material_indices[emissive_triangle_index];
			float emission_luminance = materials[mat_index].emission.luminance() * materials[mat_index].emission_strength * materials[mat_index].global_emissive_factor;

			float area_power = emission_luminance * triangle_area;

			power_area_list[i] = area_power;
			power_area_sum += area_power;
		}

		std::vector<float> alias_probas;
		std::vector<int> alias_aliases;
		Utils::compute_alias_table(power_area_list, power_area_sum, alias_probas, alias_aliases);

		alias_table_probas_buffer.resize(emissive_triangle_indices.size());
		alias_table_alias_buffer.resize(emissive_triangle_indices.size());

		alias_table_probas_buffer.upload_data(alias_probas);
		alias_table_alias_buffer.upload_data(alias_aliases);

		power_area_alias_table.alias_table_probas = alias_table_probas_buffer.get_device_pointer();
		power_area_alias_table.alias_table_alias = alias_table_alias_buffer.get_device_pointer();
		power_area_alias_table.size = emissive_triangle_indices.size();
		power_area_alias_table.sum_elements = power_area_sum;
	});
}

void GPURenderer::free_emissives_power_area_alias_table()
{
	if (m_hiprt_scene.emissive_power_area_alias_table_alias.get_element_count() > 0)
		m_hiprt_scene.emissive_power_area_alias_table_alias.free();

	if (m_hiprt_scene.emissive_power_area_alias_table_probas.get_element_count() > 0)
		m_hiprt_scene.emissive_power_area_alias_table_probas.free();

	m_render_data.buffers.emissives_power_area_alias_table.alias_table_alias = nullptr;
	m_render_data.buffers.emissives_power_area_alias_table.alias_table_probas = nullptr;
	m_render_data.buffers.emissives_power_area_alias_table.size = 0;
	m_render_data.buffers.emissives_power_area_alias_table.sum_elements = 0;
}

std::shared_ptr<GMoNRenderPass> GPURenderer::get_gmon_render_pass()
{
	return std::dynamic_pointer_cast<GMoNRenderPass>(m_render_graph.get_render_pass(GMoNRenderPass::GMON_RENDER_PASS_NAME));
}

std::shared_ptr<ReSTIRDIRenderPass> GPURenderer::get_ReSTIR_DI_render_pass()
{
	return std::dynamic_pointer_cast<ReSTIRDIRenderPass>(m_render_graph.get_render_pass(ReSTIRDIRenderPass::RESTIR_DI_RENDER_PASS_NAME));
}

std::shared_ptr<ReSTIRGIRenderPass> GPURenderer::get_ReSTIR_GI_render_pass()
{
	return std::dynamic_pointer_cast<ReSTIRGIRenderPass>(m_render_graph.get_render_pass(ReSTIRGIRenderPass::RESTIR_GI_RENDER_PASS_NAME));
}

RenderGraph& GPURenderer::get_render_graph()
{
	return m_render_graph;
}

NEEPlusPlusGPUData& GPURenderer::get_nee_plus_plus_data()
{
	return m_nee_plus_plus;
}

void GPURenderer::setup_filter_functions()
{
	// Function called on intersections for handling alpha testing
	hiprtFuncNameSet alpha_testing_func_set = { nullptr, "filter_function" };
	m_func_name_sets.push_back(alpha_testing_func_set);

	hiprtFuncDataSet func_data_set;
	hiprtFuncTable func_table;
	HIPRT_CHECK_ERROR(hiprtCreateFuncTable(m_hiprt_orochi_ctx->hiprt_ctx, 1, 1, func_table));
	HIPRT_CHECK_ERROR(hiprtSetFuncTable(m_hiprt_orochi_ctx->hiprt_ctx, func_table, 0, 0, func_data_set));

	m_render_data.hiprt_function_table = func_table;
}

void GPURenderer::setup_render_passes()
{
	m_global_compiler_options = std::make_shared<GPUKernelCompilerOptions>();
	// Adding hardware acceleration by default if supported
	m_global_compiler_options->set_macro_value("__USE_HWI__", device_supports_hardware_acceleration() == HardwareAccelerationSupport::SUPPORTED);

	// Some default values are set for USE_SHARED_STACK_BVH_TRAVERSAL and SHARED_STACK_BVH_TRAVERSAL_SIZE
	// which I found work approximately well in terms of performance on various scenes (not perfect though and, on top of not 
	// being perfect, this was measured on a 7900XTX with hardware accelerated ray tracing so... your mileage in terms of what 
	// numbers are the best may vary.)
	
	// Configuring the render passes
	m_render_graph = RenderGraph(this);

	std::shared_ptr<FillGBufferRenderPass> camera_rays_render_pass = std::make_shared<FillGBufferRenderPass>(this);

	std::shared_ptr<ReSTIRDIRenderPass> restir_di_render_pass = std::make_shared<ReSTIRDIRenderPass>(this);
	restir_di_render_pass->add_dependency(camera_rays_render_pass);

	// Note that the megakernel pass will only be used if ReSTIR GI is not used.
	// But we're still adding the render pass to the render graph in case the user
	// switches from ReSTIR GI to classical path tracing at runtime
	std::shared_ptr<MegaKernelRenderPass> megakernel_render_pass = std::make_shared<MegaKernelRenderPass>(this);
	megakernel_render_pass->add_dependency(camera_rays_render_pass);
	megakernel_render_pass->add_dependency(restir_di_render_pass);

	std::shared_ptr<ReSTIRGIRenderPass> restir_gi_render_pass = std::make_shared<ReSTIRGIRenderPass>(this);
	restir_gi_render_pass->add_dependency(camera_rays_render_pass);
	restir_gi_render_pass->add_dependency(restir_di_render_pass);

	std::shared_ptr<GMoNRenderPass> gmon_render_pass  = std::make_shared<GMoNRenderPass>(this);
	// GMoN depends on the main path tracing pass which is the megakernel pass or ReSTIR GI, whichever is
	// active
	gmon_render_pass->add_dependency(megakernel_render_pass);
	gmon_render_pass->add_dependency(restir_gi_render_pass);

	m_render_graph.add_render_pass(camera_rays_render_pass);
	m_render_graph.add_render_pass(restir_di_render_pass);
	m_render_graph.add_render_pass(megakernel_render_pass);
	m_render_graph.add_render_pass(restir_gi_render_pass);
	m_render_graph.add_render_pass(gmon_render_pass);

	m_render_graph.compile(m_hiprt_orochi_ctx, m_func_name_sets);

	if (m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS) == KERNEL_OPTION_TRUE)
		m_nee_plus_plus.compile_finalize_accumulation_kernel(m_hiprt_orochi_ctx, m_func_name_sets);
}

void GPURenderer::pre_render_update(float delta_time, RenderWindow* render_window)
{
	step_animations(delta_time);

	if (m_render_graph.pre_render_compilation_check(m_hiprt_orochi_ctx, m_func_name_sets, true, true))
		// Some kernels have been recompiled, renderer is now dirty
		render_window->set_render_dirty(true);
	m_render_data_buffers_invalidated |= m_render_graph.pre_render_update(delta_time);

	internal_pre_render_update_clear_device_status_buffers();
	internal_pre_render_update_global_stack_buffer();
	internal_pre_render_update_adaptive_sampling_buffers();
	internal_pre_render_update_nee_plus_plus(delta_time);

	update_render_data();

	m_updated = true;
}

void GPURenderer::post_render_update()
{
	m_render_graph.post_render_update();

	m_render_data.render_settings.sample_number++;
	m_render_data.render_settings.denoiser_AOV_accumulation_counter++;

	// We only reset once so after rendering a frame, we're sure that we don't need to reset anymore 
	// so we're setting the flag to false (it will be set to true again if we need to reset the render
	// again)
	m_render_data.render_settings.need_to_reset = false;
	m_render_data.nee_plus_plus.reset_visibility_map = false;

	// Saving the current frame camera to be the previous camera of the next frame
	m_previous_frame_camera = m_camera;
}

void GPURenderer::step_animations(float delta_time)
{
	m_envmap.update(this, delta_time);
	m_camera_animation.animation_step(this, delta_time);
}

void GPURenderer::download_status_buffers()
{
	OROCHI_CHECK_ERROR(oroMemcpy(&m_status_buffers_values.one_ray_active, m_status_buffers.still_one_ray_active_buffer.get_device_pointer(), sizeof(unsigned char), oroMemcpyDeviceToHost));
	OROCHI_CHECK_ERROR(oroMemcpy(&m_status_buffers_values.pixel_converged_count, m_status_buffers.pixels_converged_count_buffer.get_device_pointer(), sizeof(unsigned int), oroMemcpyDeviceToHost));
}

void GPURenderer::internal_pre_render_update_clear_device_status_buffers()
{
	unsigned char false_data = false;
	unsigned int zero_data = 0;
	// Uploading false to reset the flag
	m_status_buffers.still_one_ray_active_buffer.upload_data(&false_data);
	// Resetting the counter of pixels converged to 0
	m_status_buffers.pixels_converged_count_buffer.upload_data(&zero_data);
}

void GPURenderer::internal_clear_m_status_buffers()
{
	m_status_buffers_values.one_ray_active = true;
	m_status_buffers_values.pixel_converged_count = 0;
}

void GPURenderer::internal_pre_render_update_adaptive_sampling_buffers()
{
	bool buffers_needed = m_render_data.render_settings.has_access_to_adaptive_sampling_buffers();

	if (buffers_needed)
	{
		bool pixels_squared_luminance_needs_resize = m_pixels_squared_luminance_buffer.get_element_count() == 0;
		bool pixels_sample_count_needs_resize = m_pixels_sample_count_buffer.get_element_count() == 0;
		bool pixels_converged_sample_count_needs_resize = m_pixels_converged_sample_count_buffer->get_element_count() == 0;

		if (pixels_squared_luminance_needs_resize || pixels_sample_count_needs_resize || pixels_converged_sample_count_needs_resize)
			// At least on buffer is going to be resized so buffers are invalidated
			m_render_data_buffers_invalidated = true;

		if (pixels_squared_luminance_needs_resize)
			// Only allocating if it isn't already
			m_pixels_squared_luminance_buffer.resize(m_render_resolution.x * m_render_resolution.y);

		if (pixels_sample_count_needs_resize)
			// Only allocating if it isn't already
			m_pixels_sample_count_buffer.resize(m_render_resolution.x * m_render_resolution.y);

		if (pixels_converged_sample_count_needs_resize)
			m_pixels_converged_sample_count_buffer->resize(m_render_resolution.x * m_render_resolution.y);

	}
	else
	{
		if (m_pixels_squared_luminance_buffer.get_element_count() > 0 || m_pixels_sample_count_buffer.get_element_count() > 0 || m_pixels_converged_sample_count_buffer->get_element_count() > 0)
		{
			m_pixels_squared_luminance_buffer.free();
			m_pixels_sample_count_buffer.free();
			m_pixels_converged_sample_count_buffer->free();

			m_render_data_buffers_invalidated = true;
		}
	}
}

void GPURenderer::internal_pre_render_update_nee_plus_plus(float delta_time)
{
	if (m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS) == KERNEL_OPTION_FALSE)
	{
		// Not using NEE++, we just need to free the buffers if they weren't already

		if (m_nee_plus_plus.packed_buffer.get_element_count() != 0)
		{
			m_nee_plus_plus.packed_buffer.free();
			m_nee_plus_plus.total_shadow_ray_queries.free();
			m_nee_plus_plus.shadow_rays_actually_traced.free();

			m_render_data_buffers_invalidated = true;
		}

		return;
	}

	float3 min_grid_extent_with_envmap, max_grid_extent_with_envmap;
	m_nee_plus_plus.get_grid_extents(m_nee_plus_plus.grid_dimensions_no_envmap, min_grid_extent_with_envmap, max_grid_extent_with_envmap);

	// Adding (2, 2, 2) for envmap NEE++
	m_render_data.nee_plus_plus.grid_dimensions = m_nee_plus_plus.grid_dimensions_no_envmap + make_int3(2, 2, 2);
	m_render_data.nee_plus_plus.grid_min_point = min_grid_extent_with_envmap;
	m_render_data.nee_plus_plus.grid_max_point = max_grid_extent_with_envmap;

	// Allocating / deallocating buffers
	unsigned int matrix_element_count = m_nee_plus_plus.get_visibility_matrix_element_count(m_render_data.nee_plus_plus.grid_dimensions);
	if (m_nee_plus_plus.packed_buffer.get_element_count() != matrix_element_count)
	{
		m_nee_plus_plus.packed_buffer.resize(matrix_element_count);
		m_nee_plus_plus.shadow_rays_actually_traced.resize(1);
		m_nee_plus_plus.total_shadow_ray_queries.resize(1);

		m_render_data_buffers_invalidated = true;
	}

	// Clearing the visibility map if this has been asked by the user
	if (m_render_data.nee_plus_plus.reset_visibility_map)
	{
		// Clearing the visibility map by memseting everything to 0
		m_nee_plus_plus.packed_buffer.memset_whole_buffer(0);
		m_nee_plus_plus.total_shadow_ray_queries.memset_whole_buffer(1);
		m_nee_plus_plus.shadow_rays_actually_traced.memset_whole_buffer(1);
	}

	if (m_render_data.render_settings.sample_number > m_nee_plus_plus.stop_update_samples)
		// Past a certain number of samples, there isn't really a point to keep updating, the visibility map
		// is probably converged enough that it doesn't make a difference anymore
		m_render_data.nee_plus_plus.update_visibility_map = false;

	m_nee_plus_plus.milliseconds_before_finalizing_accumulation -= delta_time;
	m_nee_plus_plus.milliseconds_before_finalizing_accumulation = hippt::max(0.0f, m_nee_plus_plus.milliseconds_before_finalizing_accumulation); // Clamping for nice display in ImGui (0.0f instead of negative values)
	if (m_nee_plus_plus.milliseconds_before_finalizing_accumulation <= 0.0f && m_render_data.nee_plus_plus.packed_buffers != nullptr)
	{
		m_nee_plus_plus.milliseconds_before_finalizing_accumulation = NEEPlusPlusGPUData::FINALIZE_ACCUMULATION_TIMER;

		// Because the visibility map data is packed, we can't just use a memcpy() to copy from the accumulation
		// buffers to the visibilit map, we have to use a kernel that the does unpacking-copy
		void* launch_args[] = { &m_render_data.nee_plus_plus };
		m_nee_plus_plus.finalize_accumulation_kernel->launch_asynchronous(256, 1, matrix_element_count, 1, launch_args, m_main_stream);
	}
	
	m_nee_plus_plus.statistics_refresh_timer -= delta_time;
	if (m_nee_plus_plus.statistics_refresh_timer <= 0.0f && m_render_data.nee_plus_plus.do_update_shadow_rays_traced_statistics)
	{
		m_nee_plus_plus.statistics_refresh_timer = NEEPlusPlusGPUData::STATISTICS_REFRESH_TIMER;

		OROCHI_CHECK_ERROR(oroMemcpy(&m_nee_plus_plus.total_shadow_ray_queries_cpu, m_nee_plus_plus.total_shadow_ray_queries.get_device_pointer(), sizeof(unsigned long long int), oroMemcpyDeviceToHost));
		OROCHI_CHECK_ERROR(oroMemcpy(&m_nee_plus_plus.shadow_rays_actually_traced_cpu, m_nee_plus_plus.shadow_rays_actually_traced.get_device_pointer(), sizeof(unsigned long long int), oroMemcpyDeviceToHost));
	}
}

void GPURenderer::internal_pre_render_update_global_stack_buffer()
{
	if (needs_global_bvh_stack_buffer())
	{
		bool buffer_needs_update = false;
		// Buffer isn't allocated
		buffer_needs_update |= m_render_data.global_traversal_stack_buffer.stackData == nullptr;
		// Buffer is allocated but the stack size has been changed (through ImGui probably)
		buffer_needs_update |= m_render_data.global_traversal_stack_buffer_size != m_render_data.global_traversal_stack_buffer.stackSize;

		if (buffer_needs_update)
			recreate_global_bvh_stack_buffer();
	}
	else
	{
		if (m_render_data.global_traversal_stack_buffer.stackData != nullptr)
		{
			// Freeing if the buffer already exists
			HIPRT_CHECK_ERROR(hiprtDestroyGlobalStackBuffer(m_hiprt_orochi_ctx->hiprt_ctx, m_render_data.global_traversal_stack_buffer));
			m_render_data.global_traversal_stack_buffer.stackData = nullptr;
		}
	}
}

bool GPURenderer::needs_global_bvh_stack_buffer()
{
	for (const auto& name_to_kernel : m_render_graph.get_tracing_kernels())
	{
		bool global_stack_buffer_needed = false;
		global_stack_buffer_needed |= name_to_kernel.second->get_kernel_options().get_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL) == KERNEL_OPTION_TRUE;

		if (global_stack_buffer_needed)
			return true;
	}

	return false;
}

void GPURenderer::recreate_global_bvh_stack_buffer()
{
	int nbBlocksX = std::ceil(m_render_resolution.x / (float)KernelBlockWidthHeight) * KernelBlockWidthHeight;
	int nbBlocksY = std::ceil(m_render_resolution.y / (float)KernelBlockWidthHeight) * KernelBlockWidthHeight;

	// Resizing the global stack buffer for BVH traversal
	hiprtGlobalStackBufferInput stackBufferInput
	{
		hiprtStackTypeGlobal,
		hiprtStackEntryTypeInteger,
		static_cast<uint32_t>(m_render_data.global_traversal_stack_buffer_size),
		static_cast<uint32_t>(nbBlocksX * nbBlocksY)
	};

	if (m_render_data.global_traversal_stack_buffer.stackData != nullptr)
		// Freeing if the buffer already exists
		HIPRT_CHECK_ERROR(hiprtDestroyGlobalStackBuffer(m_hiprt_orochi_ctx->hiprt_ctx, m_render_data.global_traversal_stack_buffer));

	HIPRT_CHECK_ERROR(hiprtCreateGlobalStackBuffer(m_hiprt_orochi_ctx->hiprt_ctx, stackBufferInput, m_render_data.global_traversal_stack_buffer));
}

void GPURenderer::render()
{
	if (!m_updated)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "render() was called on the GPURenderer without update() being called.");
		Utils::debugbreak();

		return;
	}

	// Resetting the update state since we're now rendering a new frame
	m_updated = false;

	// Making sure kernels are compiled
	ThreadManager::join_threads(ThreadManager::COMPILE_KERNELS_THREAD_KEY);

	map_buffers_for_render();
	
	if (m_debug_trace_kernel.has_been_compiled())
		render_debug_kernel();
	else
		render_path_tracing();
}

void GPURenderer::render_debug_kernel()
{
	m_frame_rendered = false;

	OROCHI_CHECK_ERROR(oroEventRecord(m_frame_start_event, m_main_stream));

	// Updating the previous and current camera
	m_render_data.current_camera = m_camera.to_hiprt();
	m_render_data.prev_camera = m_previous_frame_camera.to_hiprt();

	launch_debug_kernel();

	// Recording GPU frame time stop timestamp and computing the frame time
	OROCHI_CHECK_ERROR(oroEventRecord(m_frame_stop_event, m_main_stream));
	OROCHI_CHECK_ERROR(oroLaunchHostFunc(m_main_stream, [](void* payload) {
		*reinterpret_cast<bool*>(payload) = true;
	}, &m_frame_rendered));

	post_render_update();
}

void GPURenderer::render_path_tracing()
{
	m_frame_rendered = false;

	OROCHI_CHECK_ERROR(oroEventRecord(m_frame_start_event, m_main_stream));

	if (m_render_data.render_settings.sample_number == 0)
		// If this is the very first sample, launching the prepass
		// of all the render passes
		m_render_graph.prepass();

	for (int i = 1; i <= m_render_data.render_settings.samples_per_frame; i++)
	{
		// Updating the previous and current camera
		m_render_data.current_camera = m_camera.to_hiprt();
		m_render_data.prev_camera = m_previous_frame_camera.to_hiprt();

		if (i == m_render_data.render_settings.samples_per_frame)
			// Last sample of the frame so we are going to enable the update 
			// of the status buffers (number of pixels converged, how many rays still
			// active, ...)
			m_render_data.render_settings.do_update_status_buffers = true;
		
		m_render_graph.launch();

		post_render_update();
	}

	// Recording GPU frame time stop timestamp and computing the frame time
	OROCHI_CHECK_ERROR(oroEventRecord(m_frame_stop_event, m_main_stream));
	OROCHI_CHECK_ERROR(oroLaunchHostFunc(m_main_stream, [](void* payload){
		*reinterpret_cast<bool*>(payload) = true;
	}, &m_frame_rendered));

	m_was_last_frame_low_resolution = m_render_data.render_settings.do_render_low_resolution();
	// We just rendered a new frame so we're setting this flag to true
	// such that the animated components of the scene are not allowed to step
	// their animations until the render window signals the renderer the the
	// frame has been fully rendered and thus that the animations can step forward
	m_animation_state.can_step_animation = false;
}

void GPURenderer::launch_nee_plus_plus_caching_prepass()
{
	unsigned int caching_sample_count = 1024;
	void* launch_args[] = { &m_render_data, &caching_sample_count };

	m_render_data.random_number = m_rng.xorshift32();
	m_kernels[GPURenderer::NEE_PLUS_PLUS_CACHING_PREPASS_ID]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);
}

void GPURenderer::launch_debug_kernel()
{
	void* launch_args[] = { &m_render_data, &m_render_resolution };

	m_render_data.random_number = m_rng.xorshift32();
	m_debug_trace_kernel.launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);
}

void GPURenderer::synchronize_kernel()
{
	if (m_main_stream == nullptr)
		return;

	OROCHI_CHECK_ERROR(oroStreamSynchronize(m_main_stream));
}

bool GPURenderer::frame_render_done()
{
	return m_frame_rendered;
}

bool GPURenderer::was_last_frame_low_resolution()
{
	return m_was_last_frame_low_resolution;
}

void GPURenderer::resize(int new_width, int new_height)
{
	// Needed so that this function can eventually be called from another thread
	OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

	m_render_resolution = make_int2(new_width, new_height);

	synchronize_kernel();
	unmap_buffers();

	m_framebuffer->resize(new_width * new_height);
	m_denoiser_buffers.m_denoised_framebuffer->resize(new_width * new_height);
	m_denoiser_buffers.resize_normals_buffer(new_width * new_height);
	m_denoiser_buffers.resize_albedo_buffer(new_width * new_height);

	if (m_render_data.render_settings.has_access_to_adaptive_sampling_buffers())
		m_pixels_converged_sample_count_buffer->resize(new_width * new_height);

	if (m_render_data.render_settings.has_access_to_adaptive_sampling_buffers())
	{
		m_pixels_squared_luminance_buffer.resize(new_width * new_height);
		m_pixels_sample_count_buffer.resize(new_width * new_height);
	}

	m_render_graph.resize(new_width, new_height);

	m_pixel_active.resize(new_width * new_height);

	// Recomputing the perspective projection matrix since the aspect ratio
	// may have changed
	float new_aspect = (float)new_width / new_height;
	m_camera.set_aspect(new_aspect);

	if (needs_global_bvh_stack_buffer())
		recreate_global_bvh_stack_buffer();

	m_render_data.render_settings.render_resolution = m_render_resolution;
	m_render_data_buffers_invalidated = true;
	m_render_data.render_settings.need_to_reset = true;
}

void GPURenderer::map_buffers_for_render()
{
	m_render_data.buffers.accumulated_ray_colors = m_framebuffer->map();
	m_render_data.buffers.gmon_estimator.result_framebuffer = get_gmon_render_pass()->map_result_framebuffer();

	m_render_data.aux_buffers.denoiser_normals = m_denoiser_buffers.map_normals_buffer();
	m_render_data.aux_buffers.denoiser_albedo = m_denoiser_buffers.map_albedo_buffer();
	if (m_render_data.render_settings.has_access_to_adaptive_sampling_buffers())
		m_render_data.aux_buffers.pixel_converged_sample_count = m_pixels_converged_sample_count_buffer->get_device_pointer();
}

void GPURenderer::unmap_buffers()
{
	// TODO we should only unmap buffers that need unmapping here

	m_framebuffer->unmap();
	get_gmon_render_pass()->unmap_result_framebuffer();
	m_denoiser_buffers.unmap_normals_buffer();
	m_denoiser_buffers.unmap_albedo_buffer();
}

void GPURenderer::set_use_denoiser_AOVs_interop_buffers(bool use_interop) { m_denoiser_buffers.set_use_interop_AOV_buffers(this, use_interop); }

std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GPURenderer::get_color_interop_framebuffer() 
{ 
	// TODO use render graph here with render_graph.get_output_framebuffer()
	if (get_gmon_render_pass()->is_render_pass_used() && get_gmon_render_pass()->buffers_allocated())
		return get_gmon_render_pass()->get_result_framebuffer();
	else
		return m_framebuffer; 
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GPURenderer::get_default_interop_framebuffer()
{
	return m_framebuffer;
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GPURenderer::get_denoised_interop_framebuffer() { return m_denoiser_buffers.m_denoised_framebuffer;}
std::shared_ptr<OpenGLInteropBuffer<float3>> GPURenderer::get_denoiser_normals_AOV_interop_buffer() 
{
	if (!m_denoiser_buffers.use_interop_AOVs)
		// No using the interop buffers so let's not return a buffer that cannot be used
		return nullptr;

	return m_denoiser_buffers.m_normals_AOV_interop_buffer; 
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GPURenderer::get_denoiser_albedo_AOV_interop_buffer() 
{ 
	if (!m_denoiser_buffers.use_interop_AOVs)
		// No using the interop buffers so let's not return a buffer that cannot be used
		return nullptr;

	return m_denoiser_buffers.m_albedo_AOV_interop_buffer; 
}

std::shared_ptr<OrochiBuffer<float3>> GPURenderer::get_denoiser_normals_AOV_no_interop_buffer() { return m_denoiser_buffers.m_normals_AOV_no_interop_buffer; }
std::shared_ptr<OrochiBuffer<ColorRGB32F>> GPURenderer::get_denoiser_albedo_AOV_no_interop_buffer() { return m_denoiser_buffers.m_albedo_AOV_no_interop_buffer; }

std::shared_ptr<OrochiBuffer<int>>& GPURenderer::get_pixels_converged_sample_count_buffer() { return m_pixels_converged_sample_count_buffer; }
const StatusBuffersValues& GPURenderer::get_status_buffer_values() const { return m_status_buffers_values; }

HIPRTRenderSettings& GPURenderer::get_render_settings() { return m_render_data.render_settings; }
std::shared_ptr<ApplicationSettings> GPURenderer::get_application_settings() { return m_application_settings; }
WorldSettings& GPURenderer::get_world_settings() { return m_render_data.world_settings; }
HIPRTRenderData& GPURenderer::get_render_data() { return m_render_data; }
HIPRTScene& GPURenderer::get_hiprt_scene() { return m_hiprt_scene; }
std::shared_ptr<HIPRTOrochiCtx> GPURenderer::get_hiprt_orochi_ctx() { return m_hiprt_orochi_ctx; }

void GPURenderer::invalidate_render_data_buffers() 
{ 
	m_render_data_buffers_invalidated = true; 
}

oroDeviceProp GPURenderer::get_device_properties() { return m_device_properties;}

std::string getDeviceName(oroCtx m_ctxt, oroDevice m_device)
{
	oroDeviceProp prop;
	OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_ctxt));
	OROCHI_CHECK_ERROR(oroGetDeviceProperties(&prop, m_device));
	return std::string(prop.name);
}

std::string getGcnArchName(oroCtx m_ctxt, oroDevice m_device)
{
	oroDeviceProp prop;
	OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_ctxt));
	OROCHI_CHECK_ERROR(oroGetDeviceProperties(&prop, m_device));
	return std::string(prop.gcnArchName);
}

uint32_t getGcnArchNumber(oroCtx m_ctxt, oroDevice m_device)
{
	oroDeviceProp prop;
	OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_ctxt));
	OROCHI_CHECK_ERROR(oroGetDeviceProperties(&prop, m_device));
	return prop.gcnArch;
}

bool enableHwi(oroCtx m_ctxt, oroDevice m_device)
{
	std::string	   deviceName = getDeviceName(m_ctxt, m_device);
	const uint32_t archNumber = getGcnArchNumber(m_ctxt, m_device);
	return (archNumber >= 1030 && deviceName.find("NVIDIA") == std::string::npos);
}

HardwareAccelerationSupport GPURenderer::device_supports_hardware_acceleration()
{
	bool enabled = reinterpret_cast<hiprt::Context*>(m_hiprt_orochi_ctx->hiprt_ctx)->enableHwi();
	if (enabled)
		return HardwareAccelerationSupport::SUPPORTED;
	else
	{
		if (std::string(m_device_properties.name).find("NVIDIA") != std::string::npos)
		{
			// Not supported on NVIDIA
			return HardwareAccelerationSupport::NVIDIA_UNSUPPORTED;
		}
		else
		{
			// Not NVIDIA but hardware acceleration not supported, assuming too old AMD
			return HardwareAccelerationSupport::AMD_UNSUPPORTED;
		}
	}
}

std::shared_ptr<GPUKernelCompilerOptions> GPURenderer::get_global_compiler_options()
{
	return m_global_compiler_options;
}

// Variables used to give the priority to the main thread when compiling shaders
extern bool g_main_thread_compiling;
extern std::condition_variable g_condition_for_compilation;

void GPURenderer::recompile_kernels(bool use_cache)
{
	synchronize_kernel();

	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Recompiling kernels...");

	// Notifying all threads that may be compiling that the main thread wants to
	// compile. This will block threads other than the main thread from compiling
	// and thus give the priority to the main thread
	take_kernel_compilation_priority();

	for (auto& name_to_kenel : m_kernels)
		name_to_kenel.second->compile(m_hiprt_orochi_ctx, m_func_name_sets, use_cache, false);

	m_render_graph.recompile(m_hiprt_orochi_ctx, m_func_name_sets, false, use_cache);

	if (m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS) == KERNEL_OPTION_TRUE)
		m_nee_plus_plus.recompile(m_hiprt_orochi_ctx);

	// The main thread is done with the compilation, we can release the other threads
	// so that they can continue compiling (background compilation of shaders most likely)
	release_kernel_compilation_priority();
}

// Variables used to give the priority to the main thread when compiling shaders
extern std::thread::id g_priority_thread_id;
extern bool g_main_thread_compiling;
extern std::condition_variable g_condition_for_compilation;
void GPURenderer::take_kernel_compilation_priority()
{
	// Notifying all threads that may be compiling that the main thread wants to
	// compile. This will block threads other than the main thread from compiling
	// and thus give the priority to the main thread
	g_main_thread_compiling = true;
	g_condition_for_compilation.notify_all();
	g_priority_thread_id = std::this_thread::get_id();
}

void GPURenderer::release_kernel_compilation_priority()
{
	// The main thread is done with the compilation, we can release the other threads
	// so that they can continue compiling (background compilation of shaders most likely)
	g_main_thread_compiling = false;
	g_condition_for_compilation.notify_all();
}

void GPURenderer::precompile_kernels()
{
	return;

	g_imgui_logger.add_line_with_name(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, ImGuiLogger::BACKGROUND_KERNEL_PARSING_LINE_NAME, "Parsing kernel permutations in the background... [%d / %d]", 0, 1);
	g_imgui_logger.add_line_with_name(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, ImGuiLogger::BACKGROUND_KERNEL_COMPILATION_LINE_NAME, "Compiling kernel permutations in the background... [%d / %d]", 0, 1);

	// Launching all the threads actually takes some time
	// so we're launching threads from a thread :D
	// 
	// We're not going to join the thread started right below
	// so we can use a const char* for the key, we don't a constant
	// defined in ThreadManager. Quick and dirty.
	ThreadManager::start_thread(ThreadManager::GPU_RENDERER_PRECOMPILE_KERNELS_THREAD_KEY, [this]() {
		OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

		precompile_direct_light_sampling_kernels();
		precompile_ReSTIR_DI_kernels();
	});

	ThreadManager::detach_threads(ThreadManager::GPU_RENDERER_PRECOMPILE_KERNELS_THREAD_KEY);
}

extern bool g_background_shader_compilation_enabled;
void GPURenderer::stop_background_shader_compilation()
{
	g_background_shader_compilation_enabled = false;
	g_condition_for_compilation.notify_all();
}

void GPURenderer::resume_background_shader_compilation()
{
	g_background_shader_compilation_enabled = true;
	g_condition_for_compilation.notify_all();
}

void GPURenderer::precompile_direct_light_sampling_kernels()
{
	for (int init_target_function_vis = 0; init_target_function_vis <= 1; init_target_function_vis++)
	{
		for (int use_envmap_mis = 0; use_envmap_mis <= 1; use_envmap_mis++)
		{
			for (int envmap_sampling_strategy = ESS_NO_SAMPLING; envmap_sampling_strategy < ESS_ALIAS_TABLE; envmap_sampling_strategy++)
			{
				for (int direct_light_sampling_strategy = LSS_NO_DIRECT_LIGHT_SAMPLING; direct_light_sampling_strategy <= LSS_RESTIR_DI - 1; direct_light_sampling_strategy++)
				{
					// Starting from what the renderer is currently using to ease our life a little
					// (partials_options like USE_HWI, BVH_TRAVERSAL_STACK_SIZE, ... would have to be copied
					// manually otherwise so just copying everything here is handy)
					GPUKernelCompilerOptions partials_options;
					// Clearing the default state of the partials_options added by the constructor
					partials_options.clear();
					partials_options.set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY, direct_light_sampling_strategy);
					partials_options.set_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY, envmap_sampling_strategy);
					partials_options.set_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BSDF_MIS, use_envmap_mis);

					// Recompiling all the kernels with the new options
					/*for (const std::string& kernel_id : get_all_kernel_ids())
						precompile_kernel(kernel_id, partials_options);*/

					/*for (auto& render_pass : m_render_passes)
						render_pass.second->precompile_kernels();
					m_restir_di_render_pass.precompile_kernels(partials_options, m_hiprt_orochi_ctx, m_func_name_sets);*/

					if (direct_light_sampling_strategy == LSS_RIS_BSDF_AND_LIGHT)
					{
						// Additional compilation for RIS with the visibility in the target function
						// for the value we haven't compiled yet
						partials_options.set_macro_value(GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION, 1 - m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION));

						// Recompiling all the kernels with the new options
						/*for (const std::string& kernel_id : get_all_kernel_ids())
							precompile_kernel(kernel_id, partials_options);*/
						/*for (auto& render_pass : m_render_passes)
							render_pass.second->precompile_kernels();
						m_restir_di_render_pass.precompile_kernels(partials_options, m_hiprt_orochi_ctx, m_func_name_sets);*/
					}
				}
			}
		}
	}
}

void GPURenderer::precompile_ReSTIR_DI_kernels()
{
	for (int init_target_function_vis = 0; init_target_function_vis <= 1; init_target_function_vis++)
	{
		for (int spatial_target_function_vis = 0; spatial_target_function_vis <= 1; spatial_target_function_vis++)
		{
			for (int do_light_presampling = 0; do_light_presampling <= 1; do_light_presampling++)
			{
				for (int visibility_bias_correction = 0; visibility_bias_correction <= 1; visibility_bias_correction++)
				{
					for (int do_visibility_reuse = 0; do_visibility_reuse <= 1; do_visibility_reuse++)
					{
						for (int bias_correction_weight = RESTIR_DI_BIAS_CORRECTION_1_OVER_M; bias_correction_weight <= RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE; bias_correction_weight++)
						{
							// Starting from what the renderer is currently using to ease our life a little
							// (partials_options like USE_HWI, BVH_TRAVERSAL_STACK_SIZE, ... would have to be copied
							// manually otherwise so just copying everything here is handy)
							GPUKernelCompilerOptions partials_options;
							// Clearing the default state of the partials_options added by the constructor
							partials_options.clear();
							partials_options.set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY, init_target_function_vis);
							partials_options.set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY, spatial_target_function_vis);
							partials_options.set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE, do_visibility_reuse);
							partials_options.set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY, visibility_bias_correction);
							partials_options.set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS, bias_correction_weight);
							partials_options.set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_LIGHTS_PRESAMPLING, do_light_presampling);
							partials_options.set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY, LSS_RESTIR_DI);

							// Recompiling all the kernels with the new options
							/*for (const std::string& kernel_id : get_all_kernel_ids())
								precompile_kernel(kernel_id, partials_options);*/

							/*for (auto& render_pass : m_render_passes)
								render_pass.second->precompile_kernels();
							m_restir_di_render_pass.precompile_kernels(partials_options, m_hiprt_orochi_ctx, m_func_name_sets);*/
						}
					}
				}
			}
		}
	}
}

void GPURenderer::precompile_kernel(const std::string& id, GPUKernelCompilerOptions partial_options)
{
	GPUKernelCompilerOptions options = m_kernels[id]->get_kernel_options().deep_copy();
	partial_options.apply_onto(options);

	ThreadManager::start_thread(ThreadManager::RENDERER_PRECOMPILE_KERNELS, ThreadFunctions::precompile_kernel,
		m_kernels[id]->get_kernel_function_name(),
		m_kernels[id]->get_kernel_file_path(),
		options, m_hiprt_orochi_ctx, std::ref(m_func_name_sets));

	ThreadManager::detach_threads(ThreadManager::RENDERER_PRECOMPILE_KERNELS);
}

std::map<std::string, std::shared_ptr<GPUKernel>> GPURenderer::get_all_kernels()
{
	std::map<std::string, std::shared_ptr<GPUKernel>> kernels;

	for (auto& pair : m_kernels)
		kernels[pair.first] = pair.second;

	for (auto& name_to_kernel : m_render_graph.get_all_kernels())
		kernels[name_to_kernel.first] = name_to_kernel.second;

	return kernels;
}

std::map<std::string, std::shared_ptr<GPUKernel>> GPURenderer::get_tracing_kernels()
{
	std::map<std::string, std::shared_ptr<GPUKernel>> kernels;

	for (auto& pair : m_kernels)
		kernels[pair.first] = pair.second;

	for (auto& name_to_kernel : m_render_graph.get_tracing_kernels())
		kernels[name_to_kernel.first] = name_to_kernel.second;

	return kernels;
}

void GPURenderer::set_debug_trace_kernel(const std::string& kernel_name, GPUKernelCompilerOptions options)
{
	if (kernel_name == "")
		// Clearing the debug kernel
		m_debug_trace_kernel = GPUKernel();
	else
	{
		m_debug_trace_kernel = GPUKernel(DEVICE_KERNELS_DIRECTORY "/" + kernel_name + ".h", kernel_name);

		// Setting all the custom options
		m_debug_trace_kernel.get_kernel_options() = options;
		m_debug_trace_kernel.compile(m_hiprt_orochi_ctx);
	}
}

bool GPURenderer::is_using_debug_kernel()
{
	return m_debug_trace_kernel.has_been_compiled();
}

oroStream_t GPURenderer::get_main_stream()
{
	return m_main_stream;
}

void GPURenderer::compute_render_pass_times()
{
	// Registering the render times of all the kernels by iterating over all the kernels
	m_render_graph.compute_render_times();

	if (m_debug_trace_kernel.has_been_compiled())
		// If the debug kernel is being used... read its execution time
		// Note that we check for 'has_been_compiled()' because if the debug kernel isn't in use,
		// then the kernel (m_debug_trace_kernel) is empty, and if it's empty, then it hasn't
		// been compiled yet
		m_render_pass_times[GPURenderer::DEBUG_KERNEL_TIME_KEY] = m_debug_trace_kernel.get_last_execution_time();

	// The total frame time is the sum of every passes
	float sum = 0.0f;
	for (auto pair : m_render_pass_times)
	{
		if (pair.first == GPURenderer::ALL_RENDER_PASSES_TIME_KEY)
			continue;

		sum += pair.second;
	}

	m_render_pass_times[GPURenderer::ALL_RENDER_PASSES_TIME_KEY] = sum;
}

std::unordered_map<std::string, float>& GPURenderer::get_render_pass_times()
{
	return m_render_pass_times;
}

float GPURenderer::get_last_frame_time()
{
	return m_render_pass_times[GPURenderer::ALL_RENDER_PASSES_TIME_KEY];
}

void GPURenderer::update_perf_metrics(std::shared_ptr<PerformanceMetricsComputer> perf_metrics)
{
	compute_render_pass_times();

	// Also adding the times of the various passes
	/*for (const std::string& kernel_id : get_all_kernel_ids())
		perf_metrics->add_value(kernel_id, m_render_pass_times[kernel_id]);*/

	m_render_graph.update_perf_metrics(perf_metrics);

	perf_metrics->add_value(GPURenderer::ALL_RENDER_PASSES_TIME_KEY, m_render_pass_times[GPURenderer::ALL_RENDER_PASSES_TIME_KEY]);

	if (m_debug_trace_kernel.has_been_compiled())
		// Adding the time for the debug kernel if it is in use
		perf_metrics->add_value(GPURenderer::DEBUG_KERNEL_TIME_KEY, m_render_pass_times[GPURenderer::DEBUG_KERNEL_TIME_KEY]);
}

void GPURenderer::reset(bool reset_by_camera_movement)
{
	m_DEBUG_SUMS.memset_whole_buffer(0);
	m_DEBUG_SUM_COUNT.memset_whole_buffer(0);

	m_render_graph.reset();

	if (m_render_data.render_settings.accumulate)
	{
		// Only resetting the seed for deterministic rendering if we're accumulating.
		// If we're not accumulating, we want each frame of the render to be different
		// so we don't get into that if block and we don't reset the seed
		m_rng.m_state.seed = 42;
		m_render_data.random_number = 42;
		m_render_data.render_settings.need_to_reset = true;
	}

	reset_nee_plus_plus();

	internal_clear_m_status_buffers();

	bool moving_camera_while_not_accumulating = reset_by_camera_movement && !m_render_data.render_settings.accumulate;
	if (!moving_camera_while_not_accumulating)
		m_render_data.render_settings.need_to_reset = true;
}

Xorshift32Generator& GPURenderer::get_rng_generator()
{
	return m_rng;
}

void GPURenderer::update_render_data()
{
	if (m_render_data_buffers_invalidated)
	{
		m_render_data.GPU_BVH = m_hiprt_scene.geometry.m_geometry;

		m_render_data.render_settings.DEBUG_SUMS = reinterpret_cast<AtomicType<float>*>(m_DEBUG_SUMS.get_device_pointer());
		m_render_data.render_settings.DEBUG_SUM_COUNT = reinterpret_cast<AtomicType<unsigned long long int>*>(m_DEBUG_SUM_COUNT.get_device_pointer());

		m_render_data.buffers.triangles_indices = reinterpret_cast<int*>(m_hiprt_scene.geometry.m_mesh.triangleIndices);
		m_render_data.buffers.vertices_positions = reinterpret_cast<float3*>(m_hiprt_scene.geometry.m_mesh.vertices);
		m_render_data.buffers.has_vertex_normals = reinterpret_cast<unsigned char*>(m_hiprt_scene.has_vertex_normals.get_device_pointer());
		m_render_data.buffers.vertex_normals = reinterpret_cast<float3*>(m_hiprt_scene.vertex_normals.get_device_pointer());
		m_render_data.buffers.material_indices = reinterpret_cast<int*>(m_hiprt_scene.material_indices.get_device_pointer());
		m_render_data.buffers.materials_buffer = m_hiprt_scene.materials_buffer.get_device_SoA_struct();
		m_render_data.buffers.material_opaque = m_hiprt_scene.material_opaque.get_device_pointer();
		m_render_data.buffers.emissive_triangles_count = m_hiprt_scene.emissive_triangles_count;
		m_render_data.buffers.emissive_triangles_indices = reinterpret_cast<int*>(m_hiprt_scene.emissive_triangles_indices.get_device_pointer());

		m_render_data.bsdfs_data.sheen_ltc_parameters_texture = m_sheen_ltc_params.get_device_texture();
		m_render_data.bsdfs_data.GGX_conductor_directional_albedo = m_GGX_conductor_directional_albedo.get_device_texture();
		m_render_data.bsdfs_data.glossy_dielectric_directional_albedo = m_glossy_dielectric_directional_albedo.get_device_texture();
		m_render_data.bsdfs_data.GGX_glass_directional_albedo = m_GGX_glass_directional_albedo.get_device_texture();
		m_render_data.bsdfs_data.GGX_glass_directional_albedo_inverse = m_GGX_glass_inverse_directional_albedo.get_device_texture();
		m_render_data.bsdfs_data.GGX_thin_glass_directional_albedo = m_GGX_thin_glass_directional_albedo.get_device_texture();

		m_render_data.buffers.material_textures = reinterpret_cast<oroTextureObject_t*>(m_hiprt_scene.gpu_materials_textures.get_device_pointer());
		m_render_data.buffers.texcoords = reinterpret_cast<float2*>(m_hiprt_scene.texcoords_buffer.get_device_pointer());

		if (m_render_data.render_settings.has_access_to_adaptive_sampling_buffers())
		{
			m_render_data.aux_buffers.pixel_sample_count = m_pixels_sample_count_buffer.get_device_pointer();
			m_render_data.aux_buffers.pixel_squared_luminance = m_pixels_squared_luminance_buffer.get_device_pointer();
		}

		m_render_data.aux_buffers.pixel_active = m_pixel_active.get_device_pointer();
		m_render_data.aux_buffers.still_one_ray_active = m_status_buffers.still_one_ray_active_buffer.get_device_pointer();
		m_render_data.aux_buffers.stop_noise_threshold_converged_count = reinterpret_cast<AtomicType<unsigned int>*>(m_status_buffers.pixels_converged_count_buffer.get_device_pointer());

		m_render_data.nee_plus_plus.packed_buffers = reinterpret_cast<AtomicType<unsigned int>*>(m_nee_plus_plus.packed_buffer.get_device_pointer());
		m_render_data.nee_plus_plus.shadow_rays_actually_traced = reinterpret_cast<AtomicType<unsigned int>*>(m_nee_plus_plus.shadow_rays_actually_traced.get_device_pointer());
		m_render_data.nee_plus_plus.total_shadow_ray_queries = reinterpret_cast<AtomicType<unsigned int>*>(m_nee_plus_plus.total_shadow_ray_queries.get_device_pointer());

		m_render_graph.update_render_data();

		m_render_data_buffers_invalidated = false;
		m_render_data.render_settings.need_to_reset = true;
	}
}

void GPURenderer::set_hiprt_scene_from_scene(const Scene& scene)
{
	if (scene.triangle_indices.size() == 0)
		// Empty scene, nothing todo
		return;

	m_hiprt_scene.geometry.upload_triangle_indices(scene.triangle_indices);
	m_hiprt_scene.geometry.upload_vertices_positions(scene.vertices_positions);
	m_hiprt_scene.geometry.m_hiprt_ctx = m_hiprt_orochi_ctx->hiprt_ctx;
	rebuild_renderer_bvh(hiprtBuildFlagBitPreferHighQualityBuild, true);

	m_hiprt_scene.has_vertex_normals.resize(scene.has_vertex_normals.size());
	m_hiprt_scene.has_vertex_normals.upload_data(scene.has_vertex_normals.data());

	m_hiprt_scene.vertex_normals.resize(scene.vertex_normals.size());
	m_hiprt_scene.vertex_normals.upload_data(scene.vertex_normals.data());

	m_hiprt_scene.material_indices.resize(scene.material_indices.size());
	m_hiprt_scene.material_indices.upload_data(scene.material_indices.data());

	// Uploading the materials after the textures have been parsed because texture
	// parsing can modify the materials (emission of constant textures are stored in the
	// material directly for example) so we need to wait for the end of texture parsing
	// to upload the materials
	ThreadManager::add_dependency(ThreadManager::RENDERER_UPLOAD_MATERIALS, ThreadManager::SCENE_TEXTURES_LOADING_THREAD_KEY);
	ThreadManager::start_thread(ThreadManager::RENDERER_UPLOAD_MATERIALS, [this, &scene]() 
	{
		OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

		std::vector<DevicePackedTexturedMaterial> packed_gpu_materials(scene.materials.size());
		for (int i = 0; i < scene.materials.size(); i++)
			packed_gpu_materials[i] = scene.materials[i].pack_to_GPU();

		m_hiprt_scene.materials_buffer.resize(scene.materials.size());
		m_hiprt_scene.materials_buffer.upload_data(packed_gpu_materials);

		// Computing the opaqueness of materials i.e. whether or not they are FULLY opaque
		std::vector<unsigned char> material_opaque(scene.materials.size());
		for (int i = 0; i < scene.materials.size(); i++)
			material_opaque[i] = scene.material_has_opaque_base_color_texture[i] && scene.materials[i].alpha_opacity == 1.0f;
		m_hiprt_scene.material_opaque.resize(material_opaque.size());
		m_hiprt_scene.material_opaque.upload_data(material_opaque);
		m_hiprt_scene.material_has_opaque_base_color_texture = scene.material_has_opaque_base_color_texture;

		m_hiprt_scene.texcoords_buffer.resize(scene.texcoords.size());
		m_hiprt_scene.texcoords_buffer.upload_data(scene.texcoords.data());
	});

	ThreadManager::add_dependency(ThreadManager::RENDERER_UPLOAD_TEXTURES, ThreadManager::SCENE_TEXTURES_LOADING_THREAD_KEY);
	ThreadManager::start_thread(ThreadManager::RENDERER_UPLOAD_TEXTURES, [this, &scene]() {
		OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

		if (scene.textures.size() > 0)
		{
			std::vector<oroTextureObject_t> oro_textures(scene.textures.size());
			m_hiprt_scene.orochi_materials_textures.reserve(scene.textures.size());
			for (int i = 0; i < scene.textures.size(); i++)
			{
				if (scene.textures[i].width == 0 || scene.textures[i].height == 0)
				{
					// It can happen that for emissive textures for example, we had a texture but its color is constant.
					// As a result, we have not read the texture but rather just stored the constant emissive color in the
					// emission filed of the material so we have no texture to read here

					// The shader will never read from that texture (because the texture index of the material has been set to -1)
					// so we set it to nullptr
					oro_textures[i] = nullptr;

					continue;
				}

				// We need to keep the texture alive so they are not destroyed when returning from 
				// this function so we're adding them to a member buffer
				m_hiprt_scene.orochi_materials_textures.push_back(OrochiTexture(scene.textures[i], hipFilterModeLinear));

				oro_textures[i] = m_hiprt_scene.orochi_materials_textures.back().get_device_texture();
			}

			m_hiprt_scene.gpu_materials_textures.resize(oro_textures.size());
			m_hiprt_scene.gpu_materials_textures.upload_data(oro_textures.data());
		}
	});

	ThreadManager::add_dependency(ThreadManager::RENDERER_UPLOAD_EMISSIVE_TRIANGLES, ThreadManager::SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES);
	ThreadManager::start_thread(ThreadManager::RENDERER_UPLOAD_EMISSIVE_TRIANGLES, [this, &scene]() {
		m_hiprt_scene.emissive_triangles_count = scene.emissive_triangle_indices.size();
		if (m_hiprt_scene.emissive_triangles_count > 0)
		{
			OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

			m_hiprt_scene.emissive_triangles_indices.resize(scene.emissive_triangle_indices.size());
			m_hiprt_scene.emissive_triangles_indices.upload_data(scene.emissive_triangle_indices.data());
		}
	});
}

void GPURenderer::rebuild_renderer_bvh(hiprtBuildFlags build_flags, bool do_compaction)
{
	m_hiprt_scene.geometry.build_bvh(build_flags, do_compaction, m_main_stream);
}

void GPURenderer::set_scene(const Scene& scene)
{
	set_hiprt_scene_from_scene(scene);
	setup_nee_plus_plus_from_scene(scene);
	compute_emissives_power_area_alias_table(scene);

	m_original_materials = scene.materials;
	m_current_materials = scene.materials;
	m_parsed_scene_metadata = scene.metadata;
}

void GPURenderer::set_envmap(const Image32Bit& envmap_image, const std::string& envmap_filepath)
{
	ThreadManager::add_dependency(ThreadManager::RENDERER_SET_ENVMAP, ThreadManager::ENVMAP_LOAD_FROM_DISK_THREAD);
	ThreadManager::start_thread(ThreadManager::RENDERER_SET_ENVMAP, [this, &envmap_image, &envmap_filepath]() {
		OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

		if (envmap_image.width == 0 || envmap_image.height == 0)
		{
			if (m_render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP)
				// We were going for the envmap but it's not available so defaulting to
				// uniform lighting instead
				m_render_data.world_settings.ambient_light_type = AmbientLightType::UNIFORM;

			g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_WARNING, "Empty envmap set on the GPURenderer... Defaulting to uniform ambient light instead.");

			return;
		}

		m_envmap.init_from_image(envmap_image, envmap_filepath);
		m_envmap.recompute_sampling_data_structure(this, &envmap_image);

		m_render_data.world_settings.envmap = m_envmap.get_packed_data_pointer();
		m_render_data.world_settings.envmap_width = m_envmap.get_width();
		m_render_data.world_settings.envmap_height = m_envmap.get_height();
		// We found an envmap so let's use it
		m_render_data.world_settings.ambient_light_type = AmbientLightType::ENVMAP;

#if EnvmapSamplingStrategy == ESS_BINARY_SEARCH
		m_render_data.world_settings.envmap_cdf = m_envmap.get_cdf_device_pointer();

		m_render_data.world_settings.alias_table_probas = nullptr;
		m_render_data.world_settings.alias_table_alias = nullptr;
#elif EnvmapSamplingStrategy == ESS_ALIAS_TABLE
		m_render_data.world_settings.envmap_cdf = nullptr;

		m_envmap.get_alias_table_device_pointers(m_render_data.world_settings.envmap_alias_table.alias_table_probas, m_render_data.world_settings.envmap_alias_table.alias_table_alias);
#endif
	});
}

bool GPURenderer::has_envmap()
{
	return m_render_data.world_settings.envmap_height != 0 && m_render_data.world_settings.envmap_width != 0;
}

const std::vector<CPUMaterial>& GPURenderer::get_original_materials()
{
	return m_original_materials;
}

const std::vector<CPUMaterial>& GPURenderer::get_current_materials()
{
	return m_current_materials;
}

const std::vector<std::string>& GPURenderer::get_material_names()
{
	return m_parsed_scene_metadata.material_names;
}

void GPURenderer::update_all_materials(std::vector<CPUMaterial>& materials)
{
	m_current_materials = materials;

	std::vector<unsigned char> new_opacity(materials.size());
	std::vector<DevicePackedTexturedMaterial> packed_gpu_materials(materials.size());
	for (int i = 0; i < materials.size(); i++)
	{
		packed_gpu_materials[i] = materials[i].pack_to_GPU();

		// The material is fully opaque if its base color texture is fully opaque
		// and if the alpha opacity is fully opaque too (1.0f)
		new_opacity[i] = materials[i].alpha_opacity == 1.0f && m_hiprt_scene.material_has_opaque_base_color_texture[i];
	}

	// Because the materials have changed, reuploading the "precomputed oapcity" of the materials
	m_hiprt_scene.material_opaque.upload_data(new_opacity);
	m_hiprt_scene.materials_buffer.upload_data(packed_gpu_materials);
}

void GPURenderer::update_one_material(CPUMaterial& material, int material_index)
{
	m_current_materials[material_index] = material;

	DevicePackedTexturedMaterial packed_gpu_material = material.pack_to_GPU();
	// The material is fully opaque if its base color texture is fully opaque
	// and if the alpha opacity is fully opaque too (1.0f)
	unsigned char new_opacity = material.alpha_opacity == 1.0f && m_hiprt_scene.material_has_opaque_base_color_texture[material_index];

	// Because the materials have changed, reuploading the "precomputed oapcity" of the materials
	m_hiprt_scene.material_opaque.upload_data_partial(material_index, &new_opacity, 1);
	m_hiprt_scene.materials_buffer.upload_data_partial(material_index, &packed_gpu_material, 1);
}


const std::vector<BoundingBox>& GPURenderer::get_mesh_bounding_boxes()
{
	return m_parsed_scene_metadata.mesh_bounding_boxes;
}

const std::vector<std::string>& GPURenderer::get_mesh_names()
{
	return m_parsed_scene_metadata.mesh_names;
}

const std::vector<int>& GPURenderer::get_mesh_material_indices()
{
	return m_parsed_scene_metadata.mesh_material_indices;
}

Camera& GPURenderer::get_camera()
{
	return m_camera;
}

Camera& GPURenderer::get_previous_frame_camera()
{
	return m_previous_frame_camera;
}

CameraAnimation& GPURenderer::get_camera_animation()
{
	return m_camera_animation;
}

RendererEnvmap& GPURenderer::get_envmap()
{
	return m_envmap;
}

void GPURenderer::set_camera(const Camera& camera)
{
	m_camera = camera;
	m_camera_animation.set_camera(&m_camera);
}

void GPURenderer::resize_g_buffer_ray_volume_states()
{
	std::dynamic_pointer_cast<FillGBufferRenderPass>(m_render_graph.get_render_pass(FillGBufferRenderPass::FILL_GBUFFER_RENDER_PASS_NAME))->resize_g_buffer_ray_volume_states();
}

void GPURenderer::translate_camera_view(glm::vec3 translation)
{
	m_camera.translate(translation);
}

void GPURenderer::rotate_camera_view(glm::vec3 rotation_angles)
{
	m_camera.rotate(rotation_angles);
}

void GPURenderer::zoom_camera_view(float offset)
{
	m_camera.zoom(offset);
}

RendererAnimationState& GPURenderer::get_animation_state()
{
	return m_animation_state;
}
