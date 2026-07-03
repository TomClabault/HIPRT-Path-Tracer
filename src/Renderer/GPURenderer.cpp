/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerOptions.h"
#include "Device/includes/BSDFs/LTCsData/GGXConductorLTCFitData.h"
#include "Device/includes/BSDFs/LTCsData/ZeltnerSheenLTCFitData.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"
#include "Renderer/Baker/GPUBaker.h"
#include "Renderer/Baker/GPUBakerConstants.h"
#include "Renderer/GPURenderer.h"
#include "RenderPasses/FillGBufferRenderPass.h"
#include "Threads/ThreadManager.h"
#include "UI/RenderWindow.h"

#include <condition_variable>

// List of partials_options that will be specific to each kernel. We don't want these partials_options
// to be synchronized between kernels
const std::unordered_set<std::string> GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED = {
	GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL,
	GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE,
};

const std::string GPURenderer::ALL_RENDER_PASSES_TIME_KEY	= "FullFrameTime";
const std::string GPURenderer::FULL_FRAME_TIME_WITH_CPU_KEY = "FullFrameTimeWithCPU";

GPURenderer::GPURenderer(RenderWindow* render_window, std::shared_ptr<HIPRTOrochiCtx> hiprt_oro_ctx, std::shared_ptr<ApplicationSettings> application_settings)
{
	// Creating buffers
	m_framebuffer									   = std::make_shared<OpenGLInteropBuffer<ColorRGB32F>>();
	m_denoiser_buffers.m_denoised_framebuffer		   = std::make_shared<OpenGLInteropBuffer<ColorRGB32F>>();
	m_denoiser_buffers.m_normals_AOV_interop_buffer	   = std::make_shared<OpenGLInteropBuffer<float3_t>>();
	m_denoiser_buffers.m_normals_AOV_no_interop_buffer = std::make_shared<OrochiBuffer<float3_t>>();
	m_denoiser_buffers.m_albedo_AOV_interop_buffer	   = std::make_shared<OpenGLInteropBuffer<ColorRGB32F>>();
	m_denoiser_buffers.m_albedo_AOV_no_interop_buffer  = std::make_shared<OrochiBuffer<ColorRGB32F>>();
	m_pixels_converged_sample_count_buffer			   = std::make_shared<OrochiBuffer<int>>();

	m_DEBUG_BUFFER_ULL_1.resize(1024);
	m_DEBUG_BUFFER_ULL_2.resize(1024);
	m_DEBUG_BUFFER_FLOAT.resize(1024);
	m_DEBUG_BUFFER_STRINGS.resize(1024 * HIPRTRenderSettings::DEBUG_STRING_MAX_LENGTH);

	m_hiprt_orochi_ctx = hiprt_oro_ctx;
	OROCHI_CHECK_ERROR(oroStreamCreate(&m_main_stream));

	m_power_sampling_data_structure			 = PowerSamplingDataStructure(this);
	m_light_tree_ats_sampling_data_structure = LightTreeATSSamplingDataStructure(this);
	m_light_tree_sg_sampling_data_structure	 = LightTreeSGSamplingDataStructure(this);

	m_render_thread.init(render_window, this);
	m_device_properties									= m_hiprt_orochi_ctx->device_properties;
	m_application_settings								= application_settings;
	m_render_data.render_settings.output_debug_sample_N = hippt::max(0, m_application_settings->max_sample_count - 1);

	std::shared_ptr<GPUKernelCompilerOptions> global_compiler_options = get_global_compiler_options();
	// Adding hardware acceleration by default if supported
	global_compiler_options->set_macro_value("__USE_HWI__", device_supports_hardware_acceleration() == HardwareAccelerationSupport::SUPPORTED);
	// Just "fixing" the ReGIR options to be in sync with the UI
	if (global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_REGIR &&
		(global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR) == LSS_ONE_LIGHT ||
		 global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR) == LSS_MIS_LIGHT_BSDF))
		global_compiler_options->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR, LSS_RIS_BSDF_AND_LIGHT);

	setup_brdfs_data();
	setup_filter_functions();
	m_render_thread.setup_render_graphs();

	// Buffer that keeps track of whether at least one ray is still alive or not
	m_status_buffers.still_one_ray_active_buffer.resize(1);
	m_status_buffers.still_one_ray_active_buffer.memset_whole_buffer(1);
	m_status_buffers.pixels_converged_count_buffer.resize(1);
}

GPURenderer::~GPURenderer()
{
	m_render_thread.request_exit();
}

void GPURenderer::start_render_thread()
{
	m_render_thread.start();
}

void GPURenderer::setup_brdfs_data()
{
	load_sheen_ltc_texture();
	load_ltc_textures();

	load_GGX_energy_compensation_textures();
	load_glossy_dielectric_energy_compensation_textures();
	load_GGX_glass_energy_compensation_textures();
}

void GPURenderer::load_sheen_ltc_texture()
{
	// CUDA/HIP do not handle 3 channels textures so we're padding it to 4 channels
	std::vector<float> padded_ltc(32 * 32 * 4);

	for (int y = 0; y < 32; y++)
	{
		for (int x = 0; x < 32; x++)
		{
			int padded_index	 = (y * 32 + x) * 4;
			int non_padded_index = y * 32 + x;

			padded_ltc[padded_index + 0] = zeltner_2022_sheen_ltc_fit_parameters[non_padded_index].x;
			padded_ltc[padded_index + 1] = zeltner_2022_sheen_ltc_fit_parameters[non_padded_index].y;
			padded_ltc[padded_index + 2] = zeltner_2022_sheen_ltc_fit_parameters[non_padded_index].z;
			padded_ltc[padded_index + 3] = 0.0f;
		}
	}

	Image32Bit sheen_ltc_params_image(padded_ltc.data(), 32, 32, 4);
	m_sheen_ltc_params = OrochiTexture(sheen_ltc_params_image, hipFilterModeLinear, hipAddressModeClamp);
}

void GPURenderer::load_ltc_textures()
{
	Image32Bit GGX_conductor_ltc_params(reinterpret_cast<const float*>(ggx_conductor_ltc_fit.data()), GGX_CONDUCTOR_LTC_FIT_SIZE, GGX_CONDUCTOR_LTC_FIT_SIZE,
										4);
	Image32Bit GGX_conductor_amplitude_ltc_data(reinterpret_cast<const float*>(ggx_conductor_ltc_amplitude_data.data()), GGX_CONDUCTOR_LTC_FIT_SIZE,
												GGX_CONDUCTOR_LTC_FIT_SIZE, 1);
	Image32Bit GGX_conductor_fresnel_ltc_data(reinterpret_cast<const float*>(ggx_conductor_ltc_fresnel_data.data()), GGX_CONDUCTOR_LTC_FIT_SIZE,
											  GGX_CONDUCTOR_LTC_FIT_SIZE, 1);

	m_GGX_conductor_ltc_params		   = OrochiTexture(GGX_conductor_ltc_params, hipFilterModeLinear, hipAddressModeClamp);
	m_GGX_conductor_ltc_amplitude_data = OrochiTexture(GGX_conductor_amplitude_ltc_data, hipFilterModeLinear, hipAddressModeClamp);
	m_GGX_conductor_ltc_fresnel_data   = OrochiTexture(GGX_conductor_fresnel_ltc_data, hipFilterModeLinear, hipAddressModeClamp);
}

void GPURenderer::load_GGX_energy_compensation_textures(hipTextureFilterMode filtering_mode)
{
	Image32Bit GGXEss_image =
		Image32Bit::read_image_hdr(BRDFS_DATA_DIRECTIONAL_ALBEDO_DIRECTORY "/GGX/" +
									   GPUBakerConstants::get_GGX_conductor_directional_albedo_texture_filename(m_render_data.bsdfs_data.GGX_masking_shadowing),
								   1, true);
	m_GGX_conductor_directional_albedo = OrochiTexture(GGXEss_image, filtering_mode, hipAddressModeClamp);

	m_render_data_buffers_invalidated = true;
}

void GPURenderer::load_glossy_dielectric_energy_compensation_textures(hipTextureFilterMode filtering_mode)
{
	synchronize_all_kernels();

	std::vector<Image32Bit> images(GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR);
	for (int i = 0; i < GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR; i++)
	{
		std::string filename =
			std::to_string(i) + GPUBakerConstants::get_glossy_dielectric_directional_albedo_texture_filename(m_render_data.bsdfs_data.GGX_masking_shadowing);
		std::string filepath = BRDFS_DATA_DIRECTIONAL_ALBEDO_DIRECTORY "/GlossyDielectrics/" + filename;
		images[i]			 = Image32Bit::read_image_hdr(filepath, 1, true);
	}
	m_glossy_dielectric_directional_albedo =
		OrochiTexture3D(images, filtering_mode == hipFilterModeLinear ? ORO_TR_FILTER_MODE_LINEAR : ORO_TR_FILTER_MODE_POINT, ORO_TR_ADDRESS_MODE_CLAMP);

	m_render_data_buffers_invalidated = true;
}

void GPURenderer::load_GGX_glass_energy_compensation_textures(hipTextureFilterMode filtering_mode)
{
	synchronize_all_kernels();

	std::vector<Image32Bit> images(GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR);
	for (int i = 0; i < GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR; i++)
	{
		std::string filename =
			std::to_string(i) + GPUBakerConstants::get_GGX_glass_directional_albedo_texture_filename(m_render_data.bsdfs_data.GGX_masking_shadowing);
		std::string filepath = BRDFS_DATA_DIRECTIONAL_ALBEDO_DIRECTORY "/GGX/Glass/" + filename;
		images[i]			 = Image32Bit::read_image_hdr(filepath, 1, true);
	}
	m_GGX_glass_directional_albedo =
		OrochiTexture3D(images, filtering_mode == hipFilterModeLinear ? ORO_TR_FILTER_MODE_LINEAR : ORO_TR_FILTER_MODE_POINT, ORO_TR_ADDRESS_MODE_CLAMP);

	for (int i = 0; i < GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR; i++)
	{
		std::string filename =
			std::to_string(i) + GPUBakerConstants::get_GGX_glass_directional_albedo_inv_texture_filename(m_render_data.bsdfs_data.GGX_masking_shadowing);
		std::string filepath = BRDFS_DATA_DIRECTIONAL_ALBEDO_DIRECTORY "/GGX/Glass/" + filename;
		images[i]			 = Image32Bit::read_image_hdr(filepath, 1, true);
	}
	m_GGX_glass_inverse_directional_albedo =
		OrochiTexture3D(images, filtering_mode == hipFilterModeLinear ? ORO_TR_FILTER_MODE_LINEAR : ORO_TR_FILTER_MODE_POINT, ORO_TR_ADDRESS_MODE_CLAMP);

	images.resize(GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR);
	for (int i = 0; i < GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR; i++)
	{
		std::string filename =
			std::to_string(i) + GPUBakerConstants::get_GGX_thin_glass_directional_albedo_texture_filename(m_render_data.bsdfs_data.GGX_masking_shadowing);
		std::string filepath = BRDFS_DATA_DIRECTIONAL_ALBEDO_DIRECTORY "/GGX/Glass/" + filename;
		images[i]			 = Image32Bit::read_image_hdr(filepath, 1, true);
	}
	m_GGX_thin_glass_directional_albedo =
		OrochiTexture3D(images, filtering_mode == hipFilterModeLinear ? ORO_TR_FILTER_MODE_LINEAR : ORO_TR_FILTER_MODE_POINT, ORO_TR_ADDRESS_MODE_CLAMP);

	m_render_data_buffers_invalidated = true;
}

void GPURenderer::compute_emissives_sampling_data_structure_from_scene(const Scene& scene)
{
	m_power_sampling_data_structure.compute_from_scene(scene, get_active_render_graph().get_compiler_options());
	m_light_tree_ats_sampling_data_structure.compute_from_scene(scene, get_active_render_graph().get_compiler_options());
	m_light_tree_sg_sampling_data_structure.compute_from_scene(scene, get_active_render_graph().get_compiler_options());
}

void GPURenderer::recompute_emissives_sampling_data_structure()
{
	synchronize_all_kernels();

	m_power_sampling_data_structure.recompute_if_needed_or_free(get_active_render_graph().get_compiler_options());
	m_light_tree_ats_sampling_data_structure.recompute_if_needed_or_free(get_active_render_graph().get_compiler_options());
	m_light_tree_sg_sampling_data_structure.recompute_if_needed_or_free(get_active_render_graph().get_compiler_options());

	get_NEE_plus_plus_render_pass()->reset(false);
}

LightTreeATSBuilderOptions& GPURenderer::get_light_tree_ats_build_options()
{
	return m_light_tree_ats_sampling_data_structure.get_builder_options();
}

LightTreeATSSamplingDataStructure& GPURenderer::get_light_tree_ats_sampling_data_structure()
{
	return m_light_tree_ats_sampling_data_structure;
}

LightTreeATSBuilderOptions& GPURenderer::get_light_tree_sg_build_options()
{
	return m_light_tree_sg_sampling_data_structure.get_builder_options();
}

LightTreeSGSamplingDataStructure& GPURenderer::get_light_tree_sg_sampling_data_structure()
{
	return m_light_tree_sg_sampling_data_structure;
}

bool GPURenderer::gmon_used() const
{
	return get_gmon_render_pass() && get_gmon_render_pass()->is_render_pass_used(*get_global_compiler_options());
}

std::shared_ptr<GMoNRenderPass> GPURenderer::get_gmon_render_pass()
{
	return m_render_thread.get_gmon_render_pass();
}

std::shared_ptr<GMoNRenderPass> GPURenderer::get_gmon_render_pass() const
{
	return m_render_thread.get_gmon_render_pass();
}

std::shared_ptr<SSBNPermutationRenderPass> GPURenderer::get_ssbn_permutation_render_pass()
{
	return m_render_thread.get_ssbn_permutation_render_pass();
}

std::shared_ptr<NEEPlusPlusRenderPass> GPURenderer::get_NEE_plus_plus_render_pass()
{
	return m_render_thread.get_NEE_plus_plus_render_pass();
}

std::shared_ptr<ReGIRRenderPass> GPURenderer::get_ReGIR_render_pass()
{
	return m_render_thread.get_ReGIR_render_pass();
}

std::shared_ptr<ReSTIRDIRenderPass> GPURenderer::get_ReSTIR_DI_render_pass()
{
	return m_render_thread.get_ReSTIR_DI_render_pass();
}

std::shared_ptr<ReSTIRGIRenderPass> GPURenderer::get_ReSTIR_GI_render_pass()
{
	return m_render_thread.get_ReSTIR_GI_render_pass();
}

std::shared_ptr<ReSTIRPTRenderPass> GPURenderer::get_ReSTIR_PT_render_pass()
{
	return m_render_thread.get_ReSTIR_PT_render_pass();
}

std::shared_ptr<ReSTIRPGRenderPass> GPURenderer::get_ReSTIR_PG_render_pass()
{
	return m_render_thread.get_ReSTIR_PG_render_pass();
}

NEEPlusPlusHashGridStorage& GPURenderer::get_nee_plus_plus_storage()
{
	return get_NEE_plus_plus_render_pass()->get_nee_plus_plus_storage();
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

void GPURenderer::step_animations(float delta_time)
{
	m_envmap.update(this, delta_time);
	m_camera_animation.animation_step(this, delta_time);
}

void GPURenderer::prepare_light_sampling_data_structures()
{
	m_power_sampling_data_structure.recompute_if_needed_or_free(get_active_render_graph().get_compiler_options(), true);
	m_light_tree_ats_sampling_data_structure.recompute_if_needed_or_free(get_active_render_graph().get_compiler_options(), true);
	m_light_tree_sg_sampling_data_structure.recompute_if_needed_or_free(get_active_render_graph().get_compiler_options(), true);
}

void GPURenderer::download_status_buffers()
{
	OROCHI_CHECK_ERROR(oroMemcpy(&m_status_buffers_values.one_ray_active, m_status_buffers.still_one_ray_active_buffer.get_device_pointer(),
								 sizeof(unsigned char), oroMemcpyDeviceToHost));
	OROCHI_CHECK_ERROR(oroMemcpy(&m_status_buffers_values.pixel_converged_count, m_status_buffers.pixels_converged_count_buffer.get_device_pointer(),
								 sizeof(unsigned int), oroMemcpyDeviceToHost));
}

void GPURenderer::internal_clear_m_status_buffers()
{
	m_status_buffers_values.one_ray_active		  = true;
	m_status_buffers_values.pixel_converged_count = 0;
}

bool GPURenderer::needs_global_bvh_stack_buffer()
{
	for (const auto& name_to_kernel : m_render_thread.get_render_graphs()[GPURendererThread::RENDER_GRAPH_FULL_NAME].get_tracing_kernels())
	{
		bool global_stack_buffer_needed = false;
		global_stack_buffer_needed |=
			name_to_kernel.second->get_kernel_options().get_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL) == KERNEL_OPTION_TRUE;

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
	hiprtGlobalStackBufferInput stackBufferInput{ hiprtStackTypeGlobal, hiprtStackEntryTypeInteger,
												  static_cast<uint32_t>(m_render_data.global_traversal_stack_buffer_size),
												  static_cast<uint32_t>(nbBlocksX * nbBlocksY) };

	if (m_render_data.global_traversal_stack_buffer.stackData != nullptr)
		// Freeing if the buffer already exists
		HIPRT_CHECK_ERROR(hiprtDestroyGlobalStackBuffer(m_hiprt_orochi_ctx->hiprt_ctx, m_render_data.global_traversal_stack_buffer));

	HIPRT_CHECK_ERROR(hiprtCreateGlobalStackBuffer(m_hiprt_orochi_ctx->hiprt_ctx, stackBufferInput, m_render_data.global_traversal_stack_buffer));
}

void GPURenderer::synchronize_all_kernels()
{
	if (m_main_stream == nullptr)
		return;

	OROCHI_CHECK_ERROR(oroStreamSynchronize(m_main_stream));

	m_render_thread.wait_on_render_completion();
}

bool GPURenderer::was_last_frame_low_resolution()
{
	return m_was_last_frame_low_resolution;
}

bool GPURenderer::frame_render_done()
{
	return m_render_thread.frame_render_done();
}

void GPURenderer::resize(int new_width, int new_height)
{
	// Needed so that this function can eventually be called from another thread
	OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

	m_render_resolution = make_int2(new_width, new_height);

	synchronize_all_kernels();
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

	m_render_thread.resize(new_width, new_height);
	m_pixel_active.resize(new_width * new_height);

	m_last_frame_ray_colors.resize(new_width * new_height);

	m_input_seeds.resize(new_width * new_height);
	m_updated_random_seeds.resize(new_width * new_height);

	// Recomputing the perspective projection matrix since the aspect ratio
	// may have changed
	float new_aspect = (float)new_width / new_height;
	m_camera.set_aspect(new_aspect);

	if (needs_global_bvh_stack_buffer())
		recreate_global_bvh_stack_buffer();

	m_render_data.render_settings.render_resolution = m_render_resolution;
	m_render_data.render_settings.need_to_reset		= true;
	m_render_data_buffers_invalidated				= true;
}

void GPURenderer::reload_ssbn_permutation_blue_noise_texture(unsigned int new_width, unsigned int new_height)
{
	std::shared_ptr<SSBNPermutationRenderPass> ssbn_permutation_render_pass = get_ssbn_permutation_render_pass();
	if (!ssbn_permutation_render_pass)
		return;

	ssbn_permutation_render_pass->reload_blue_noise_texture_and_retargeting_data(new_width, new_height);
}

void GPURenderer::render(float delta_time_gpu, RenderWindow* render_window)
{
	RenderGraph* active_render_graph;

	bool imgui_item_held										= reset_when_holding_imgui_items();
	bool interactivity_render_graph_when_interacting_with_imgui = imgui_item_held && reset_when_holding_imgui_items();

	if ((render_window->is_interacting() || interactivity_render_graph_when_interacting_with_imgui) && m_render_data.render_settings.accumulate)
		active_render_graph = &m_render_thread.get_render_graphs()[GPURendererThread::RENDER_GRAPH_INTERACTIVITY_NAME];
	else
		active_render_graph = &m_render_thread.get_render_graphs()[GPURendererThread::RENDER_GRAPH_FULL_NAME];

	m_render_thread.set_active_render_graph(active_render_graph);

	pre_render_update(delta_time_gpu);

	// Mapping the render buffers on the main thread so that we can use them in the render thread.
	//
	// This is done on the main thread because using OpenGL (required when mapping the buffers from OpenGL to CUDA/HIP)
	// on a non-main thread is a bit sketchy
	map_buffers_for_render();

	if (m_render_data.render_settings.sample_number == 0)
		// If this is the very first sample, launching the prepass
		// of all the render passes
		get_active_render_graph().prepass();

	HIPRTRenderData render_data_for_frame				= m_render_data;
	GPUKernelCompilerOptions compiler_options_for_frame = active_render_graph->get_compiler_options()->deep_copy();
	m_render_thread.request_frame(render_data_for_frame, compiler_options_for_frame);
}

void GPURenderer::pre_render_update(float delta_time)
{
	m_render_thread.pre_render_update(delta_time);
}

void GPURenderer::map_buffers_for_render()
{
	m_render_data.buffers.accumulated_ray_colors = m_framebuffer->map();
	if (get_gmon_render_pass())
		m_render_data.buffers.gmon_estimator.result_framebuffer = get_gmon_render_pass()->map_result_framebuffer();

	m_render_data.aux_buffers.denoiser_normals = m_denoiser_buffers.map_normals_buffer();
	m_render_data.aux_buffers.denoiser_albedo  = m_denoiser_buffers.map_albedo_buffer();
	if (m_render_data.render_settings.has_access_to_adaptive_sampling_buffers())
		m_render_data.aux_buffers.pixel_converged_sample_count = m_pixels_converged_sample_count_buffer->get_device_pointer();
}

void GPURenderer::unmap_buffers()
{
	// TODO we should only unmap buffers that need unmapping here

	m_framebuffer->unmap();
	if (get_gmon_render_pass())
		get_gmon_render_pass()->unmap_result_framebuffer();
	m_denoiser_buffers.unmap_normals_buffer();
	m_denoiser_buffers.unmap_albedo_buffer();
}

void GPURenderer::set_use_denoiser_AOVs_interop_buffers(bool use_interop)
{
	m_denoiser_buffers.set_use_interop_AOV_buffers(this, use_interop);
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GPURenderer::get_color_interop_framebuffer()
{
	if (gmon_used() && get_gmon_render_pass()->buffers_allocated())
		return get_gmon_render_pass()->get_result_framebuffer();
	else
		return m_framebuffer;
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GPURenderer::get_default_interop_framebuffer()
{
	return m_framebuffer;
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GPURenderer::get_denoised_interop_framebuffer()
{
	return m_denoiser_buffers.m_denoised_framebuffer;
}
std::shared_ptr<OpenGLInteropBuffer<float3_t>> GPURenderer::get_denoiser_normals_AOV_interop_buffer()
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

std::shared_ptr<OrochiBuffer<float3_t>> GPURenderer::get_denoiser_normals_AOV_no_interop_buffer()
{
	return m_denoiser_buffers.m_normals_AOV_no_interop_buffer;
}
std::shared_ptr<OrochiBuffer<ColorRGB32F>> GPURenderer::get_denoiser_albedo_AOV_no_interop_buffer()
{
	return m_denoiser_buffers.m_albedo_AOV_no_interop_buffer;
}

std::shared_ptr<OrochiBuffer<int>>& GPURenderer::get_pixels_converged_sample_count_buffer()
{
	return m_pixels_converged_sample_count_buffer;
}
const StatusBuffersValues& GPURenderer::get_status_buffer_values() const
{
	return m_status_buffers_values;
}

HIPRTRenderSettings& GPURenderer::get_render_settings()
{
	return m_render_data.render_settings;
}

std::shared_ptr<ApplicationSettings> GPURenderer::get_application_settings()
{
	return m_application_settings;
}

HIPRTRenderData& GPURenderer::get_render_data()
{
	return m_render_data;
}

HIPRTScene& GPURenderer::get_hiprt_scene()
{
	return m_hiprt_scene;
}

std::shared_ptr<HIPRTOrochiCtx> GPURenderer::get_hiprt_orochi_ctx()
{
	return m_hiprt_orochi_ctx;
}

void GPURenderer::invalidate_render_data_buffers()
{
	m_render_data_buffers_invalidated = true;
}

oroDeviceProp GPURenderer::get_device_properties()
{
	return m_device_properties;
}

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
	std::string deviceName	  = getDeviceName(m_ctxt, m_device);
	const uint32_t archNumber = getGcnArchNumber(m_ctxt, m_device);
	return (archNumber >= 1030 && deviceName.find("NVIDIA") == std::string::npos);
}

HardwareAccelerationSupport GPURenderer::device_supports_hardware_acceleration()
{
	bool enabled = m_hiprt_orochi_ctx->has_hardware_ray_tracing_support();
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
	return m_render_thread.get_render_graphs().at(GPURendererThread::RENDER_GRAPH_FULL_NAME).get_compiler_options();
}

const std::shared_ptr<GPUKernelCompilerOptions> GPURenderer::get_global_compiler_options() const
{
	return m_render_thread.get_render_graphs().at(GPURendererThread::RENDER_GRAPH_FULL_NAME).get_compiler_options();
}

// Variables used to give the priority to the main thread when compiling shaders
extern bool g_main_thread_compiling;
extern std::condition_variable g_condition_for_compilation;

void GPURenderer::recompile_kernels(bool use_cache)
{
	synchronize_all_kernels();

	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Recompiling kernels...");

	// Notifying all threads that may be compiling that the main thread wants to
	// compile. This will block threads other than the main thread from compiling
	// and thus give the priority to the main thread

	for (auto& [rg_name, render_graph] : m_render_thread.get_render_graphs())
		render_graph.recompile(m_hiprt_orochi_ctx, m_func_name_sets, false, use_cache);
}

std::map<std::string, std::shared_ptr<GPUKernel>> GPURenderer::get_all_kernels()
{
	std::map<std::string, std::shared_ptr<GPUKernel>> kernels;

	for (auto& name_to_kernel : get_active_render_graph().get_all_kernels())
		kernels[name_to_kernel.first] = name_to_kernel.second;

	return kernels;
}

std::map<std::string, std::shared_ptr<GPUKernel>> GPURenderer::get_tracing_kernels()
{
	std::map<std::string, std::shared_ptr<GPUKernel>> kernels;

	for (auto& name_to_kernel : get_active_render_graph().get_tracing_kernels())
		kernels[name_to_kernel.first] = name_to_kernel.second;

	return kernels;
}

std::string GPURenderer::read_debug_buffer_string(char* DEBUG_BUFFER_STRINGS, int index)
{
	std::vector<char> debug_string_CPU = OrochiBuffer<char>::download_data(DEBUG_BUFFER_STRINGS, 1024 * HIPRTRenderSettings::DEBUG_STRING_MAX_LENGTH);

	return std::string(&debug_string_CPU[index * HIPRTRenderSettings::DEBUG_STRING_MAX_LENGTH]);
}

oroStream_t GPURenderer::get_main_stream()
{
	return m_main_stream;
}

void GPURenderer::compute_render_pass_times()
{
	// Registering the render times of all the kernels by iterating over all the kernels
	get_active_render_graph().compute_render_times();

	m_render_pass_times[GPURenderer::ALL_RENDER_PASSES_TIME_KEY] = get_active_render_graph().get_full_frame_time();
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

	get_active_render_graph().update_perf_metrics(perf_metrics);

	perf_metrics->add_value(GPURenderer::ALL_RENDER_PASSES_TIME_KEY, m_render_pass_times[GPURenderer::ALL_RENDER_PASSES_TIME_KEY]);
}

void GPURenderer::reset(bool reset_by_camera_movement)
{
	m_DEBUG_BUFFER_ULL_1.memset_whole_buffer(HIPRTRenderSettings::DEBUG_DEFAULT_ULL);
	m_DEBUG_BUFFER_ULL_2.memset_whole_buffer(HIPRTRenderSettings::DEBUG_DEFAULT_ULL);
	m_DEBUG_BUFFER_FLOAT.memset_whole_buffer(HIPRTRenderSettings::DEBUG_DEFAULT_FLOAT);
	m_DEBUG_BUFFER_STRINGS.memset_whole_buffer(0);

	if (m_render_data.render_settings.accumulate)
	{
		// Only resetting the seed for deterministic rendering if we're accumulating.
		// If we're not accumulating, we want each frame of the render to be different
		// so we don't get into that if block and we don't reset the seed
		//
		// Also we want to not reset the random number seed if we're rendering an animation and the user has asked
		// for random seeds each frame (to avoid same-noise pattern each frame of the animation)

		if ((m_animation_state.randomize_seeds_each_frame && m_animation_state.do_animations) || !m_animation_state.do_animations)
		{
			m_rng.m_state.seed										 = 42;
			m_render_data.render_settings.need_to_reset_random_seeds = true;
		}
		m_render_data.render_settings.need_to_reset = true;
	}

	internal_clear_m_status_buffers();

	bool moving_camera_while_not_accumulating = reset_by_camera_movement && !m_render_data.render_settings.accumulate;
	if (!moving_camera_while_not_accumulating)
		m_render_data.render_settings.need_to_reset = true;

	m_render_thread.reset(reset_by_camera_movement);
}

bool GPURenderer::reset_when_holding_imgui_items()
{
	bool regir = get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_REGIR;
	bool light_distributions =
		get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_USE_PER_CELL_LIGHT_DISTRIBUTIONS) == KERNEL_OPTION_TRUE;
	bool regir_light_distributions = regir && light_distributions;

	return regir_light_distributions;
}

Xorshift32Generator& GPURenderer::get_rng_generator()
{
	return m_rng;
}

void GPURenderer::update_render_data()
{
	if (m_render_data_buffers_invalidated)
	{
		m_render_data.buffers.set_updated_random_seed_pointer(m_updated_random_seeds.get_device_pointer());
		m_render_data.buffers.set_input_random_seed_pointer(m_input_seeds.get_device_pointer());

		m_render_data.GPU_BVH		= m_hiprt_scene.whole_scene_BLAS.m_geometry;
		m_render_data.light_GPU_BVH = m_hiprt_scene.emissive_triangles_BLAS.m_geometry;

		m_render_data.render_settings.DEBUG_BUFFER_ULL_1   = m_DEBUG_BUFFER_ULL_1.get_atomic_device_pointer();
		m_render_data.render_settings.DEBUG_BUFFER_ULL_2   = m_DEBUG_BUFFER_ULL_2.get_atomic_device_pointer();
		m_render_data.render_settings.DEBUG_BUFFER_FLOAT   = m_DEBUG_BUFFER_FLOAT.get_atomic_device_pointer();
		m_render_data.render_settings.DEBUG_BUFFER_STRINGS = m_DEBUG_BUFFER_STRINGS.get_device_pointer();

		m_render_data.buffers.triangles_indices	 = reinterpret_cast<int*>(m_hiprt_scene.whole_scene_BLAS.m_mesh.triangleIndices);
		m_render_data.buffers.vertices_positions = reinterpret_cast<float3_t*>(m_hiprt_scene.whole_scene_BLAS.m_mesh.vertices);
		m_render_data.buffers.has_vertex_normals = m_hiprt_scene.has_vertex_normals.get_device_pointer();
		m_render_data.buffers.vertex_normals	 = m_hiprt_scene.vertex_normals.get_device_pointer();

		m_render_data.buffers.material_indices		   = m_hiprt_scene.material_indices.get_device_pointer();
		m_render_data.buffers.materials_buffer_soa	   = m_hiprt_scene.materials_buffer.get_device_SoA_struct();
		m_render_data.buffers.material_opaque		   = m_hiprt_scene.material_opaque.get_device_pointer();
		m_render_data.buffers.emissive_triangles_count = m_hiprt_scene.emissive_triangles_count;
		if (m_hiprt_scene.emissive_triangles_primitive_indices.size() > 0)
			m_render_data.buffers.emissive_triangles_primitive_indices =
				reinterpret_cast<int*>(m_hiprt_scene.emissive_triangles_primitive_indices.get_device_pointer());
		if (m_hiprt_scene.emissive_triangles_indices_and_emissive_textures.size() > 0)
			m_render_data.buffers.emissive_triangles_primitive_indices_and_emissive_textures =
				reinterpret_cast<int*>(m_hiprt_scene.emissive_triangles_indices_and_emissive_textures.get_device_pointer());
		if (m_hiprt_scene.emissive_triangles_primitive_indices.size() > 0)
		{
			m_render_data.buffers.emissive_meshes_data						 = m_hiprt_scene.emissive_meshes_data.to_device();
			m_render_data.buffers.triangles_average_emissive_luminance		 = m_hiprt_scene.triangle_average_emissive_luminance.get_device_pointer();
			m_render_data.buffers.triangles_average_emissive_power_luminance = m_hiprt_scene.triangle_average_emissive_power_luminance.get_device_pointer();
		}
		m_render_data.buffers.triangles_areas = m_hiprt_scene.triangle_areas.get_device_pointer();
		if (m_hiprt_scene.gpu_materials_textures.size() > 0)
			m_render_data.buffers.material_textures = m_hiprt_scene.gpu_materials_textures.get_device_pointer();
		if (m_hiprt_scene.texcoords_buffer.size() > 0)
			m_render_data.buffers.texcoords = reinterpret_cast<float2_t*>(m_hiprt_scene.texcoords_buffer.get_device_pointer());

		m_render_data.bsdfs_data.ltcs_data.sheen_zeltner_texture_ltc_params = m_sheen_ltc_params.get_device_texture();
		m_render_data.bsdfs_data.ltcs_data.GGX_conductor_ltc_params			= m_GGX_conductor_ltc_params.get_device_texture();
		m_render_data.bsdfs_data.ltcs_data.GGX_conductor_ltc_amplitude_data = m_GGX_conductor_ltc_amplitude_data.get_device_texture();
		m_render_data.bsdfs_data.ltcs_data.GGX_conductor_ltc_fresnel_data	= m_GGX_conductor_ltc_fresnel_data.get_device_texture();

		m_render_data.bsdfs_data.GGX_conductor_directional_albedo	  = m_GGX_conductor_directional_albedo.get_device_texture();
		m_render_data.bsdfs_data.glossy_dielectric_directional_albedo = m_glossy_dielectric_directional_albedo.get_device_texture();
		m_render_data.bsdfs_data.GGX_glass_directional_albedo		  = m_GGX_glass_directional_albedo.get_device_texture();
		m_render_data.bsdfs_data.GGX_glass_inverse_directional_albedo = m_GGX_glass_inverse_directional_albedo.get_device_texture();
		m_render_data.bsdfs_data.GGX_thin_glass_directional_albedo	  = m_GGX_thin_glass_directional_albedo.get_device_texture();

		m_render_data.buffers.last_frame_ray_colors = m_last_frame_ray_colors.get_device_pointer();
		if (m_render_data.render_settings.has_access_to_adaptive_sampling_buffers())
		{
			m_render_data.aux_buffers.pixel_sample_count	  = m_pixels_sample_count_buffer.get_device_pointer();
			m_render_data.aux_buffers.pixel_squared_luminance = m_pixels_squared_luminance_buffer.get_device_pointer();
		}

		m_render_data.aux_buffers.pixel_active				   = m_pixel_active.get_device_pointer();
		m_render_data.aux_buffers.still_one_ray_active		   = m_status_buffers.still_one_ray_active_buffer.get_device_pointer();
		m_render_data.aux_buffers.pixel_count_converged_so_far = m_status_buffers.pixels_converged_count_buffer.get_atomic_device_pointer();

		m_render_data_buffers_invalidated = false;
	}
}

void GPURenderer::set_hiprt_scene_from_scene(const Scene& scene)
{
	if (scene.triangles_vertex_indices.size() == 0)
		// Empty scene, nothing todo
		return;

	m_hiprt_scene.whole_scene_BLAS.upload_triangle_indices(scene.triangles_vertex_indices);
	m_hiprt_scene.whole_scene_BLAS.upload_vertices_positions(scene.vertices_positions);
	m_hiprt_scene.whole_scene_BLAS.m_hiprt_ctx = m_hiprt_orochi_ctx->hiprt_ctx;
	m_hiprt_scene.total_triangle_count		   = scene.triangles_vertex_indices.size() / 3;
	rebuild_bvh(m_hiprt_scene.whole_scene_BLAS, hiprtBuildFlagBitPreferHighQualityBuild, true, true);

	m_hiprt_scene.emissive_triangles_BLAS.upload_triangle_indices(scene.emissive_triangle_vertex_indices);
	m_hiprt_scene.emissive_triangles_BLAS.copy_vertices_positions_from(m_hiprt_scene.whole_scene_BLAS);
	m_hiprt_scene.emissive_triangles_BLAS.m_hiprt_ctx = m_hiprt_orochi_ctx->hiprt_ctx;
	rebuild_bvh(m_hiprt_scene.emissive_triangles_BLAS, hiprtBuildFlagBitPreferHighQualityBuild, true, true);

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
	ThreadManager::start_thread(ThreadManager::RENDERER_UPLOAD_MATERIALS,
								[this, &scene]()
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

	ThreadManager::add_dependency(ThreadManager::RENDERER_UPLOAD_TRIANGLE_AREAS, ThreadManager::SCENE_LOADING_COMPUTE_TRIANGLE_AREAS);
	ThreadManager::start_thread(ThreadManager::RENDERER_UPLOAD_TRIANGLE_AREAS,
								[this, &scene]()
								{
									OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

									m_hiprt_scene.triangle_areas.resize(scene.triangle_areas.size());
									m_hiprt_scene.triangle_areas.upload_data(scene.triangle_areas.data());
								});

	ThreadManager::add_dependency(ThreadManager::RENDERER_UPLOAD_TEXTURES, ThreadManager::SCENE_TEXTURES_LOADING_THREAD_KEY);
	ThreadManager::start_thread(ThreadManager::RENDERER_UPLOAD_TEXTURES,
								[this, &scene]()
								{
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
											m_hiprt_scene.orochi_materials_textures.push_back(OrochiTexture(scene.textures[i], hipFilterModePoint));

											oro_textures[i] = m_hiprt_scene.orochi_materials_textures.back().get_device_texture();
										}

										m_hiprt_scene.gpu_materials_textures.resize(oro_textures.size());
										m_hiprt_scene.gpu_materials_textures.upload_data(oro_textures.data());
									}
								});

	ThreadManager::add_dependency(ThreadManager::RENDERER_UPLOAD_EMISSIVE_TRIANGLES, ThreadManager::SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES);
	ThreadManager::start_thread(
		ThreadManager::RENDERER_UPLOAD_EMISSIVE_TRIANGLES,
		[this, &scene]()
		{
			m_hiprt_scene.emissive_triangles_count = scene.emissive_triangles_primitive_indices.size();
			if (m_hiprt_scene.emissive_triangles_count > 0)
			{
				OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

				m_hiprt_scene.triangle_average_emissive_luminance.resize(scene.triangles_average_emissive_luminance.size());
				m_hiprt_scene.triangle_average_emissive_luminance.upload_data(scene.triangles_average_emissive_luminance.data());

				m_hiprt_scene.triangle_average_emissive_power_luminance.resize(scene.triangles_average_emissive_power_luminance.size());
				m_hiprt_scene.triangle_average_emissive_power_luminance.upload_data(scene.triangles_average_emissive_power_luminance.data());

				m_hiprt_scene.emissive_triangles_primitive_indices.resize(scene.emissive_triangles_primitive_indices.size());
				m_hiprt_scene.emissive_triangles_primitive_indices.upload_data(scene.emissive_triangles_primitive_indices.data());

				m_hiprt_scene.emissive_triangles_indices_and_emissive_textures.resize(scene.emissive_triangles_primitive_indices_and_emissive_textures.size());
				m_hiprt_scene.emissive_triangles_indices_and_emissive_textures.upload_data(
					scene.emissive_triangles_primitive_indices_and_emissive_textures.data());
			}

			// Uploading emissive meshes
			m_hiprt_scene.emissive_meshes_data.load_from_emissive_meshes(scene);
		});
}

void GPURenderer::rebuild_bvh(HIPRTGeometry& geometry, hiprtBuildFlags build_flags, bool do_compaction, bool disable_spatial_splits_on_OOM)
{
	synchronize_all_kernels();

	geometry.build_bvh(build_flags, do_compaction, disable_spatial_splits_on_OOM, m_main_stream);
}

void GPURenderer::rebuild_whole_scene_bvh(hiprtBuildFlags build_flags, bool do_compaction, bool disable_spatial_splits_on_OOM)
{
	rebuild_bvh(m_hiprt_scene.whole_scene_BLAS, build_flags, do_compaction, disable_spatial_splits_on_OOM);
}

void GPURenderer::set_scene(const Scene& scene)
{
	set_hiprt_scene_from_scene(scene);
	// TODO multithread this call here
	compute_emissives_sampling_data_structure_from_scene(scene);

	m_original_materials	= scene.materials;
	m_current_materials		= scene.materials;
	m_parsed_scene_metadata = scene.metadata;
}

void GPURenderer::set_envmap(const Image32Bit& envmap_image, const std::string& envmap_filepath)
{
	ThreadManager::add_dependency(ThreadManager::RENDERER_SET_ENVMAP, ThreadManager::ENVMAP_LOAD_FROM_DISK_THREAD);
	ThreadManager::start_thread(ThreadManager::RENDERER_SET_ENVMAP,
								[this, &envmap_image, &envmap_filepath]()
								{
									OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

									if (envmap_image.width == 0 || envmap_image.height == 0)
									{
										if (m_render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP)
											// We were going for the envmap but it's not available so defaulting to
											// uniform lighting instead
											m_render_data.world_settings.ambient_light_type = AmbientLightType::UNIFORM;

										g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_WARNING,
																"Empty envmap set on the GPURenderer... Defaulting to uniform ambient light instead.");

										return;
									}

									m_envmap.init_from_image(envmap_image, envmap_filepath);
									m_envmap.recompute_sampling_data_structure(this, &envmap_image);

									m_render_data.world_settings.envmap						  = m_envmap.get_packed_data_pointer();
									m_render_data.world_settings.envmap_width				  = m_envmap.get_width();
									m_render_data.world_settings.envmap_height				  = m_envmap.get_height();
									m_render_data.world_settings.envmap_packed_scaling_factor = m_envmap.get_envmap_packed_scaling_factor();
									// We found an envmap so let's use it
									m_render_data.world_settings.ambient_light_type = AmbientLightType::ENVMAP;

#if EnvmapSamplingStrategy == ESS_BINARY_SEARCH
									m_render_data.world_settings.envmap_cdf = m_envmap.get_cdf_device_pointer();

									m_render_data.world_settings.alias_table_probas = nullptr;
									m_render_data.world_settings.alias_table_alias	= nullptr;
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

const std::vector<AABB>& GPURenderer::get_mesh_bounding_boxes()
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

unsigned int GPURenderer::get_emissive_mesh_count() const
{
	return m_hiprt_scene.emissive_meshes_data.get_emissive_mesh_count();
}

unsigned int GPURenderer::get_total_triangle_count() const
{
	return m_hiprt_scene.total_triangle_count;
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

SceneMetadata& GPURenderer::get_scene_metadata()
{
	return m_parsed_scene_metadata;
}

std::unordered_map<std::string, RenderGraph>& GPURenderer::get_render_graphs()
{
	return m_render_thread.get_render_graphs();
}

RenderGraph& GPURenderer::get_active_render_graph()
{
	return m_render_thread.get_active_render_graph();
}

void GPURenderer::set_camera(const Camera& camera)
{
	m_camera = camera;
	m_camera_animation.set_camera(&m_camera);
}

void GPURenderer::resize_g_buffer_ray_volume_states()
{
	for (auto& [rg_name, render_graph] : m_render_thread.get_render_graphs())
		std::dynamic_pointer_cast<FillGBufferRenderPass>(render_graph.get_render_pass(FillGBufferRenderPass::FILL_GBUFFER_RENDER_PASS_NAME))
			->resize_g_buffer_ray_volume_states();
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
