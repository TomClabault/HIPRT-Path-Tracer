/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/MLPTrainingTestRenderPass.h"

const std::string MLPTrainingTestRenderPass::MLP_TRAINING_TEST_RENDER_PASS_NAME = "MLP Training Test";
const std::string MLPTrainingTestRenderPass::MLP_TRAIN							= "MLP Train";
const std::string MLPTrainingTestRenderPass::MLP_OPTIMIZE						= "MLP Optimize";
const std::string MLPTrainingTestRenderPass::MLP_PREDICT						= "MLP Predict";

MLPTrainingTestRenderPass::MLPTrainingTestRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: MLPTrainingTestRenderPass(MLPTrainingTestRenderPass::MLP_TRAINING_TEST_RENDER_PASS_NAME, renderer, options)
{
}

MLPTrainingTestRenderPass::MLPTrainingTestRenderPass(const std::string& name, GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(name, renderer, options)
{
	m_kernels[MLPTrainingTestRenderPass::MLP_TRAIN] = std::make_shared<GPUKernel>(this->get_name() + "::" + MLPTrainingTestRenderPass::MLP_TRAIN);
	m_kernels[MLPTrainingTestRenderPass::MLP_TRAIN]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Neural/Testing/MLPTrain.h");
	m_kernels[MLPTrainingTestRenderPass::MLP_TRAIN]->set_kernel_function_name("MLPTrain");

	m_kernels[MLPTrainingTestRenderPass::MLP_OPTIMIZE] = std::make_shared<GPUKernel>(this->get_name() + "::" + MLPTrainingTestRenderPass::MLP_OPTIMIZE);
	m_kernels[MLPTrainingTestRenderPass::MLP_OPTIMIZE]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Neural/Testing/MLPOptimize.h");
	m_kernels[MLPTrainingTestRenderPass::MLP_OPTIMIZE]->set_kernel_function_name("MLPOptimize");

	m_kernels[MLPTrainingTestRenderPass::MLP_PREDICT] = std::make_shared<GPUKernel>(this->get_name() + "::" + MLPTrainingTestRenderPass::MLP_PREDICT);
	m_kernels[MLPTrainingTestRenderPass::MLP_PREDICT]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Neural/Testing/MLPPredict.h");
	m_kernels[MLPTrainingTestRenderPass::MLP_PREDICT]->set_kernel_function_name("MLPPredict");
}

bool MLPTrainingTestRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
															 const std::vector<hiprtFuncNameSet>& func_name_sets,
															 bool silent,
															 bool use_cache)
{
	if (!is_render_pass_used(*m_compiler_options))
		return false;

	bool updated = false;

	if (!m_kernels[MLPTrainingTestRenderPass::MLP_TRAIN]->has_been_compiled())
	{
		updated = true;
		m_kernels[MLPTrainingTestRenderPass::MLP_TRAIN]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	if (!m_kernels[MLPTrainingTestRenderPass::MLP_OPTIMIZE]->has_been_compiled())
	{
		updated = true;
		m_kernels[MLPTrainingTestRenderPass::MLP_OPTIMIZE]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	if (!m_kernels[MLPTrainingTestRenderPass::MLP_PREDICT]->has_been_compiled())
	{
		updated = true;
		m_kernels[MLPTrainingTestRenderPass::MLP_PREDICT]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	return updated;
}

void MLPTrainingTestRenderPass::resize(unsigned int new_width, unsigned int new_height) {}

bool MLPTrainingTestRenderPass::pre_render_update(float delta_time)
{
	if (!is_render_pass_used(*m_compiler_options))
		return false;

	if (m_mlp.size() == 0)
	{
		m_mlp.resize();
		m_mlp.initialize();

		m_apple					= Image8Bit::read_image("F:\\Repos\\Dx12NN\\src\\assets\\circle.png", 3, true);
		m_texture_data			= OrochiBuffer<unsigned char>(m_apple.data());
		m_out_predicted_texture = OrochiBuffer<unsigned char>(m_apple.width * m_apple.height * 3);
	}

	return false;
}

bool MLPTrainingTestRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))
		return false;

	MLPDevice mlp_device	 = m_mlp.to_device();
	mlp_device.training_step = render_data.render_settings.sample_number;

	unsigned int batch_size		= 2048;
	unsigned char* texture_data = m_texture_data.get_device_pointer();
	unsigned int tex_w			= m_apple.width;
	unsigned int tex_h			= m_apple.height;

	// Train
	unsigned int frame_number = render_data.render_settings.sample_number;
	void* train_launch_args[] = { &mlp_device, &texture_data, &tex_w, &tex_h, &frame_number };
	m_kernels[MLPTrainingTestRenderPass::MLP_TRAIN]->launch_asynchronous(KernelBlockWidthHeight, 1, batch_size, 1, train_launch_args,
																		 m_renderer->get_main_stream());
	oroStreamSynchronize(m_renderer->get_main_stream());

	// Optimize
	void* optimize_launch_args[] = { &mlp_device };
	m_kernels[MLPTrainingTestRenderPass::MLP_OPTIMIZE]->launch_asynchronous(1024, 1, MLP_CONNECTIONS_COUNT, 1, optimize_launch_args,
																			m_renderer->get_main_stream());
	oroStreamSynchronize(m_renderer->get_main_stream());

	// Predict
	unsigned char* predicted_texture_data = m_out_predicted_texture.get_device_pointer();
	unsigned int predicted_texture_width  = m_apple.width;
	unsigned int predicted_texture_height = m_apple.height;
	void* predict_launch_args[]			  = { &mlp_device, &predicted_texture_data, &predicted_texture_width, &predicted_texture_height };
	m_kernels[MLPTrainingTestRenderPass::MLP_PREDICT]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_apple.width, m_apple.height,
																		   predict_launch_args, m_renderer->get_main_stream());
	oroStreamSynchronize(m_renderer->get_main_stream());

	// Save prediction
	std::vector<unsigned char> out_predicted_texture = m_out_predicted_texture.download_data();
	Image8Bit predicted_image(out_predicted_texture, m_apple.width, m_apple.height, 3);
	std::string out_name = "image_predicted" + std::to_string(render_data.render_settings.sample_number) + ".png";
	predicted_image.write_image_png(out_name);

	// Reset gradient counter for next frame
	m_mlp.m_mlp_data.memset_buffer<MLPDataHostBuffers::MLP_LAST_TRAINING_SAMPLE_COUNT>(0);

	return true;
}

void MLPTrainingTestRenderPass::reset(bool reset_by_camera_movement)
{
	m_mlp.initialize();
}

bool MLPTrainingTestRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	return true;
}
