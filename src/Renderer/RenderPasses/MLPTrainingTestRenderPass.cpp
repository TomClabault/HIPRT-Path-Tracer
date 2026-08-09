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
	m_kernels[MLPTrainingTestRenderPass::MLP_TRAIN]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Neural/Testing/MLPFullyFusedTrain.h");
	m_kernels[MLPTrainingTestRenderPass::MLP_TRAIN]->set_kernel_function_name("MLPFullyFusedTrain");

	m_kernels[MLPTrainingTestRenderPass::MLP_OPTIMIZE] = std::make_shared<GPUKernel>(this->get_name() + "::" + MLPTrainingTestRenderPass::MLP_OPTIMIZE);
	m_kernels[MLPTrainingTestRenderPass::MLP_OPTIMIZE]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Neural/Testing/MLPFullyFusedOptimize.h");
	m_kernels[MLPTrainingTestRenderPass::MLP_OPTIMIZE]->set_kernel_function_name("MLPFullyFusedOptimize");

	m_kernels[MLPTrainingTestRenderPass::MLP_PREDICT] = std::make_shared<GPUKernel>(this->get_name() + "::" + MLPTrainingTestRenderPass::MLP_PREDICT);
	m_kernels[MLPTrainingTestRenderPass::MLP_PREDICT]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Neural/Testing/MLPFullyFusedPredict.h");
	m_kernels[MLPTrainingTestRenderPass::MLP_PREDICT]->set_kernel_function_name("MLPFullyFusedPredict");
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

bool MLPTrainingTestRenderPass::pre_sample_update(float delta_time)
{
	if (!is_render_pass_used(*m_compiler_options))
		return false;

	if (m_mlp.maximum_size() == 0)
	{
		unsigned int batch_size = 2048;
		m_mlp.resize(batch_size);
		m_mlp.initialize(false);

		Image32Bit image_exr = Image32Bit::read_image_exr("../data/Skyspheres/envmap.exr", true);

		// Convert image to 8 bit just for testing
		m_image = Image8Bit(image_exr.width, image_exr.height, 3);
		for (unsigned int y = 0; y < image_exr.height; ++y)
		{
			for (unsigned int x = 0; x < image_exr.width; ++x)
			{
				ColorRGB32F pixel = image_exr.get_pixel_ColorRGB32F(y * image_exr.width + x);

				m_image.data()[(y * image_exr.width + x) * 3 + 0] = static_cast<unsigned char>(std::clamp(pixel.r * 255.0f, 0.0f, 255.0f));
				m_image.data()[(y * image_exr.width + x) * 3 + 1] = static_cast<unsigned char>(std::clamp(pixel.g * 255.0f, 0.0f, 255.0f));
				m_image.data()[(y * image_exr.width + x) * 3 + 2] = static_cast<unsigned char>(std::clamp(pixel.b * 255.0f, 0.0f, 255.0f));
			}
		}

		m_texture_data			= OrochiBuffer<unsigned char>(m_image.data());
		m_out_predicted_texture = OrochiBuffer<unsigned char>(m_image.width * m_image.height * 3);
	}

	return false;
}

bool MLPTrainingTestRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))
		return false;

	TrainingTestMLP mlp_device = m_mlp.to_device();
	mlp_device.training_step   = render_data.render_settings.sample_number;

	unsigned int batch_size		= 2048;
	unsigned char* texture_data = m_texture_data.get_device_pointer();
	unsigned int tex_w			= m_image.width;
	unsigned int tex_h			= m_image.height;

	// Train
	unsigned int frame_number	= render_data.render_settings.sample_number;
	fp16* train_activations_ptr = reinterpret_cast<fp16*>(m_mlp.m_mlp_data.get_buffer_data_ptr<MLPDataHostBuffers::MLP_TRAIN_ACTIVATIONS>());
	void* train_launch_args[]	= { &mlp_device, &texture_data, &tex_w, &tex_h, &frame_number, &train_activations_ptr };
	m_kernels[MLPTrainingTestRenderPass::MLP_TRAIN]->launch_asynchronous(TrainingTestMLP::BLOCK_SIZE, 1, batch_size, 1, train_launch_args,
																		 m_renderer->get_main_stream());
	oroStreamSynchronize(m_renderer->get_main_stream());

	// Optimize
	void* optimize_launch_args[] = { &mlp_device };
	m_kernels[MLPTrainingTestRenderPass::MLP_OPTIMIZE]->launch_asynchronous(1024, 1, TrainingTestMLP::CONNECTIONS_COUNT, 1, optimize_launch_args,
																			m_renderer->get_main_stream());
	oroStreamSynchronize(m_renderer->get_main_stream());

	// Predict
	unsigned char* predicted_texture_data = m_out_predicted_texture.get_device_pointer();
	unsigned int predicted_texture_width  = m_image.width;
	unsigned int predicted_texture_height = m_image.height;
	void* predict_launch_args[]			  = { &mlp_device, &predicted_texture_data, &predicted_texture_width, &predicted_texture_height };
	m_kernels[MLPTrainingTestRenderPass::MLP_PREDICT]->launch_asynchronous(TrainingTestMLP::BLOCK_SIZE, 1, m_image.width * m_image.height, 1,
																		   predict_launch_args, m_renderer->get_main_stream());
	oroStreamSynchronize(m_renderer->get_main_stream());

	// Save prediction
	if (m_renderer->get_application_settings()->max_sample_count / 4 > 0 &&
		render_data.render_settings.sample_number % (m_renderer->get_application_settings()->max_sample_count / 4) == 0)
	{
		std::vector<unsigned char> out_predicted_texture = m_out_predicted_texture.download_data();
		Image8Bit predicted_image(out_predicted_texture, m_image.width, m_image.height, 3);
		std::string out_name = "image_predicted" + std::to_string(render_data.render_settings.sample_number) + ".png";
		predicted_image.write_image_png(out_name);
	}

	// Reset gradient counter for next frame
	m_mlp.m_mlp_data.memset_buffer<MLPDataHostBuffers::MLP_LAST_TRAINING_SAMPLE_COUNT>(0);

	return true;
}

void MLPTrainingTestRenderPass::reset(bool reset_by_camera_movement)
{
	m_mlp.initialize(false);
}

bool MLPTrainingTestRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	return true;
}
