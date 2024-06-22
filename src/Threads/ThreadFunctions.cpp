/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "Threads/ThreadFunctions.h"

void ThreadFunctions::compile_kernel(std::shared_ptr<GPURenderer> renderer, std::string kernel_file, std::string kernel_function)
{
    renderer->compile_trace_kernel(kernel_file.c_str(), kernel_function.c_str());
}

void ThreadFunctions::compile_kernel_pass(hiprtContext hiprt_ctx, oroFunction* out_kernel_pass_function, std::vector<std::string> compiler_options, std::string kernel_file, std::string kernel_function)
{
    // TODO this should all be refactored in a GPUKernelFunction class or something

	std::cout << "Compiling kernel pass \"" << kernel_function << "\"..." << std::endl;

	auto start = std::chrono::high_resolution_clock::now();

	hiprtApiFunction trace_function_out;
	std::vector<const char*> options;
	std::vector<std::string> additional_includes = { KERNEL_COMPILER_ADDITIONAL_INCLUDE, DEVICE_INCLUDES_DIRECTORY, OROCHI_INCLUDES_DIRECTORY, "-I./" };

	for (const std::string& option : compiler_options)
		options.push_back(option.c_str());

	if (HIPPTOrochiUtils::build_trace_kernel(hiprt_ctx, kernel_file, kernel_function.c_str(), trace_function_out, additional_includes, options, 0, 1, false) != hiprtError::hiprtSuccess)
	{
		std::cerr << "Unable to compile kernel \"" << kernel_function << "\". Cannot continue." << std::endl;
		int ignored = std::getchar();
		std::exit(1);
	}

	std::cout << std::endl;

	*out_kernel_pass_function = *reinterpret_cast<oroFunction*>(&trace_function_out);

	int numRegs = 0;
	int numSmem = 0;
	OROCHI_CHECK_ERROR(oroFuncGetAttribute(&numRegs, ORO_FUNC_ATTRIBUTE_NUM_REGS, *out_kernel_pass_function));
	OROCHI_CHECK_ERROR(oroFuncGetAttribute(&numSmem, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, *out_kernel_pass_function));

	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << "Trace kernel: " << numRegs << " registers, shared memory " << numSmem << std::endl;
	std::cout << "Kernel \"" << kernel_function << "\" compiled in " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
}

void ThreadFunctions::load_texture(Scene& parsed_scene, std::string scene_path, const std::vector<std::pair<aiTextureType, std::string>>& tex_paths, int thread_index, int nb_threads)
{
    // Preparing the scene_filepath so that it's ready to be appended with the texture name
    std::string corrected_filepath;
    corrected_filepath = scene_path;
    corrected_filepath = corrected_filepath.substr(0, corrected_filepath.rfind('/') + 1);

    while (thread_index < parsed_scene.textures.size())
    {
        std::string full_path;
        full_path = corrected_filepath + tex_paths[thread_index].second;

        ImageRGBA texture = ImageRGBA::read_image(full_path, false);
        parsed_scene.textures_dims[thread_index] = make_int2(texture.width, texture.height);
        parsed_scene.textures[thread_index] = texture;

        thread_index += nb_threads;
    }

}
