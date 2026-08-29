/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_BSDFS_MICROFACET_MULTIPLE_SCATTERING_CUI2023_MACROS_H
#define DEVICE_INCLUDES_BSDFS_MICROFACET_MULTIPLE_SCATTERING_CUI2023_MACROS_H

#include "HostDeviceCommon/KernelOptions/PrincipledBSDFKernelOptions.h"

// Generator code:
/*std::ofstream generated_code_file("generated_code.txt");

	for (int i = 1; i < 16; i++)
	{
		if (i == 1)
			generated_code_file << "#if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == " << i << std::endl;
		else
			generated_code_file << "#elif PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == " << i << std::endl;

		generated_code_file << std::endl;
		generated_code_file << "#define MS_CUI2023_DECLARE_G	   fp16 ";
		for (int var_index = 0; var_index < i; var_index++)
		{
			generated_code_file << "g" << std::to_string(var_index);
			if (var_index != i - 1)
				generated_code_file << ", ";
		}
		generated_code_file << ";" << std::endl;

		generated_code_file << "#define MS_CUI2023_DECLARE_LAMBDAS	   fp16 ";
		for (int var_index = 0; var_index < i; var_index++)
		{
			generated_code_file << "lambda_" << std::to_string(var_index);
			if (var_index != i - 1)
				generated_code_file << ", ";
		}
		generated_code_file << ";" << std::endl;
		generated_code_file << std::endl;

		generated_code_file << "#define MS_CUI2023_GET_G_BODY \\" << std::endl;
		generated_code_file << "switch (i) \\" << std::endl;
		generated_code_file << "{\\" << std::endl;
		for (int var_index = 0; var_index < i; var_index++)
		{
			generated_code_file << "case " << var_index << ":\\" << std::endl;
			generated_code_file << "return g" << var_index << ";\\" << std::endl;
			generated_code_file << "break;\\" << std::endl << "\\";
			generated_code_file << std::endl;
		}

		generated_code_file << "default:\\" << std::endl;
		generated_code_file << "return g0;\\" << std::endl;

		generated_code_file << "}\\" << std::endl;

		generated_code_file << std::endl;

		generated_code_file << "#define MS_CUI2023_SET_G_BODY \\" << std::endl;
		generated_code_file << "switch (i) \\" << std::endl;
		generated_code_file << "{\\" << std::endl;
		for (int var_index = 0; var_index < i; var_index++)
		{
			generated_code_file << "case " << var_index << ":\\" << std::endl;
			generated_code_file << "g" << var_index << " = value;\\" << std::endl;
			generated_code_file << "break;\\" << std::endl << "\\";
			generated_code_file << std::endl;
		}
		generated_code_file << "}\\" << std::endl;

		generated_code_file << std::endl;

		generated_code_file << "#define MS_CUI2023_GET_L_BODY \\" << std::endl;
		generated_code_file << "switch (i) \\" << std::endl;
		generated_code_file << "{\\" << std::endl;
		for (int var_index = 0; var_index < i; var_index++)
		{
			generated_code_file << "case " << var_index << ":\\" << std::endl;
			generated_code_file << "return lambda_" << var_index << ";\\" << std::endl;
			generated_code_file << "break;\\" << std::endl << "\\";
			generated_code_file << std::endl;
		}

		generated_code_file << "default:\\" << std::endl;
		generated_code_file << "return lambda_0;\\" << std::endl;

		generated_code_file << "}\\" << std::endl;

		generated_code_file << std::endl;

		generated_code_file << "#define MS_CUI2023_SET_L_BODY \\" << std::endl;
		generated_code_file << "switch (i) \\" << std::endl;
		generated_code_file << "{\\" << std::endl;
		for (int var_index = 0; var_index < i; var_index++)
		{
			generated_code_file << "case " << var_index << ":\\" << std::endl;
			generated_code_file << "lambda_" << var_index << " = value;\\" << std::endl;
			generated_code_file << "break;\\" << std::endl << "\\";
			generated_code_file << std::endl;
		}
		generated_code_file << "}\\" << std::endl;

		generated_code_file << std::endl;

		if (i == 15)
		{
			generated_code_file << "#else" << std::endl;
			generated_code_file << "#error \"Unsupported number of bounces for MicrofacetMultipleScatteringCui2023. Only 1 to 16 bounces are supported.\""
								<< std::endl;
			generated_code_file << "#endif" << std::endl;
		}
	}

	return 0;*/

/**
 * The goal of these macros is to avoid using local arrays for storing the g and lambda values for each bounce because local arrays tend to spill on the GPU.
 * Using a switch statement with a fixed number of variables I've measured was faster on my GPU
 */
#if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#define MS_CUI2023_DECLARE_G	   fp16 g0;
#define MS_CUI2023_DECLARE_LAMBDAS fp16 lambda_0;

#define MS_CUI2023_GET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return g0;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return g0;                                                                                                                                             \
	}

#define MS_CUI2023_SET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		g0 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
	}

#define MS_CUI2023_GET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return lambda_0;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return lambda_0;                                                                                                                                       \
	}

#define MS_CUI2023_SET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		lambda_0 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
	}

#elif PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 2 // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#define MS_CUI2023_DECLARE_G	   fp16 g0, g1;
#define MS_CUI2023_DECLARE_LAMBDAS fp16 lambda_0, lambda_1;

#define MS_CUI2023_GET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return g0;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return g1;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return g0;                                                                                                                                             \
	}

#define MS_CUI2023_SET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		g0 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		g1 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
	}

#define MS_CUI2023_GET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return lambda_0;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return lambda_1;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return lambda_0;                                                                                                                                       \
	}

#define MS_CUI2023_SET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		lambda_0 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		lambda_1 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
	}

#elif PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 3 // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#define MS_CUI2023_DECLARE_G	   fp16 g0, g1, g2;
#define MS_CUI2023_DECLARE_LAMBDAS fp16 lambda_0, lambda_1, lambda_2;

#define MS_CUI2023_GET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return g0;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return g1;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return g2;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return g0;                                                                                                                                             \
	}

#define MS_CUI2023_SET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		g0 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		g1 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		g2 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
	}

#define MS_CUI2023_GET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return lambda_0;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return lambda_1;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return lambda_2;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return lambda_0;                                                                                                                                       \
	}

#define MS_CUI2023_SET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		lambda_0 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		lambda_1 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		lambda_2 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
	}

#elif PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 4 // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#define MS_CUI2023_DECLARE_G	   fp16 g0, g1, g2, g3;
#define MS_CUI2023_DECLARE_LAMBDAS fp16 lambda_0, lambda_1, lambda_2, lambda_3;

#define MS_CUI2023_GET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return g0;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return g1;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return g2;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return g3;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return g0;                                                                                                                                             \
	}

#define MS_CUI2023_SET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		g0 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		g1 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		g2 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		g3 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
	}

#define MS_CUI2023_GET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return lambda_0;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return lambda_1;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return lambda_2;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return lambda_3;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return lambda_0;                                                                                                                                       \
	}

#define MS_CUI2023_SET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		lambda_0 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		lambda_1 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		lambda_2 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		lambda_3 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
	}

#elif PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 5 // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#define MS_CUI2023_DECLARE_G	   fp16 g0, g1, g2, g3, g4;
#define MS_CUI2023_DECLARE_LAMBDAS fp16 lambda_0, lambda_1, lambda_2, lambda_3, lambda_4;

#define MS_CUI2023_GET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return g0;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return g1;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return g2;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return g3;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return g4;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return g0;                                                                                                                                             \
	}

#define MS_CUI2023_SET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		g0 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		g1 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		g2 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		g3 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		g4 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
	}

#define MS_CUI2023_GET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return lambda_0;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return lambda_1;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return lambda_2;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return lambda_3;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return lambda_4;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return lambda_0;                                                                                                                                       \
	}

#define MS_CUI2023_SET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		lambda_0 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		lambda_1 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		lambda_2 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		lambda_3 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		lambda_4 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
	}

#elif PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 6 // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#define MS_CUI2023_DECLARE_G	   fp16 g0, g1, g2, g3, g4, g5;
#define MS_CUI2023_DECLARE_LAMBDAS fp16 lambda_0, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5;

#define MS_CUI2023_GET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return g0;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return g1;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return g2;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return g3;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return g4;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return g5;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return g0;                                                                                                                                             \
	}

#define MS_CUI2023_SET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		g0 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		g1 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		g2 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		g3 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		g4 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		g5 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
	}

#define MS_CUI2023_GET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return lambda_0;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return lambda_1;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return lambda_2;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return lambda_3;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return lambda_4;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return lambda_5;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return lambda_0;                                                                                                                                       \
	}

#define MS_CUI2023_SET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		lambda_0 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		lambda_1 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		lambda_2 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		lambda_3 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		lambda_4 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		lambda_5 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
	}

#elif PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 7 // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#define MS_CUI2023_DECLARE_G	   fp16 g0, g1, g2, g3, g4, g5, g6;
#define MS_CUI2023_DECLARE_LAMBDAS fp16 lambda_0, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6;

#define MS_CUI2023_GET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return g0;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return g1;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return g2;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return g3;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return g4;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return g5;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return g6;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return g0;                                                                                                                                             \
	}

#define MS_CUI2023_SET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		g0 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		g1 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		g2 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		g3 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		g4 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		g5 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		g6 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
	}

#define MS_CUI2023_GET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return lambda_0;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return lambda_1;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return lambda_2;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return lambda_3;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return lambda_4;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return lambda_5;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return lambda_6;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return lambda_0;                                                                                                                                       \
	}

#define MS_CUI2023_SET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		lambda_0 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		lambda_1 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		lambda_2 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		lambda_3 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		lambda_4 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		lambda_5 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		lambda_6 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
	}

#elif PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 8 // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#define MS_CUI2023_DECLARE_G	   fp16 g0, g1, g2, g3, g4, g5, g6, g7;
#define MS_CUI2023_DECLARE_LAMBDAS fp16 lambda_0, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7;

#define MS_CUI2023_GET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return g0;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return g1;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return g2;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return g3;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return g4;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return g5;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return g6;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return g7;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return g0;                                                                                                                                             \
	}

#define MS_CUI2023_SET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		g0 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		g1 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		g2 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		g3 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		g4 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		g5 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		g6 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		g7 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
	}

#define MS_CUI2023_GET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return lambda_0;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return lambda_1;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return lambda_2;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return lambda_3;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return lambda_4;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return lambda_5;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return lambda_6;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return lambda_7;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return lambda_0;                                                                                                                                       \
	}

#define MS_CUI2023_SET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		lambda_0 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		lambda_1 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		lambda_2 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		lambda_3 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		lambda_4 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		lambda_5 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		lambda_6 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		lambda_7 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
	}

#elif PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 9 // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#define MS_CUI2023_DECLARE_G	   fp16 g0, g1, g2, g3, g4, g5, g6, g7, g8;
#define MS_CUI2023_DECLARE_LAMBDAS fp16 lambda_0, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8;

#define MS_CUI2023_GET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return g0;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return g1;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return g2;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return g3;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return g4;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return g5;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return g6;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return g7;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		return g8;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return g0;                                                                                                                                             \
	}

#define MS_CUI2023_SET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		g0 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		g1 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		g2 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		g3 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		g4 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		g5 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		g6 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		g7 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		g8 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
	}

#define MS_CUI2023_GET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return lambda_0;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return lambda_1;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return lambda_2;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return lambda_3;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return lambda_4;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return lambda_5;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return lambda_6;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return lambda_7;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		return lambda_8;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return lambda_0;                                                                                                                                       \
	}

#define MS_CUI2023_SET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		lambda_0 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		lambda_1 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		lambda_2 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		lambda_3 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		lambda_4 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		lambda_5 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		lambda_6 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		lambda_7 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		lambda_8 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
	}

#elif PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 10 // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#define MS_CUI2023_DECLARE_G	   fp16 g0, g1, g2, g3, g4, g5, g6, g7, g8, g9;
#define MS_CUI2023_DECLARE_LAMBDAS fp16 lambda_0, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9;

#define MS_CUI2023_GET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return g0;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return g1;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return g2;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return g3;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return g4;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return g5;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return g6;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return g7;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		return g8;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		return g9;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return g0;                                                                                                                                             \
	}

#define MS_CUI2023_SET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		g0 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		g1 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		g2 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		g3 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		g4 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		g5 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		g6 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		g7 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		g8 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		g9 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
	}

#define MS_CUI2023_GET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return lambda_0;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return lambda_1;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return lambda_2;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return lambda_3;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return lambda_4;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return lambda_5;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return lambda_6;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return lambda_7;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		return lambda_8;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		return lambda_9;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return lambda_0;                                                                                                                                       \
	}

#define MS_CUI2023_SET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		lambda_0 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		lambda_1 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		lambda_2 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		lambda_3 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		lambda_4 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		lambda_5 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		lambda_6 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		lambda_7 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		lambda_8 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		lambda_9 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
	}

#elif PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 11 // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#define MS_CUI2023_DECLARE_G	   fp16 g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10;
#define MS_CUI2023_DECLARE_LAMBDAS fp16 lambda_0, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10;

#define MS_CUI2023_GET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return g0;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return g1;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return g2;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return g3;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return g4;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return g5;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return g6;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return g7;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		return g8;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		return g9;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		return g10;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return g0;                                                                                                                                             \
	}

#define MS_CUI2023_SET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		g0 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		g1 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		g2 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		g3 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		g4 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		g5 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		g6 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		g7 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		g8 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		g9 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		g10 = value;                                                                                                                                           \
		break;                                                                                                                                                 \
	}

#define MS_CUI2023_GET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return lambda_0;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return lambda_1;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return lambda_2;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return lambda_3;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return lambda_4;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return lambda_5;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return lambda_6;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return lambda_7;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		return lambda_8;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		return lambda_9;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		return lambda_10;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return lambda_0;                                                                                                                                       \
	}

#define MS_CUI2023_SET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		lambda_0 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		lambda_1 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		lambda_2 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		lambda_3 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		lambda_4 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		lambda_5 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		lambda_6 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		lambda_7 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		lambda_8 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		lambda_9 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		lambda_10 = value;                                                                                                                                     \
		break;                                                                                                                                                 \
	}

#elif PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 12 // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#define MS_CUI2023_DECLARE_G fp16 g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11;
#define MS_CUI2023_DECLARE_LAMBDAS                                                                                                                             \
	fp16 lambda_0, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, lambda_11;

#define MS_CUI2023_GET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return g0;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return g1;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return g2;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return g3;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return g4;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return g5;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return g6;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return g7;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		return g8;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		return g9;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		return g10;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		return g11;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return g0;                                                                                                                                             \
	}

#define MS_CUI2023_SET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		g0 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		g1 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		g2 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		g3 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		g4 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		g5 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		g6 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		g7 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		g8 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		g9 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		g10 = value;                                                                                                                                           \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		g11 = value;                                                                                                                                           \
		break;                                                                                                                                                 \
	}

#define MS_CUI2023_GET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return lambda_0;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return lambda_1;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return lambda_2;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return lambda_3;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return lambda_4;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return lambda_5;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return lambda_6;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return lambda_7;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		return lambda_8;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		return lambda_9;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		return lambda_10;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		return lambda_11;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return lambda_0;                                                                                                                                       \
	}

#define MS_CUI2023_SET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		lambda_0 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		lambda_1 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		lambda_2 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		lambda_3 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		lambda_4 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		lambda_5 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		lambda_6 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		lambda_7 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		lambda_8 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		lambda_9 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		lambda_10 = value;                                                                                                                                     \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		lambda_11 = value;                                                                                                                                     \
		break;                                                                                                                                                 \
	}

#elif PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 13 // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#define MS_CUI2023_DECLARE_G fp16 g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12;
#define MS_CUI2023_DECLARE_LAMBDAS                                                                                                                             \
	fp16 lambda_0, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, lambda_11, lambda_12;

#define MS_CUI2023_GET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return g0;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return g1;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return g2;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return g3;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return g4;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return g5;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return g6;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return g7;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		return g8;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		return g9;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		return g10;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		return g11;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 12:                                                                                                                                                   \
		return g12;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return g0;                                                                                                                                             \
	}

#define MS_CUI2023_SET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		g0 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		g1 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		g2 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		g3 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		g4 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		g5 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		g6 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		g7 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		g8 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		g9 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		g10 = value;                                                                                                                                           \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		g11 = value;                                                                                                                                           \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 12:                                                                                                                                                   \
		g12 = value;                                                                                                                                           \
		break;                                                                                                                                                 \
	}

#define MS_CUI2023_GET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return lambda_0;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return lambda_1;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return lambda_2;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return lambda_3;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return lambda_4;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return lambda_5;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return lambda_6;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return lambda_7;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		return lambda_8;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		return lambda_9;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		return lambda_10;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		return lambda_11;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 12:                                                                                                                                                   \
		return lambda_12;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return lambda_0;                                                                                                                                       \
	}

#define MS_CUI2023_SET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		lambda_0 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		lambda_1 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		lambda_2 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		lambda_3 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		lambda_4 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		lambda_5 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		lambda_6 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		lambda_7 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		lambda_8 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		lambda_9 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		lambda_10 = value;                                                                                                                                     \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		lambda_11 = value;                                                                                                                                     \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 12:                                                                                                                                                   \
		lambda_12 = value;                                                                                                                                     \
		break;                                                                                                                                                 \
	}

#elif PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 14 // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#define MS_CUI2023_DECLARE_G fp16 g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13;
#define MS_CUI2023_DECLARE_LAMBDAS                                                                                                                             \
	fp16 lambda_0, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, lambda_11, lambda_12, lambda_13;

#define MS_CUI2023_GET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return g0;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return g1;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return g2;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return g3;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return g4;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return g5;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return g6;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return g7;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		return g8;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		return g9;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		return g10;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		return g11;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 12:                                                                                                                                                   \
		return g12;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 13:                                                                                                                                                   \
		return g13;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return g0;                                                                                                                                             \
	}

#define MS_CUI2023_SET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		g0 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		g1 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		g2 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		g3 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		g4 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		g5 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		g6 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		g7 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		g8 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		g9 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		g10 = value;                                                                                                                                           \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		g11 = value;                                                                                                                                           \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 12:                                                                                                                                                   \
		g12 = value;                                                                                                                                           \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 13:                                                                                                                                                   \
		g13 = value;                                                                                                                                           \
		break;                                                                                                                                                 \
	}

#define MS_CUI2023_GET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return lambda_0;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return lambda_1;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return lambda_2;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return lambda_3;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return lambda_4;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return lambda_5;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return lambda_6;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return lambda_7;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		return lambda_8;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		return lambda_9;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		return lambda_10;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		return lambda_11;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 12:                                                                                                                                                   \
		return lambda_12;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 13:                                                                                                                                                   \
		return lambda_13;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return lambda_0;                                                                                                                                       \
	}

#define MS_CUI2023_SET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		lambda_0 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		lambda_1 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		lambda_2 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		lambda_3 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		lambda_4 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		lambda_5 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		lambda_6 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		lambda_7 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		lambda_8 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		lambda_9 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		lambda_10 = value;                                                                                                                                     \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		lambda_11 = value;                                                                                                                                     \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 12:                                                                                                                                                   \
		lambda_12 = value;                                                                                                                                     \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 13:                                                                                                                                                   \
		lambda_13 = value;                                                                                                                                     \
		break;                                                                                                                                                 \
	}

#elif PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 15 // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#define MS_CUI2023_DECLARE_G fp16 g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14;
#define MS_CUI2023_DECLARE_LAMBDAS                                                                                                                             \
	fp16 lambda_0, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, lambda_11, lambda_12, lambda_13,       \
		lambda_14;

#define MS_CUI2023_GET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return g0;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return g1;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return g2;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return g3;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return g4;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return g5;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return g6;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return g7;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		return g8;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		return g9;                                                                                                                                             \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		return g10;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		return g11;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 12:                                                                                                                                                   \
		return g12;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 13:                                                                                                                                                   \
		return g13;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 14:                                                                                                                                                   \
		return g14;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return g0;                                                                                                                                             \
	}

#define MS_CUI2023_SET_G_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		g0 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		g1 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		g2 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		g3 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		g4 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		g5 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		g6 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		g7 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		g8 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		g9 = value;                                                                                                                                            \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		g10 = value;                                                                                                                                           \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		g11 = value;                                                                                                                                           \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 12:                                                                                                                                                   \
		g12 = value;                                                                                                                                           \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 13:                                                                                                                                                   \
		g13 = value;                                                                                                                                           \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 14:                                                                                                                                                   \
		g14 = value;                                                                                                                                           \
		break;                                                                                                                                                 \
	}

#define MS_CUI2023_GET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		return lambda_0;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		return lambda_1;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		return lambda_2;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		return lambda_3;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		return lambda_4;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		return lambda_5;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		return lambda_6;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		return lambda_7;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		return lambda_8;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		return lambda_9;                                                                                                                                       \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		return lambda_10;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		return lambda_11;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 12:                                                                                                                                                   \
		return lambda_12;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 13:                                                                                                                                                   \
		return lambda_13;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 14:                                                                                                                                                   \
		return lambda_14;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	default:                                                                                                                                                   \
		return lambda_0;                                                                                                                                       \
	}

#define MS_CUI2023_SET_L_BODY                                                                                                                                  \
	switch (i)                                                                                                                                                 \
	{                                                                                                                                                          \
	case 0:                                                                                                                                                    \
		lambda_0 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 1:                                                                                                                                                    \
		lambda_1 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 2:                                                                                                                                                    \
		lambda_2 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 3:                                                                                                                                                    \
		lambda_3 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 4:                                                                                                                                                    \
		lambda_4 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 5:                                                                                                                                                    \
		lambda_5 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 6:                                                                                                                                                    \
		lambda_6 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 7:                                                                                                                                                    \
		lambda_7 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 8:                                                                                                                                                    \
		lambda_8 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 9:                                                                                                                                                    \
		lambda_9 = value;                                                                                                                                      \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 10:                                                                                                                                                   \
		lambda_10 = value;                                                                                                                                     \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 11:                                                                                                                                                   \
		lambda_11 = value;                                                                                                                                     \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 12:                                                                                                                                                   \
		lambda_12 = value;                                                                                                                                     \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 13:                                                                                                                                                   \
		lambda_13 = value;                                                                                                                                     \
		break;                                                                                                                                                 \
                                                                                                                                                               \
	case 14:                                                                                                                                                   \
		lambda_14 = value;                                                                                                                                     \
		break;                                                                                                                                                 \
	}

#else // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1
#error "Unsupported number of bounces for MicrofacetMultipleScatteringCui2023. Only 1 to 16 bounces are supported."
#endif // #if PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces == 1

#endif // #ifndef DEVICE_INCLUDES_BSDFS_MICROFACET_MULTIPLE_SCATTERING_CUI2023_MACROS_H
