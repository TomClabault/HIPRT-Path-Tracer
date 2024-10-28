/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/Baker/GPUBakerConstants.h"

const std::string GPUBakerConstants::GGX_ESS_FILE_NAME = "GGX_Ess_" + std::to_string(GPUBakerConstants::GGX_ESS_TEXTURE_SIZE) + "x" + std::to_string(GPUBakerConstants::GGX_ESS_TEXTURE_SIZE) + ".hdr";
const std::string GPUBakerConstants::GGX_GLASS_ESS_FILE_NAME = GPUBakerConstants::get_GGX_glass_Ess_filename(GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_COS_THETA_O, GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_ROUGHNESS, GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_IOR);
const std::string GPUBakerConstants::GGX_GLASS_INVERSE_ESS_FILE_NAME = GPUBakerConstants::get_GGX_glass_inv_Ess_filename(GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_COS_THETA_O, GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_ROUGHNESS, GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_IOR);
