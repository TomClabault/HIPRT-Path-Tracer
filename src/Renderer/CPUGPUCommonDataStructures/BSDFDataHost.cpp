/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/BSDFs/LTCsData/GGXSpecularLambertDiffuseLTCFitData.h"
#include "Device/includes/BSDFs/LTCsData/ZeltnerSheenLTCFitData.h"

#include "Renderer/Baker/GPUBakerConstants.h"
#include "Renderer/CPUGPUCommonDataStructures/BSDFDataHost.h"

void BSDFDataHost::load_bsdf_data(HIPRTRenderData& render_data)
{
    m_sheen_zeltner_2022_ltc_params = Image32Bit(reinterpret_cast<const float*>(zeltner_2022_sheen_ltc_fit_parameters.data()), 32, 32, 3);
    m_GGX_conductor_directional_albedo = Image32Bit::read_image_hdr(BRDFS_DATA_DIRECTIONAL_ALBEDO_DIRECTORY "/GGX/" + GPUBakerConstants::get_GGX_conductor_directional_albedo_texture_filename(render_data.bsdfs_data.GGX_masking_shadowing), 1, true);

    std::vector<Image32Bit> images(GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR);
    for (int i = 0; i < GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR; i++)
    {
        std::string filename = std::to_string(i) + GPUBakerConstants::get_glossy_dielectric_directional_albedo_texture_filename(render_data.bsdfs_data.GGX_masking_shadowing);
        std::string filepath = BRDFS_DATA_DIRECTIONAL_ALBEDO_DIRECTORY "/GlossyDielectrics/" + filename;
        images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
    }
    m_GGX_glossy_dielectrics_directional_albedo = Image32Bit3D(images);

    images.resize(GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR);
    for (int i = 0; i < GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR; i++)
    {
        std::string filename = std::to_string(i) + GPUBakerConstants::get_GGX_glass_directional_albedo_texture_filename(render_data.bsdfs_data.GGX_masking_shadowing);
        std::string filepath = BRDFS_DATA_DIRECTIONAL_ALBEDO_DIRECTORY "/GGX/Glass/" + filename;
        images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
    }
    m_GGX_glass_directional_albedo = Image32Bit3D(images);

    for (int i = 0; i < GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR; i++)
    {
        std::string filename = std::to_string(i) + GPUBakerConstants::get_GGX_glass_directional_albedo_inv_texture_filename(render_data.bsdfs_data.GGX_masking_shadowing);
        std::string filepath = BRDFS_DATA_DIRECTIONAL_ALBEDO_DIRECTORY "/GGX/Glass/" + filename;
        images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
    }
    m_GGX_glass_inverse_directional_albedo = Image32Bit3D(images);

    images.resize(GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR);
    for (int i = 0; i < GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR; i++)
    {
        std::string filename = std::to_string(i) + GPUBakerConstants::get_GGX_thin_glass_directional_albedo_texture_filename(render_data.bsdfs_data.GGX_masking_shadowing);
        std::string filepath = BRDFS_DATA_DIRECTIONAL_ALBEDO_DIRECTORY "/GGX/Glass/" + filename;
        images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
    }
    m_GGX_thin_glass_directional_albedo = Image32Bit3D(images);

    m_GGX_specular_lambert_diffuse_ltc_params = Image32Bit(reinterpret_cast<const float*>(ggx_specular_lambert_diffuse_ltc_fit_parameters.data()), GGX_SPECULAR_LAMBERT_DIFFUSE_LTC_FIT_SIZE, GGX_SPECULAR_LAMBERT_DIFFUSE_LTC_FIT_SIZE, 4);
    m_GGX_specular_lambert_diffuse_inverse_ltc_params = Image32Bit(reinterpret_cast<const float*>(ggx_specular_lambert_diffuse_ltc_inverse_fit_parameters.data()), GGX_SPECULAR_LAMBERT_DIFFUSE_LTC_FIT_SIZE, GGX_SPECULAR_LAMBERT_DIFFUSE_LTC_FIT_SIZE, 4);
}

void BSDFDataHost::to_device(HIPRTRenderData& render_data)
{
    render_data.bsdfs_data.ltcs_data.sheen_zeltner_texture_ltc_params = &m_sheen_zeltner_2022_ltc_params;
    render_data.bsdfs_data.ltcs_data.GGX_specular_lambert_diffuse_ltc_params = &m_GGX_specular_lambert_diffuse_ltc_params;
    render_data.bsdfs_data.ltcs_data.GGX_specular_lambert_diffuse_inverse_ltc_params = &m_GGX_specular_lambert_diffuse_inverse_ltc_params;

    render_data.bsdfs_data.GGX_conductor_directional_albedo = &m_GGX_conductor_directional_albedo;
    render_data.bsdfs_data.glossy_dielectric_directional_albedo = &m_GGX_glossy_dielectrics_directional_albedo;
    render_data.bsdfs_data.GGX_glass_directional_albedo = &m_GGX_glass_directional_albedo;
    render_data.bsdfs_data.GGX_glass_inverse_directional_albedo = &m_GGX_glass_inverse_directional_albedo;
    render_data.bsdfs_data.GGX_thin_glass_directional_albedo = &m_GGX_thin_glass_directional_albedo;
}
