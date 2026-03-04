/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_BSDF_DATA_HOST_H
#define RENDERER_BSDF_DATA_HOST_H

#include "HostDeviceCommon/RenderData.h"
#include "Image/Image.h"

// template <bool forGPU>
class BSDFDataHost
{
public:
	// template <bool forGPU_>/*
	// using Image32BitType<forGPU> = std::conditional_t<forGPU_, OrochiTexture, Image32Bit>;*/

	void load_bsdf_data(HIPRTRenderData& render_data);

	void to_device(HIPRTRenderData& render_data);

	// private:
	// Image32BitType<forGPU> m_sheen_zeltner_2022_ltc_params;
	Image32Bit m_sheen_zeltner_2022_ltc_params;
	Image32Bit m_GGX_conductor_ltc_params;
	Image32Bit m_GGX_conductor_ltc_amplitude_data;
	Image32Bit m_GGX_conductor_ltc_fresnel_data;

	Image32Bit m_GGX_conductor_directional_albedo;
	Image32Bit3D m_GGX_glossy_dielectrics_directional_albedo;
	Image32Bit3D m_GGX_glass_directional_albedo;
	Image32Bit3D m_GGX_glass_inverse_directional_albedo;
	Image32Bit3D m_GGX_thin_glass_directional_albedo;
};

#endif
