/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_BAKER_H
#define GPU_BAKER_H

#include "Image/Image.h"
#include "Renderer/Baker/GGXHemisphericalAlbedoSettings.h"

class GPUBaker
{
public:
	GPUBaker(std::shared_ptr<GPURenderer> renderer);

	void bake_ggx_hemispherical_albedo(const GGXHemisphericalAlbedoSettings& bake_settings);
	bool is_ggx_hemispherical_albedo_bake_complete() const;
	Image32Bit get_bake_ggx_hemispherical_albedo_result();

private:
	std::shared_ptr<GPURenderer> m_renderer = nullptr;

	oroStream_t m_bake_stream;

	GPUKernel m_ggx_hemispherical_albedo_bake_kernel;
	OrochiBuffer<float> m_ggx_hemispherical_albedo_bake_buffer;
	// Settings used by the last GGX hemispherical albedo bake
	GGXHemisphericalAlbedoSettings m_last_ggx_hemi_albedo_bake_settings;
	// This variable is set to true after a bake is completed and is
	// set back to false after the baked data has been querried by
	// calling 'get_bake_ggx_hemispherical_albedo_result()'
	bool m_ggx_hemispherical_albedo_bake_complete = false;
};

#endif
