/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OROCHI_ENVMAP_H
#define OROCHI_ENVMAP_H

#include "HIPRT-Orochi/OrochiTexture.h"

class OrochiEnvmap : public OrochiTexture
{
public:
	OrochiEnvmap () : OrochiTexture() {}
	OrochiEnvmap(Image32Bit& image);
	OrochiEnvmap(const OrochiEnvmap& other) = delete;
	OrochiEnvmap(OrochiEnvmap&& other);

	void operator=(const OrochiEnvmap other) = delete;
	void operator=(const OrochiEnvmap& other) = delete;
	void operator=(OrochiEnvmap&& other);

	void init_from_image(const Image32Bit& image);
	void compute_cdf(Image32Bit& image);
	OrochiBuffer<float>& get_cdf_buffer();
	float* get_cdf_device_pointer();

private:
	OrochiBuffer<float> m_cdf;
};

#endif
