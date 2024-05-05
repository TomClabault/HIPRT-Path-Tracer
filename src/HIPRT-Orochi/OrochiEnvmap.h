#ifndef OROCHI_ENVMAP_H
#define OROCHI_ENVMAP_H

#include "HIPRT-Orochi/OrochiTexture.h"

class OrochiEnvmap : public OrochiTexture
{
public:
	OrochiEnvmap () : OrochiTexture() {}
	OrochiEnvmap (const ImageRGBA& image);

	OrochiBuffer<float>& get_cdf_buffer();
	float* get_cdf_device_pointer();

private:
	OrochiBuffer<float> m_cdf;
};

#endif
