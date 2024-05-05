#include "HIPRT-Orochi/OrochiEnvmap.h"

OrochiEnvmap::OrochiEnvmap(const ImageRGBA& image) : OrochiTexture(image)
{
	std::vector<float> cdf = image.compute_get_cdf();

	m_cdf.resize(width * height);
	m_cdf.upload_data(cdf.data());
}

OrochiBuffer<float>& OrochiEnvmap::get_cdf_buffer()
{
	return m_cdf;
}

float* OrochiEnvmap::get_cdf_device_pointer()
{
	return m_cdf.get_device_pointer();
}
