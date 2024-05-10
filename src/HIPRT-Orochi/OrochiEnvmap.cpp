#include "HIPRT-Orochi/OrochiEnvmap.h"

OrochiEnvmap::OrochiEnvmap(const ImageRGBA& image) : OrochiTexture(image)
{
	compute_cdf(image);
}

OrochiEnvmap::OrochiEnvmap(OrochiEnvmap&& other) : OrochiTexture(std::move(other))
{
	m_cdf = std::move(other.m_cdf);
}

void OrochiEnvmap::operator=(OrochiEnvmap&& other)
{
	OrochiTexture::operator=(std::move(other));

	m_cdf = std::move(other.m_cdf);
}

void OrochiEnvmap::init_from_image(const ImageRGBA& image)
{
	OrochiTexture::init_from_image(image);
}

void OrochiEnvmap::compute_cdf(const ImageRGBA& image)
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
	if (m_cdf.get_element_count() == 0)
		std::cerr << "Trying to get the CDF of an OrochiEnvmap whose CDF wasn't computed in the first place..." << std::endl;

	return m_cdf.get_device_pointer();
}
