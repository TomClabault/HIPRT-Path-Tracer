/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HIPRT-Orochi/OrochiEnvmap.h"
#include "UI/ImGui/ImGuiLogger.h"

extern ImGuiLogger g_imgui_logger;

OrochiEnvmap::OrochiEnvmap(Image32Bit& image) : OrochiTexture(image) {}

OrochiEnvmap::OrochiEnvmap(OrochiEnvmap&& other) noexcept : OrochiTexture(std::move(other))
{
	m_cdf = std::move(other.m_cdf);
}

void OrochiEnvmap::operator=(OrochiEnvmap&& other) noexcept
{
	OrochiTexture::operator=(std::move(other));

	m_cdf = std::move(other.m_cdf);
}

void OrochiEnvmap::init_from_image(const Image32Bit& image)
{
	OrochiTexture::init_from_image(image);
}

void OrochiEnvmap::compute_cdf(const Image32Bit& image)
{
	std::vector<float> cdf = image.compute_cdf();
	// When computing the CDF, the total sum is actually the last element. Handy.
	m_luminance_total_sum = cdf.back();

	m_cdf.resize(width * height);
	m_cdf.upload_data(cdf.data());
}

float* OrochiEnvmap::get_cdf_device_pointer()
{
	if (m_cdf.size() == 0)
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to get the CDF of an OrochiEnvmap whose CDF wasn't computed in the first place...");

	return m_cdf.get_device_pointer();
}

void OrochiEnvmap::free_cdf()
{
	m_cdf.free();
}

void OrochiEnvmap::compute_alias_table(const Image32Bit& image)
{
	std::vector<float> probas;
	std::vector<int> alias;
	image.compute_alias_table(probas, alias, &m_luminance_total_sum);

	m_alias_table_probas.resize(width * height);
	m_alias_table_alias.resize(width * height);

	m_alias_table_probas.upload_data(probas.data());
	m_alias_table_alias.upload_data(alias.data());
}

void OrochiEnvmap::get_alias_table_device_pointers(float*& probas, int*& aliases)
{
	probas = m_alias_table_probas.get_device_pointer();
	aliases = m_alias_table_alias.get_device_pointer();
}

void OrochiEnvmap::free_alias_table()
{
	m_alias_table_probas.free();
	m_alias_table_alias.free();
}

float OrochiEnvmap::get_luminance_total_sum() const
{
	return m_luminance_total_sum;
}
