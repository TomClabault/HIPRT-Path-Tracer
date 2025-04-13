/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OROCHI_ENVMAP_H
#define OROCHI_ENVMAP_H

#include "HIPRT-Orochi/OrochiTexture.h"

class OrochiEnvmap : public OrochiTexture
{
public:
	OrochiEnvmap() : OrochiTexture() {}
	OrochiEnvmap(Image32Bit& image);
	OrochiEnvmap(const OrochiEnvmap& other) = delete;
	OrochiEnvmap(OrochiEnvmap&& other) noexcept;

	void operator=(const OrochiEnvmap other) = delete;
	void operator=(const OrochiEnvmap& other) = delete;
	void operator=(OrochiEnvmap&& other) noexcept;

	void init_from_image(const Image32Bit& image);

	void compute_cdf(const Image32Bit& image);
	float* get_cdf_device_pointer();
	void free_cdf();

	void compute_alias_table(const Image32Bit& image);
	void get_alias_table_device_pointers(float*& probas, int*& aliases);
	void free_alias_table();

	/**
	 * Returns the sum of the luminance of all the texels of the envmap.
	 * This value is not computed by this function but is computed by compute_cdf()
	 * and compute_alias_table() so one of these two functions must be
	 * called before calling 'get_luminance_total_sum' or 'get_luminance_total_sum'
	 * will return 0.0f
	 */
	float get_luminance_total_sum() const;

private:
	float m_luminance_total_sum = 0.0f;

	OrochiBuffer<float> m_cdf;

	OrochiBuffer<float> m_alias_table_probas;
	OrochiBuffer<int> m_alias_table_alias;
};

#endif
