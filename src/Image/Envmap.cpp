/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Image/Envmap.h"
#include "Utils/Utils.h"

EnvironmentMap::EnvironmentMap(int width, int height) : Image(width, height)
{
    compute_cdf();
}

EnvironmentMap::EnvironmentMap(Image&& image) : Image(image)
{
    compute_cdf();
}

void EnvironmentMap::compute_cdf()
{
    m_cdf.resize(height * width);
    m_cdf[0] = 0.0f;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            m_cdf[index] = m_cdf[std::max(index - 1, 0)] + luminance_of_pixel(x, y);
        }
    }
}

const std::vector<float>& EnvironmentMap::cdf() const
{
    return m_cdf;
}