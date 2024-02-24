#include "envmap.h"

#include "Utils/utils.h"

EnvironmentMap::EnvironmentMap(int width, int height) : Image(width, height)
{
    compute_cdf();
}

EnvironmentMap::EnvironmentMap(const std::vector<HIPRTColor>& data, int width, int height) : Image(data, width, height)
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

EnvironmentMap EnvironmentMap::read_from_file(const std::string& filepath)
{
    int width, height;
    std::vector<HIPRTColor> data = Utils::read_image_float(filepath, width, height, true);

    return EnvironmentMap(std::move(data), width, height);
}