#ifndef ENVMAP_H
#define ENVMAP_H

#include "Image/image.h"

class EnvironmentMap : public Image
{
public:
    EnvironmentMap() {}
    EnvironmentMap(int width, int height); 
    EnvironmentMap(Image&& data, int width, int height);

    void compute_cdf();
    const std::vector<float>& cdf() const;

    static EnvironmentMap read_from_file(const std::string& filepath);

private:
    std::vector<float> m_cdf;
};

#endif
