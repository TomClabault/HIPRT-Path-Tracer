/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_HEATMAP_H
#define DEVICE_INCLUDES_HEATMAP_H

#include "HostDeviceCommon/Color.h"

#define HEATMAP_BLUE_GREEN_RED ColorRGB32F(0.0f, 0.0f, 1.0f), ColorRGB32F(0.0f, 1.0f, 0.0f), ColorRGB32F(1.0f, 0.0f, 0.0f)

#define HEATMAP_MAGMA                                                                                                                                          \
	ColorRGB32F(0.001462f, 0.000466f, 0.013866f), ColorRGB32F(0.251537f, 0.038007f, 0.406485f), ColorRGB32F(0.716387f, 0.214982f, 0.475290f),                  \
		ColorRGB32F(0.986700f, 0.535582f, 0.382210f), ColorRGB32F(0.987053f, 0.991438f, 0.749504f)

#define HEATMAP_INFERNO                                                                                                                                        \
	ColorRGB32F(0.001462f, 0.000466f, 0.013866f), ColorRGB32F(0.341500f, 0.062325f, 0.429425f), ColorRGB32F(0.735683f, 0.215906f, 0.330245f),                  \
		ColorRGB32F(0.978422f, 0.557937f, 0.034931f), ColorRGB32F(0.988362f, 0.998364f, 0.644924f)

#define HEATMAP_VIRIDIS ColorRGB32F(0.267004f, 0.004874f, 0.329415f), ColorRGB32F(0.127568f, 0.566949f, 0.550556f), ColorRGB32F(0.993248f, 0.906157f, 0.143936f)

#define HEATMAP_GRAYSCALE ColorRGB32F(0.0f, 0.0f, 0.0f), ColorRGB32F(1.0f, 1.0f, 1.0f)

template <ColorRGB32F... heatmap_colors>
HIPRT_DEVICE ColorRGB32F map_0_1_to_heatmap_color(float scalar_0_1)
{
	static_assert(sizeof...(heatmap_colors) >= 2, "A heatmap requires at least two colors");

	scalar_0_1 = hippt::clamp(0.0f, 1.0f, scalar_0_1);

	ColorRGB32F colors[]		   = { heatmap_colors... };
	unsigned int color_count	   = sizeof...(heatmap_colors);
	float color_position		   = scalar_0_1 * static_cast<float>(color_count - 1);
	unsigned int lower_color_index = static_cast<unsigned int>(color_position);
	unsigned int upper_color_index = lower_color_index + 1;
	if (upper_color_index >= color_count)
		upper_color_index = color_count - 1;

	float interpolation = color_position - static_cast<float>(lower_color_index);
	return colors[lower_color_index] * (1.0f - interpolation) + colors[upper_color_index] * interpolation;
}

#endif // DEVICE_INCLUDES_HEATMAP_H
