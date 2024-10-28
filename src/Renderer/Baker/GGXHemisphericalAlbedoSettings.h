/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GGX_HEMISPHERICAL_ALBEDO_SETTINGS_H
#define GGX_HEMISPHERICAL_ALBEDO_SETTINGS_H

struct GGXHemisphericalAlbedoSettings
{
	int texture_size = 64;
	int integration_sample_count = 65536;
};

#endif
