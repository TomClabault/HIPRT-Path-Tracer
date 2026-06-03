/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_COMMON_SPMIS_SETTINGS_H
#define DEVICE_INCLUDES_RESTIR_COMMON_SPMIS_SETTINGS_H

struct ReSTIRCommonSPMISSettings
{
	// Screen space tile size
	int tile_size = 8;

	// Fullscreen buffer that contains the hash of the pixel G-buffer data and screen tile ID
	unsigned int* pixel_hashes = nullptr;
};

#endif
