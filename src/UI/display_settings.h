/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DISPLAY_SETTINGS_H
#define DISPLAY_SETTINGS_H

struct DisplaySettings
{
	bool display_normals = false;
	bool scale_by_frame_number = true;
	bool do_tonemapping = true;
	int sample_count_override = -1;
};

#endif