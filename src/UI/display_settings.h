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