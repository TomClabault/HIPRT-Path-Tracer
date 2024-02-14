#ifndef RENDER_SETTINGS_H
#define RENDER_SETTINGS_H

struct RenderSettings
{
	int frame_number = 0;

	int samples_per_frame = 1;
	int nb_bounces = 8;
};

#endif