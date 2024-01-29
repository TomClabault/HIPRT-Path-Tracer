#ifndef APPLICATION_SETTINGS_H
#define APPLICATION_SETTINGS_H

struct ApplicationSettings
{
	// How much to divide the translation distance by when the mouse
	// has been dragged over the window to move the camera
	// This is necessary because if 1 pixel of movement equalled
	// 1 world unit of translation, it would be way too fast!
	double view_translation_sldwn_x = 100.0f, view_translation_sldwn_y = 100.0f;
};

#endif