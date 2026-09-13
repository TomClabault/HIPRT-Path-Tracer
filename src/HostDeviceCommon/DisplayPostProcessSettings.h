/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DISPLAY_POST_PROCESS_SETTINGS_H
#define DISPLAY_POST_PROCESS_SETTINGS_H

struct DisplayPostProcessSettings
{
	int do_tonemapping = 1;
	float gamma		   = 2.2f;
	float exposure	   = 1.8f;
};

#endif // #ifndef DISPLAY_POST_PROCESS_SETTINGS_H
