/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DISPLAY_VIEW_ENUM_H
#define DISPLAY_VIEW_ENUM_H

/*
 * Enum used to 'switch' between what to display in the viewport
 */
enum DisplayViewType
{
	DEFAULT,
	DENOISED_BLEND,
	DISPLAY_NORMALS,
	DISPLAY_DENOISED_NORMALS,
	DISPLAY_ALBEDO,
	DISPLAY_DENOISED_ALBEDO,
	ADAPTIVE_SAMPLING_MAP,
	ADAPTIVE_SAMPLING_ACTIVE_PIXELS,
	UNDEFINED
};

#endif
