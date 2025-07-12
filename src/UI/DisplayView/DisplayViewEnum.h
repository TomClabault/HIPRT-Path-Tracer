/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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
	GMON_BLEND,
	DENOISED_BLEND,
	DISPLAY_DENOISER_NORMALS,
	DISPLAY_DENOISER_ALBEDO,
	PIXEL_CONVERGENCE_HEATMAP,
	PIXEL_CONVERGED_MAP,
	WHITE_FURNACE_THRESHOLD,
	UNDEFINED
};

#endif
