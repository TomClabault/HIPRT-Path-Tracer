/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_VMF_MIXTURE_COMPONENT_H
#define DEVICE_INCLUDES_VMF_MIXTURE_COMPONENT_H

#include "Device/includes/PathGuiding/VMF.h"

struct VMFMixtureComponent
{
	VMF vmf;
	float weight = 0.0f;
};

#endif
