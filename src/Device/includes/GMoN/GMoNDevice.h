/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_GMON_DEVICE_H
#define DEVICE_GMON_DEVICE_H

#include "HostDeviceCommon/Color.h"

/**
 * Data structure for the implementation of GMoN
 * 
 * Reference:
 * [1] [Firefly removal in Monte Carlo rendering with adaptive Median of meaNs, Buisine et al., 2021]
 */
struct GMoNDevice
{
    enum GMoNMode
    {
        MEDIAN_OF_MEANS = 0,
        BINARY_GMON = 1,
        ADAPTIVE_GMON = 2,
    };
    GMoNMode gmon_mode = GMoNMode::ADAPTIVE_GMON;

    // This is one very big buffer that contains all the sets we accumulate into for GMoN
    //
    // For example, for GMoNMSets == 5 and a render resolution of 1280x720,
    // this is going to be a buffer that is 1280*720*5 elements long
    ColorRGB32F* sets = nullptr;

    // This is the buffer that contains the G-median of means result of each pixel and this is going
    // to be displayed in the viewport instead of the regular framebuffer if GMoN is being used
    ColorRGB32F* result_framebuffer = nullptr;

    // Which is the next set that is going to receive the sample
    unsigned int next_set_to_accumulate = 0;
};

#endif
