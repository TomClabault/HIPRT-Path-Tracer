/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef COMMANDLINE_ARGUMENTS_H
#define COMMANDLINE_ARGUMENTS_H

#include <iostream>

struct CommandlineArguments
{
    static const std::string DEFAULT_SCENE;
    static const std::string DEFAULT_SKYSPHERE;

    static CommandlineArguments process_command_line_args(int argc, char** argv);

    int render_width = 1280, render_height = 720;

    // Default scene and skysphere paths as expected if running the application from a build
    // directory inside the repo root folder
    std::string scene_file_path = DEFAULT_SCENE;
    std::string skysphere_file_path = "../data/Skyspheres/evening_road_01_puresky_2k.hdr";

    int render_samples = 64;
    int bounces = 8;
};

#endif
