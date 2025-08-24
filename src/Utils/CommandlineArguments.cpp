/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Utils/CommandlineArguments.h"

const std::string CommandlineArguments::DEFAULT_SCENE = DATA_DIRECTORY "/GLTFs/DELETEC.gltf";
const std::string CommandlineArguments::DEFAULT_SKYSPHERE = DATA_DIRECTORY "/Skyspheres/evening_road_01_puresky_2k.hdr";

CommandlineArguments CommandlineArguments::process_command_line_args(int argc, char** argv)
{
    CommandlineArguments arguments;

    for (int i = 1; i < argc; i++)
    {
        std::string string_argv = std::string(argv[i]);
        if (string_argv.starts_with("--sky="))
            arguments.skysphere_file_path = string_argv.substr(6);
        else if (string_argv.starts_with("--samples="))
            arguments.render_samples = std::atoi(string_argv.substr(10).c_str());
        else if (string_argv.starts_with("--bounces="))
            arguments.bounces = std::atoi(string_argv.substr(10).c_str());
        else if (string_argv.starts_with("--w="))
            arguments.render_width = std::atoi(string_argv.substr(4).c_str());
        else if (string_argv.starts_with("--width="))
            arguments.render_width = std::atoi(string_argv.substr(8).c_str());
        else if (string_argv.starts_with("--h="))
            arguments.render_height = std::atoi(string_argv.substr(4).c_str());
        else if (string_argv.starts_with("--height="))
            arguments.render_height = std::atoi(string_argv.substr(9).c_str());
        else
            //Assuming scene file path
            arguments.scene_file_path = string_argv;
    }

    return arguments;
}
