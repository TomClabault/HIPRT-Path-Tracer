#ifndef COMMANDLINE_ARGUMENTS_H
#define COMMANDLINE_ARGUMENTS_H

#include <iostream>

struct CommandLineArguments
{
    static CommandLineArguments process_command_line_args(int argc, char** argv)
    {
        CommandLineArguments arguments;

        for (int i = 1; i < argc; i++)
        {
            std::string string_argv = std::string(argv[i]);
            if (string_argv.starts_with("--sky="))
                arguments.skysphere_file_path = string_argv.substr(6);
            else if (string_argv.starts_with("--samples="))
                arguments.render_samples = std::atoi(string_argv.substr(10).c_str());
            else if (string_argv.starts_with("--bounces="))
                arguments.bounces = std::atoi(string_argv.substr(10).c_str());
            else
                //Assuming scene file path
                arguments.scene_file_path = string_argv;
        }

        return arguments;
    }

    // Default scene and skysphere paths as expected if running the application from a build
    // directory inside the repo root folder
    std::string scene_file_path = "../data/GLTFs/cornell_pbr.gltf";
    std::string skysphere_file_path = "../data/Skyspheres/evening_road_01_puresky_2k.hdr";

    int render_samples = 64;
    int bounces = 8;
};

#endif