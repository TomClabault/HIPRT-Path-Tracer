/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef THREAD_STATE_H
#define THREAD_STATE_H

struct TextureLoadingThreadState
{
    std::vector<std::pair<aiTextureType, std::string>> texture_paths;
    std::string scene_filepath;
};

#endif
