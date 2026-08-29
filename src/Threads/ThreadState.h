/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef THREAD_STATE_H
#define THREAD_STATE_H

// Forward declaring the aiScene. It is included in the CPP files that use this state.
struct aiScene;

struct TextureLoadingThreadState
{
	std::vector<std::pair<aiTextureType, std::string>> texture_paths;
	std::vector<int> material_indices;

	std::string scene_filepath;
	const aiScene* assimp_scene = nullptr;
};

#endif // #ifndef THREAD_STATE_H
