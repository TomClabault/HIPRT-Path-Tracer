/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Threads/ThreadManager.h"

#include <deque>
#include <memory>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

std::string ThreadManager::COMPILE_RAY_VOLUME_STATE_SIZE_KERNEL_KEY = "CompileRayVolumeStateSizeKernelKey";
std::string ThreadManager::COMPILE_NEE_PLUS_PLUS_FINALIZE_ACCUMULATION_KERNEL_KEY = "CompileNeePlusPlusFinalizeAccumulationKernelKey";
std::string ThreadManager::COMPILE_KERNELS_THREAD_KEY = "CompileKernelPassesKey";
std::string ThreadManager::GPU_RENDERER_PRECOMPILE_KERNELS_THREAD_KEY = "GPURendererPrecompileKernelsKey";

std::string ThreadManager::RENDER_WINDOW_CONSTRUCTOR = "RenderWindowConstructor";
std::string ThreadManager::RENDER_WINDOW_RENDERER_INITIAL_RESIZE = "RenderWindowRendererInitialResize";

std::string ThreadManager::RENDERER_SET_ENVMAP = "RendererSetEnvmapKey";
std::string ThreadManager::RENDERER_BUILD_BVH = "RendererBuildBVH";
std::string ThreadManager::RENDERER_UPLOAD_MATERIALS = "RendererUploadMaterials";
std::string ThreadManager::RENDERER_UPLOAD_TEXTURES = "RendererUploadTextures";
std::string ThreadManager::RENDERER_UPLOAD_EMISSIVE_TRIANGLES = "RendererUploadEmissiveTriangles";
std::string ThreadManager::RENDERER_COMPUTE_EMISSIVES_POWER_ALIAS_TABLE = "RendererComputeEmissivesPowerAreaAliasTable";

std::string ThreadManager::RENDERER_PRECOMPILE_KERNELS = "RendererPrecompileKernel";
std::string ThreadManager::RESTIR_DI_PRECOMPILE_KERNELS = "ReSTIRDIPrecompileKernel";

std::string ThreadManager::SCENE_TEXTURES_LOADING_THREAD_KEY = "TextureThreadsKey";
std::string ThreadManager::SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES = "ParseEmissiveTrianglesKey";
std::string ThreadManager::ENVMAP_LOAD_FROM_DISK_THREAD = "EnvmapLoadThreadsKey";

bool ThreadManager::m_monothread = false;
std::unordered_map<std::string, std::shared_ptr<void>> ThreadManager::m_threads_states;
std::unordered_map<std::string, std::vector<std::thread>> ThreadManager::m_threads_map;
std::unordered_map<std::string, std::mutex> ThreadManager::m_join_mutexes;
std::unordered_map<std::string, std::unordered_set<std::string>> ThreadManager::m_dependencies;
