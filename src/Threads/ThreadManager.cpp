/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Threads/ThreadManager.h"

#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

std::string ThreadManager::COMPILE_KERNEL_THREAD_KEY = "CompileKernelKey";
std::string ThreadManager::TEXTURE_THREADS_KEY = "TextureThreadsKey";

std::unordered_map<std::string, std::shared_ptr<void>> ThreadManager::threads_states;
std::unordered_map<std::string, std::vector<std::thread>> ThreadManager::threads_map;
