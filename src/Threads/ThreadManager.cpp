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
std::string ThreadManager::ENVMAP_LOAD_THREAD_KEY = "EnvmapLoadThreadsKey";

bool ThreadManager::m_monothread = false;
std::unordered_map<std::string, std::shared_ptr<void>> ThreadManager::m_threads_states;
std::unordered_map<std::string, std::vector<std::thread>> ThreadManager::m_threads_map;

void ThreadManager::set_monothread(bool is_monothread)
{
	m_monothread = is_monothread;
}

void ThreadManager::join_threads(std::string key)
{
	auto find = m_threads_map.find(key);
	if (find != m_threads_map.end())
		for (std::thread& thread : find->second)
			thread.join();

	m_threads_map[key].clear();
}
