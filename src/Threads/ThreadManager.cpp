/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Threads/ThreadManager.h"

#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

std::string ThreadManager::COMPILE_KERNEL_PASS_THREAD_KEY = "CompileKernelPassesKey";
std::string ThreadManager::SCENE_TEXTURES_LOADING_THREAD_KEY = "TextureThreadsKey";
std::string ThreadManager::SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES = "DestroyAiScene";
std::string ThreadManager::ENVMAP_LOAD_THREAD_KEY = "EnvmapLoadThreadsKey";

bool ThreadManager::m_monothread = false;
std::unordered_map<std::string, std::shared_ptr<void>> ThreadManager::m_threads_states;
std::unordered_map<std::string, std::vector<std::thread>> ThreadManager::m_threads_map;
std::unordered_map<std::string, std::vector<std::string>> ThreadManager::m_dependencies;

std::unordered_map<std::string, std::pair<std::mutex, std::condition_variable>> ThreadManager::m_condition_variables;
std::unordered_map<std::string, std::unique_lock<std::mutex>> ThreadManager::m_unique_locks;

void ThreadManager::set_monothread(bool is_monothread)
{
	m_monothread = is_monothread;
}

void ThreadManager::join_threads(std::string key)
{
	auto find = m_threads_map.find(key);
	if (find != m_threads_map.end())
	{
		if (find->second.empty())
			// No threads to wait for
			return;

		for (std::thread& thread : find->second)
			if (thread.joinable())
				thread.join();
	}
	else
	{
		std::cerr << "Trying to joing threads with key \"" << key << "\" but no threads have been started with this key.";
		return;
	}

	m_threads_map[key].clear();
	m_unique_locks[key].unlock();
	m_condition_variables[key].second.notify_all();
}

void ThreadManager::add_dependency(const std::string& key, const std::string& dependency)
{
	m_dependencies[key].push_back(dependency);
}
