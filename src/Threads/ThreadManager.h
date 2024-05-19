/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef THREAD_MANAGER_H
#define THREAD_MANAGER_H

#include <thread>
#include <unordered_map>

/**
 * Singleton class so that threads are accessible everywhere to be .join()
 * whenever we want without having to pass them around in function calls etc...
 */
class ThreadManager
{
public:
	static std::string COMPILE_KERNEL_THREAD_KEY;
	static std::string TEXTURE_THREADS_KEY;

	static ThreadManager& instance()
	{
		static ThreadManager instance;
		return instance;
	}

	template <typename T>
	static void add_state(const std::string& key, std::shared_ptr<T> state)
	{
		threads_states[key] = std::static_pointer_cast<void>(state);
	}

	template <class _Fn, class... _Args>
	static void start_thread(std::string key, _Fn function, _Args... args)
	{
		auto find = threads_map.find(key);
		if (find == threads_map.end())
			threads_map[key] = std::vector<std::thread>();

		std::vector<std::thread>& threads_vec = threads_map.find(key)->second;
		threads_vec.push_back(std::thread(function, args...));
	}

	static void join_threads(std::string key)
	{
		auto find = threads_map.find(key);
		if (find != threads_map.end())
			for (std::thread& thread : find->second)
				thread.join();
	}

private:
	// The states are used to keep the data that the threads need alive
	static std::unordered_map<std::string, std::shared_ptr<void>> threads_states;

	// The ThreadManager can hold as many thread as we want and to find the thread
	// we want amongst all the threads stored, we use keys, hence the unordered_map
	static std::unordered_map<std::string, std::vector<std::thread>> threads_map;
};

#endif
