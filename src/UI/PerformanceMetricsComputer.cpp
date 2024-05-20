/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "PerformanceMetricsComputer.h"

std::string PerformanceMetricsComputer::SAMPLE_TIME_KEY = "SampleTimeKey";

float PerformanceMetricsComputer::data_getter(void* data, int index)
{
	return static_cast<double*>(data)[index];
}

void PerformanceMetricsComputer::init_key(const std::string& key)
{
	if (m_key_init.find(key) == m_key_init.end())
	{
		// Key data not init yet
		m_key_init[key] = true;

		m_values[key] = std::vector<double>(m_window_size, 0.0f);
		m_values_count[key] = 0;
		m_values_sum[key] = 0.0;
		m_values_sum_of_squares[key] = 0.0;
		m_data_indices[key] = 0;
		m_min_max_data[key] = std::multiset<double>();
	}
}

std::vector<double>& PerformanceMetricsComputer::get_data(const std::string& key)
{
	init_key(key);

	return m_values.find(key)->second;
}

int PerformanceMetricsComputer::get_value_count(const std::string& key)
{
	return m_values_count[key];
}

int PerformanceMetricsComputer::get_data_index(const std::string& key)
{
	return m_data_indices[key];
}

void PerformanceMetricsComputer::add_value(const std::string& key, double new_value)
{
	init_key(key);

	// Where are we going to insert the next value in the m_values vector
	unsigned int next_index = m_data_indices[key];
	m_data_indices[key] = m_data_indices[key] + 1;
	if (m_data_indices[key] == m_window_size)
		m_data_indices[key] = 0;

	double removed_value = m_values[key].at(next_index);
	m_values[key].at(next_index) = new_value;

	// Whether or not we've reached the maximum number of values we can
	// store and we are now removing a value every single time we want to insert one
	bool at_capacity = ++m_values_count[key] > m_window_size;
	m_values_count[key] -= at_capacity;

	m_values_sum[key] -= removed_value;
	m_values_sum[key] += new_value;
	m_values_sum_of_squares[key] -= removed_value * removed_value;
	m_values_sum_of_squares[key] += new_value * new_value;

	auto& multiset = m_min_max_data.find(key)->second;
	if (at_capacity)
	{
		auto position = multiset.find(removed_value);
		multiset.erase(position);
	}
	multiset.insert(new_value);
}

double PerformanceMetricsComputer::get_average(const std::string& key)
{
	return m_values_sum[key] / m_values_count[key];
}

double PerformanceMetricsComputer::get_variance(const std::string& key)
{
	double average = get_average(key);
	return m_values_sum_of_squares[key] / m_values_count[key] - average * average;
}

double PerformanceMetricsComputer::get_standard_deviation(const std::string& key)
{
	return std::sqrt(get_variance(key));
}

double PerformanceMetricsComputer::get_min(const std::string& key)
{
	return *m_min_max_data[key].begin();
}

double PerformanceMetricsComputer::get_max(const std::string& key)
{
	// rbegin() is the last element
	// end() would be past the last element so we're not using end() here
	return *m_min_max_data[key].rbegin();
}

int PerformanceMetricsComputer::get_window_size() const
{
	return m_window_size;
}

int& PerformanceMetricsComputer::get_window_size()
{
	return m_window_size;
}

void PerformanceMetricsComputer::resize_window(int new_size)
{
	if (m_window_size == new_size)
		return;

	resize_values_vectors(new_size);
	recompute_data(new_size);

	m_window_size = new_size;
}

void PerformanceMetricsComputer::resize_values_vectors(int new_size)
{
	for (auto& pair_kv : m_values)
		// Resizing the values vectors.
		// If resizing to a smaller size, .resize() throws away the elements at the end.
		// If resizing to a greater size, .resize() inserts new element at the end.
		// This is what we want.
		pair_kv.second.resize(new_size, 0.0);
}

void PerformanceMetricsComputer::recompute_data(int new_size)
{
	// This function could be recomputing elements in a smarter way but because recompute_data
	// isn't expected to be called that often at all, let's keep it simple

	if (new_size > m_window_size)
	{
		// Nothing to recompute, new elements will be added later

		// If we resized from 25 to 100 for example, we want to insert at [25] to use the
		// space for new values we just got

		for (auto pair_kv : m_values)
			m_data_indices[pair_kv.first] = m_window_size;

		return;
	}

	// Else, elements were removed from the end, we need to reocmpute the sum,
	// sums of squares, ... without taking these elements into account

	for (auto pair_kv : m_values)
	{
		const std::string& key = pair_kv.first;

		if (m_values_count[key] < new_size)
		{
			// There wasn't enough values so resizing the vector didn't remove any value,
			// nothing to recompute

			continue;
		}

		m_values_count[key] = std::min(m_values_count[key], new_size);
		m_values_sum[key] = 0.0;
		m_values_sum_of_squares[key] = 0.0;
		m_data_indices[key] = m_data_indices[key] > new_size ? 0 : m_data_indices[key];
		m_min_max_data[key].clear();

		std::vector<double>& values = pair_kv.second;
		for (int i = 0; i < new_size; i++)
		{
			double value = values[i];

			m_min_max_data[key].insert(value);
			m_values_sum[key] += value;
			m_values_sum_of_squares[key] += value * value;
		}
	}
}
