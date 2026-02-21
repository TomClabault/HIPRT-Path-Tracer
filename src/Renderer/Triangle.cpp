/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/Triangle.h"

float3_t Triangle::bbox_centroid() const { return (hippt::min(m_a, hippt::min(m_b, m_c)) + hippt::max(m_a, hippt::max(m_b, m_c))) / 2.0f; }

float Triangle::area() const { return hippt::length(hippt::cross(m_b - m_a, m_c - m_a)) / 2; }

float3_t& Triangle::operator[](int i) { return *((&m_a) + i); }

const float3_t& Triangle::operator[](int i) const { return *((&m_a) + i); }
