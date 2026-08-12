/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_NEURAL_NISML_TRAIN_PROFILING_H
#define HOST_DEVICE_COMMON_NEURAL_NISML_TRAIN_PROFILING_H

#ifdef PROFILING_ENABLED

// HIP documents clock() and clock64() as unreliable on AMD RDNA3/GFX11. Keep both opt-in for experiments, but use
// wall_clock64() by default on the RX 7900 XTX.
#define NISML_TRAIN_PROFILE_USE_CLOCK	0
#define NISML_TRAIN_PROFILE_USE_CLOCK64 0

enum NISMLTrainProfilePhase : unsigned int
{
	NISML_TRAIN_PROFILE_INPUT_ENCODING = 0,
	NISML_TRAIN_PROFILE_INPUT_ACTIVATION_STORE,
	NISML_TRAIN_PROFILE_FORWARD,
	NISML_TRAIN_PROFILE_OUTPUT_RESIDUALS,
	NISML_TRAIN_PROFILE_BASELINE,
	NISML_TRAIN_PROFILE_SOFTMAX,
	NISML_TRAIN_PROFILE_SAMPLE_WEIGHT,
	NISML_TRAIN_PROFILE_OUTPUT_GRADIENT,
	NISML_TRAIN_PROFILE_ERROR_SCALE,
	NISML_TRAIN_PROFILE_OUTPUT_ERROR_INITIALIZATION,
	NISML_TRAIN_PROFILE_ACTIVATION_RELOAD,
	NISML_TRAIN_PROFILE_WEIGHT_GRADIENTS,
	NISML_TRAIN_PROFILE_BIAS_GRADIENTS,
	NISML_TRAIN_PROFILE_ERROR_PROPAGATION,
	NISML_TRAIN_PROFILE_INPUT_GRADIENTS,
	NISML_TRAIN_PROFILE_GRID_GRADIENTS,
	NISML_TRAIN_PROFILE_PHASE_COUNT
};

struct NISMLTrainProfileRecord
{
	unsigned long long int phase_durations[NISML_TRAIN_PROFILE_PHASE_COUNT] = {};
};

#if defined(__KERNELCC__) && NISML_HAS_WMMA
#if NISML_TRAIN_PROFILE_USE_CLOCK
#define NISML_TRAIN_PROFILE_CLOCK() clock()
#elif NISML_TRAIN_PROFILE_USE_CLOCK64
#define NISML_TRAIN_PROFILE_CLOCK() clock64()
#else
#define NISML_TRAIN_PROFILE_CLOCK() wall_clock64()
#endif

#define NISML_TRAIN_PROFILE_START(profile_record, start)                                                                                                       \
	do                                                                                                                                                         \
	{                                                                                                                                                          \
		__syncthreads();                                                                                                                                       \
		if (threadIdx.x == 0)                                                                                                                                  \
			(start) = NISML_TRAIN_PROFILE_CLOCK();                                                                                                             \
		__syncthreads();                                                                                                                                       \
	} while (0)

#define NISML_TRAIN_PROFILE_STOP(profile_record, phase, start)                                                                                                 \
	do                                                                                                                                                         \
	{                                                                                                                                                          \
		__syncthreads();                                                                                                                                       \
		if (threadIdx.x == 0)                                                                                                                                  \
			(profile_record)->phase_durations[phase] = NISML_TRAIN_PROFILE_CLOCK() - (start);                                                                  \
		__syncthreads();                                                                                                                                       \
	} while (0)

#define NISML_TRAIN_PROFILE_STOP_ACCUMULATE(profile_record, phase, start)                                                                                      \
	do                                                                                                                                                         \
	{                                                                                                                                                          \
		__syncthreads();                                                                                                                                       \
		if (threadIdx.x == 0)                                                                                                                                  \
			(profile_record)->phase_durations[phase] += NISML_TRAIN_PROFILE_CLOCK() - (start);                                                                 \
		__syncthreads();                                                                                                                                       \
	} while (0)
#else
#define NISML_TRAIN_PROFILE_START(profile_record, start)                                                                                                       \
	do                                                                                                                                                         \
	{                                                                                                                                                          \
	} while (0)

#define NISML_TRAIN_PROFILE_STOP(profile_record, phase, start)                                                                                                 \
	do                                                                                                                                                         \
	{                                                                                                                                                          \
	} while (0)

#define NISML_TRAIN_PROFILE_STOP_ACCUMULATE(profile_record, phase, start)                                                                                      \
	do                                                                                                                                                         \
	{                                                                                                                                                          \
	} while (0)
#endif

#endif

#endif
