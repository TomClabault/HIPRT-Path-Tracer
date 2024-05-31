/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef ALIGN_MACRO_H
#define ALIGN_MACRO_H

#ifdef __KERNELCC__
#define ALIGN(x) __align__(x)
#elif defined (_MSC_VER) && (_MSC_VER >= 1300)
#define ALIGN(x) __declspec(align(x))
#elif defined __GNUC__ || __clang__
#define ALIGN(x)  __attribute__((aligned(x)))
#endif
#endif
