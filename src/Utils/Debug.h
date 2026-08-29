/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef UTILS_DEBUG_H
#define UTILS_DEBUG_H

class Debug
{
public:
	/**
	 * Breaks the debugger when calling this function as if a breakpoint was hit.
	 * Useful to be able to inspect the callstack at a given point in the program
	 */
	static void debugbreak();
};

#endif // #ifndef UTILS_DEBUG_H
