/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Utils/Debug.h"

#include <signal.h>

void Debug::debugbreak()
{
#if defined( _WIN32 )
    __debugbreak();
#elif defined( __GNUC__ )
    raise(SIGTRAP);
#else
    ;
#endif
}
