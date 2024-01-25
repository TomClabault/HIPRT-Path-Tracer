#ifndef OROCHI_UTILS_H
#define OROCHI_UTILS_H

#include "Orochi/Orochi.h"

#define OROCHI_CHECK_ERROR( error ) ( orochi_check_error( error, __FILE__, __LINE__ ) )
#define HIPRT_CHECK_ERROR( error ) ( hiprt_check_error( error, __FILE__, __LINE__ ) )

inline void orochi_check_error( oroError res, const char* file, uint32_t line )
{
	if ( res != oroSuccess )
	{
		const char* msg;
		oroGetErrorString( res, &msg );
		std::cerr << "Orochi error: '" << msg << "' on line " << line << " " << " in '" << file << "'." << std::endl;

		exit( EXIT_FAILURE );
	}
}

inline void hiprt_check_error(hiprtError res, const char* file, uint32_t line)
{
	if (res != hiprtSuccess)
	{
		std::cerr << "HIPRT error: '" << res << "' on line " << line << " " << " in '" << file << "'." << std::endl;

		exit(EXIT_FAILURE);
	}
}

#endif
