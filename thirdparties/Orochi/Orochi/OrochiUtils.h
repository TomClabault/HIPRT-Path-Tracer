#pragma once
#include <Orochi/Orochi.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined( GNUC )
#include <signal.h>
#endif

template<typename T, typename U>
constexpr void OROASSERT( T&& exp, [[maybe_unused]] U&& placeholder ) noexcept
{
	if( static_cast<bool>( std::forward<T>( exp ) ) != true )
	{

#if defined( _WIN32 )
		__debugbreak();
#elif defined( GNUC )
		raise( SIGTRAP );
#else
		;
#endif
	}
}

class OrochiUtils
{
  public:
	struct int4
	{
		int x, y, z, w;
	};

	OrochiUtils() = default;
	OrochiUtils(const OrochiUtils&) = delete; 
    OrochiUtils& operator=(const OrochiUtils&) = delete;
    OrochiUtils(OrochiUtils&&) = delete; 
    OrochiUtils& operator=(OrochiUtils&&) = delete;
	~OrochiUtils() = default;

	oroFunction getFunctionFromPrecompiledBinary( const std::string& path, const std::string& funcName );

	oroFunction getFunctionFromFile( oroDevice device, const char* path, const char* funcName, std::vector<const char*>* opts );
	oroFunction getFunctionFromString( oroDevice device, const char* source, const char* path, const char* funcName, std::vector<const char*>* opts, int numHeaders, const char** headers, const char** includeNames );
	oroFunction getFunction( oroDevice device, const char* code, const char* path, const char* funcName, std::vector<const char*>* opts, int numHeaders = 0, const char** headers = 0, const char** includeNames = 0 );

	static bool readSourceCode( const std::string& path, std::string& sourceCode, std::vector<std::string>* includes = 0 );
	static void getData( oroDevice device, const char* code, const char* path, std::vector<const char*>* opts, std::vector<char>& dst );
	static void getProgram( oroDevice device, const char* code, const char* path, std::vector<const char*>* optsIn, const char* funcName, orortcProgram* prog );
	static void getModule( oroDevice device, const char* code, const char* path, std::vector<const char*>* optsIn, const char* funcName, oroModule* moduleOut );
	static void launch1D( oroFunction func, int nx, const void** args, int wgSize = 64, unsigned int sharedMemBytes = 0, oroStream stream = 0 );
	static void launch2D( oroFunction func, int nx, int ny, const void** args, int wgSizeX = 8, int wgSizeY = 8, unsigned int sharedMemBytes = 0, oroStream stream = 0 );

	template<typename T>
	static void malloc( T*& ptr, size_t n )
	{
		oroError e = oroMalloc( (oroDeviceptr*)&ptr, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void mallocManaged( T*& ptr, size_t n, oroManagedMemoryAttachFlags flags )
	{
		oroError e = oroMallocManaged( (oroDeviceptr*)&ptr, sizeof( T ) * n, flags );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void free( T* ptr )
	{
		oroFree( (oroDeviceptr)ptr );
	}

	static void memset( void* ptr, int val, size_t n )
	{
		oroError e = oroMemset( (oroDeviceptr)ptr, val, n );
		OROASSERT( e == oroSuccess, 0 );
	}

	static void memsetAsync( void* ptr, int val, size_t n, oroStream stream )
	{
		oroError e = oroMemsetD8Async( (oroDeviceptr)ptr, val, n, stream );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyHtoD( T* dst, const T* src, size_t n )
	{
		oroError e = oroMemcpyHtoD( (oroDeviceptr)dst, (void*)src, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyDtoH( T* dst, T* src, size_t n )
	{
		oroError e = oroMemcpyDtoH( (void*)dst, (oroDeviceptr)src, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyDtoD( T* dst, T* src, size_t n )
	{
		oroError e = oroMemcpyDtoD( (oroDeviceptr)dst, (oroDeviceptr)src, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyHtoDAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		oroError e = oroMemcpyHtoDAsync( (oroDeviceptr)dst, (void*)src, sizeof( T ) * n, stream );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyDtoHAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		oroError e = oroMemcpyDtoHAsync( (void*)dst, (oroDeviceptr)src, sizeof( T ) * n, stream );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyDtoDAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		oroError e = oroMemcpyDtoDAsync( (oroDeviceptr)dst, (oroDeviceptr)src, sizeof( T ) * n, stream );
		OROASSERT( e == oroSuccess, 0 );
	}

	static void waitForCompletion( oroStream stream = 0 )
	{
		auto e = oroStreamSynchronize( stream );
		OROASSERT( e == oroSuccess, 0 );
	}

  public:
	std::string m_cacheDirectory = "./cache/";
	std::recursive_mutex m_mutex;
	std::unordered_map<std::string, oroFunction> m_kernelMap;
};

class OroStopwatch
{
  public:
	OroStopwatch( oroStream stream ) 
	{ 
		m_stream = stream;
		oroEventCreateWithFlags( &m_start, oroEventDefault );
		oroEventCreateWithFlags( &m_stop, oroEventDefault );
	}
	~OroStopwatch() 
	{
		oroEventDestroy( m_start );
		oroEventDestroy( m_stop );
	}

	void start() { oroEventRecord( m_start, m_stream ); }
	void stop() { oroEventRecord( m_stop, m_stream ); }

	float getMs() 
	{ 
		oroEventSynchronize( m_stop );
		float ms = 0;
		oroEventElapsedTime( &ms, m_start, m_stop );
		return ms;
	}

  public:
	oroStream m_stream;
	oroEvent m_start;
	oroEvent m_stop;
};