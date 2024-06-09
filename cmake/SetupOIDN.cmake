if (WIN32)
	set(OIDN_URL https://github.com/RenderKit/oidn/releases/download/v2.3.0-beta/oidn-2.3.0-beta.x64.windows.zip)
elseif(UNIX)
	set(OIDN_URL https://github.com/RenderKit/oidn/releases/download/v2.3.0-beta/oidn-2.3.0-beta.x86_64.linux.tar.gz)
endif()

FetchContent_Declare(
	oidnbinaries
	URL      ${OIDN_URL}
)

FetchContent_MakeAvailable(
	oidnbinaries
)
