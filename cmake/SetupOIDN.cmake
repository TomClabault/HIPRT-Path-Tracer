if (WIN32)
	set(OIDN_URL https://github.com/RenderKit/oidn/releases/download/v2.4.1/oidn-2.4.1.x64.windows.zip)
elseif(UNIX)
	set(OIDN_URL https://github.com/RenderKit/oidn/releases/download/v2.4.1/oidn-2.4.1.x86_64.linux.tar.gz)
endif()

FetchContent_Declare(
	oidnbinaries
	URL      ${OIDN_URL}
)

FetchContent_MakeAvailable(
	oidnbinaries
)
