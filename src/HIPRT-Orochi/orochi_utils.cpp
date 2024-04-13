/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HIPRT-Orochi/orochi_utils.h"

void orochi_check_error(oroError res, const char* file, uint32_t line)
{
	if (res != oroSuccess)
	{
		const char* msg;
		oroGetErrorString(res, &msg);
		std::cerr << "Orochi error: '" << msg << "' on line " << line << " " << " in '" << file << "'." << std::endl;

		exit(EXIT_FAILURE);
	}
}

void orochi_rtc_check_error(orortcResult res, const char* file, uint32_t line)
{
	if (res != ORORTC_SUCCESS)
	{
		std::cerr << "ORORTC error: '" << orortcGetErrorString(res) << "' [ " << res << " ] on line " << line << " "
			<< " in '" << file << "'." << std::endl;
		exit(EXIT_FAILURE);
	}
}

void hiprt_check_error(hiprtError res, const char* file, uint32_t line)
{
	if (res != hiprtSuccess)
	{
		std::cerr << "HIPRT error: '" << res << "' on line " << line << " " << " in '" << file << "'." << std::endl;

		exit(EXIT_FAILURE);
	}
}

bool readSourceCode(const std::filesystem::path& path, std::string& sourceCode, std::vector<std::filesystem::path>* includes)
{
	std::fstream f(path);
	if (f.is_open())
	{
		size_t sizeFile;
		f.seekg(0, std::fstream::end);
		size_t size = sizeFile = static_cast<size_t>(f.tellg());
		f.seekg(0, std::fstream::beg);
		if (includes)
		{
			sourceCode.clear();
			std::string line;
			while (std::getline(f, line))
			{
				if (line.find("#include") != std::string::npos)
				{
					int		pa = line.find("<");
					int		pb = line.find(">");
					if (pa == -1 || pb == -1)
					{
						pa = line.find("\"");
						if (pa != -1)
						{
							std::string substr = line.substr(pa + 1);
							pb = line.substr(pa + 1).find("\"") + pa + 1;
						}
					}

					std::string buf = line.substr(pa + 1, pb - pa - 1);
					includes->push_back(buf);
					sourceCode += line + '\n';
				}
				sourceCode += line + '\n';
			}
		}
		else
		{
			sourceCode.resize(size, ' ');
			f.read(&sourceCode[0], size);
		}
		f.close();
	}
	else
		return false;

	return true;
}

void buildTraceKernelFromBitcode(
	hiprtContext				   ctxt,
	const char* path,
	const char* functionName,
	oroFunction& functionOut,
	std::vector<std::string> include_paths,
	std::vector<std::pair<std::string, std::string>> precompiler_defines,
	std::vector<const char*>* opts,
	std::vector<hiprtFuncNameSet>* funcNameSets,
	uint32_t					   numGeomTypes,
	uint32_t					   numRayTypes)
{
	std::vector<const char*>		   options;
	std::vector<std::filesystem::path> includeNamesData;
	std::string						   sourceCode;

	if (!readSourceCode(path, sourceCode, &includeNamesData))
	{
		std::cerr << "Unable to find file '" << path << "'" << std::endl;
		
		exit(EXIT_FAILURE);
	}

	if (opts)
	{
		for (const auto o : *opts)
			options.push_back(o);
	}

	const bool isAmd = oroGetCurAPI(0) == ORO_API_HIP;
	if (isAmd)
	{
		options.push_back("-fgpu-rdc");
		options.push_back("-Xclang");
		options.push_back("-disable-llvm-passes");
		options.push_back("-Xclang");
		options.push_back("-mno-constructor-aliases");
	}
	else
	{
		options.push_back("--device-c");
		options.push_back("-arch=compute_60");
	}
	options.push_back("-std=c++17");

	for (std::string& include_path : include_paths)
	{
		include_path = std::string("-I") + include_path;
		options.push_back(include_path.c_str());
	}

	for (std::pair<std::string, std::string>& precompiler_option : precompiler_defines)
	{
		std::string option_string = "-D " + precompiler_option.first + "=" + precompiler_option.second;
		// Using the precompiler_option as the holder of the option string
		// If instead we only push_back(option_string.c_str()) to the options
		// this is going to a result into trash being added to the options after 
		// 'option_string' is destroyed at the end of the iteration of this loop
		precompiler_option.first = option_string;
		options.push_back(precompiler_option.first.c_str());
	}

	orortcProgram prog;
	OROCHI_RTC_CHECK_ERROR(orortcCreateProgram(&prog, sourceCode.data(), path, 0, NULL, NULL));
	OROCHI_RTC_CHECK_ERROR(orortcAddNameExpression(prog, functionName));

	orortcResult e = orortcCompileProgram(prog, static_cast<int>(options.size()), options.data());
	if (e != ORORTC_SUCCESS)
	{
		size_t logSize;
		OROCHI_RTC_CHECK_ERROR(orortcGetProgramLogSize(prog, &logSize));

		if (logSize)
		{
			std::string log(logSize, '\0');
			orortcGetProgramLog(prog, &log[0]);
			std::cerr << log << std::endl;
		}
		exit(EXIT_FAILURE);
	}

	std::string bitCodeBinary;
	size_t		size = 0;
	if (isAmd)
		OROCHI_RTC_CHECK_ERROR(orortcGetBitcodeSize(prog, &size));
	else
		OROCHI_RTC_CHECK_ERROR(orortcGetCodeSize(prog, &size));

	bitCodeBinary.resize(size);
	if (isAmd)
		OROCHI_RTC_CHECK_ERROR(orortcGetBitcode(prog, (char*)bitCodeBinary.data()));
	else
		OROCHI_RTC_CHECK_ERROR(orortcGetCode(prog, (char*)bitCodeBinary.data()));

	hiprtApiFunction function;
	HIPRT_CHECK_ERROR(hiprtBuildTraceKernelsFromBitcode(
		ctxt,
		1,
		&functionName,
		path,
		bitCodeBinary.data(),
		size,
		numGeomTypes,
		numRayTypes,
		funcNameSets != nullptr ? funcNameSets->data() : nullptr,
		&function,
		false));

	functionOut = *reinterpret_cast<oroFunction*>(&function);
}