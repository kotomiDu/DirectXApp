#include "style_transfer_util.h"
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/directx.hpp>

namespace StyleTransfer {
    static std::string getPathToExe()
    {
        const size_t module_length = 1024;
        char module_name[module_length];

#if defined(_WIN32) || defined(_WIN64)
        GetModuleFileNameA(0, module_name, module_length);
#else
        char id[module_length];
        sprintf(id, "/proc/%d/exe", getpid());
        ssize_t count = readlink(id, module_name, module_length - 1);
        if (count == -1)
            return std::string("");
        module_name[count] = '\0';
#endif

        std::string exePath(module_name);
        return exePath.substr(0, exePath.find_last_of("\\/") + 1);
    }
    static std::string readFile(const char* filename)
    {
        std::cout << "Info: try to open file (" << filename << ") in the current directory" << std::endl;
        std::ifstream input(filename, std::ios::in | std::ios::binary);

        if (!input.good())
        {
            // look in folder with executable
            input.clear();

            std::string module_name = getPathToExe() + std::string(filename);

            std::cout << "Info: try to open file: " << module_name.c_str() << std::endl;
            input.open(module_name.c_str(), std::ios::binary);
        }

        if (!input)
            throw std::logic_error((std::string("Error_opening_file_\"") + std::string(filename) + std::string("\"")).c_str());

        input.seekg(0, std::ios::end);
        std::vector<char> program_source(static_cast<int>(input.tellg()));
        input.seekg(0);

        input.read(&program_source[0], program_source.size());

        return std::string(program_source.begin(), program_source.end());
    }

    OCLFilterStore* CreateFilterStore(StyleTransfer::OCLEnv* env, const std::string& oclFile) {
        StyleTransfer::OCLFilterStore* filterStore = new StyleTransfer::OCLFilterStore(env);

        std::string buffer = readFile(oclFile.c_str());
        if (!filterStore->Create(buffer)) {
            return nullptr;
        }
        return filterStore;
    }
}