#pragma once
#include "style_transfer_opencl.h"
#include <vector>

namespace StyleTransfer {
    OCLFilterStore* CreateFilterStore(StyleTransfer::OCLEnv* env, const std::string& oclFile);
};