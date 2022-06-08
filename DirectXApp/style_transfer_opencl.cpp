#include "style_transfer_opencl.h"

#include <thread>
#include <iostream>

#define MAX_PLATFORMS       32
#define MAX_STRING_SIZE     1024

#define INIT_CL_EXT_FUNC(x)    x = (x ## _fn)clGetExtensionFunctionAddress(#x);
#define SAFE_OCL_FREE(P, FREE_FUNC)  { if (P) { FREE_FUNC(P); P = NULL; } }
#define EXT_INIT(_p, _name) _name = (_name##_fn) clGetExtensionFunctionAddressForPlatform((_p), #_name); res &= (_name != NULL);

namespace StyleTransfer {
    // OCLProgram methods
    OCLProgram::OCLProgram(OCLEnv* env) : m_program(nullptr), m_env(env) {}

    OCLProgram::~OCLProgram() {}

    bool OCLProgram::Build(const std::string& buildSource, const std::string& buildOptions) {
        std::cout << "OCLProgram: Reading and compiling OCL kernels" << std::endl;

        cl_int error = CL_SUCCESS;
        const char* buildSrcPtr = buildSource.c_str();
        m_program = clCreateProgramWithSource(m_env->GetContext(), 1, &(buildSrcPtr), NULL, &error);
        if (error) {
            std::cout << "OpenCLFilter: clCreateProgramWithSource failed. Error code: " << error << std::endl;
            return error;
        }

        // Build OCL kernel
        cl_device_id pDev = m_env->GetDevice();
        error = clBuildProgram(m_program, 1, &(pDev), buildOptions.c_str(), NULL, NULL);
        if (error == CL_BUILD_PROGRAM_FAILURE)
        {
            size_t buildLogSize = 0;
            cl_int logStatus = clGetProgramBuildInfo(m_program, m_env->GetDevice(), CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
            std::vector<char> buildLog(buildLogSize + 1);
            logStatus = clGetProgramBuildInfo(m_program, m_env->GetDevice(), CL_PROGRAM_BUILD_LOG, buildLogSize, &buildLog[0], NULL);
            std::cerr << std::string(buildLog.begin(), buildLog.end()).c_str() << std::endl;
            return false;
        }

        return (error == CL_SUCCESS);
    }

    // OCLKernel methods
    OCLKernel::OCLKernel(OCLEnv* env) : m_env(env) {
    }
    OCLKernel::~OCLKernel() {}

    // OCLKernelArg methods
    OCLKernelArg::OCLKernelArg() : m_idx(0) {}
    OCLKernelArg::~OCLKernelArg() {}

    // OCLEnv methods
    OCLEnv::OCLEnv() :
        m_d3d11device(nullptr),
        m_cldevice(nullptr),
        m_clplatform(nullptr),
        m_clcontext(nullptr),
        m_clqueue(nullptr),
        m_type(OCL_GPU_UNDEFINED) {}

    OCLEnv::~OCLEnv() {
        SAFE_OCL_FREE(m_clcontext, clReleaseContext);
        SAFE_OCL_FREE(m_clqueue, clReleaseCommandQueue);
    }

    bool OCLEnv::Init(cl_platform_id clplatform) {
        m_clplatform = clplatform;
        bool res = true;

        EXT_INIT(m_clplatform, clGetDeviceIDsFromD3D11KHR);
        EXT_INIT(m_clplatform, clCreateFromD3D11Texture2DKHR);
        EXT_INIT(m_clplatform, clEnqueueAcquireD3D11ObjectsKHR);
        EXT_INIT(m_clplatform, clEnqueueReleaseD3D11ObjectsKHR);

        return res;
    }

    cl_command_queue OCLEnv::GetCommandQueue() {
        return m_clqueue;
    }

    bool OCLEnv::SetD3DDevice(ID3D11Device* device) {
        cl_int error = CL_SUCCESS;

        m_d3d11device = device;

        cl_uint numDevices = 0;
        error = clGetDeviceIDsFromD3D11KHR(m_clplatform,
            CL_D3D11_DEVICE_KHR,
            m_d3d11device,
            CL_PREFERRED_DEVICES_FOR_D3D11_KHR,
            1,
            &m_cldevice,
            &numDevices);

        if (error != CL_SUCCESS) {
            std::cerr << "OCLEnv: clGetDeviceIDsFromD3D11KHR failed. Error code: " << error << std::endl;
            return false;
        }

        // Create context
        const cl_context_properties props[] = { CL_CONTEXT_D3D11_DEVICE_KHR, (cl_context_properties)m_d3d11device,
            CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE, NULL };
        m_clcontext = clCreateContext(props, 1, &m_cldevice, NULL, NULL, &error);
        if (error != CL_SUCCESS) {
            std::cerr << "OCLEnv: clCreateContext failed. Error code: " << error << std::endl;
            return false;
        }

        // Create command queue
        m_clqueue = clCreateCommandQueue(m_clcontext, m_cldevice, 0, &error);
        if (!m_clqueue) {
            std::cerr << "OCLEnv: clCreateCommandQueue failed. Error code: " << error << std::endl;
            return false;
        }
        printf("Create command queue: %p\n", m_clqueue);

        // Check device type
        cl_bool hostUnifiedMemory;
        clGetDeviceInfo(m_cldevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(hostUnifiedMemory), &hostUnifiedMemory, nullptr);
        m_type = hostUnifiedMemory ? OCL_GPU_INTEGRATED : OCL_GPU_DISCRETE;

        std::cout << "OCLEnv: OCL device initiated. OCL device type: "
            << ((m_type == OCL_GPU_INTEGRATED) ? "OCL_GPU_INTEGRATED" : "OCL_GPU_DISCRETE") << std::endl;

        return (error == CL_SUCCESS);
    }

    cl_mem OCLEnv::CreateSharedSurface(ID3D11Texture2D* surf, int nView, bool bIsReadOnly) {
        auto it = m_sharedSurfs.find(SurfaceKey(surf, nView));
        if (it != m_sharedSurfs.end()) {
            return it->second;
        }

        cl_int error = CL_SUCCESS;
        cl_mem mem = clCreateFromD3D11Texture2DKHR(m_clcontext, bIsReadOnly ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE, surf, nView, &error);
        if (error != CL_SUCCESS) {
            std::cerr << "clCreateFromD3D11Texture2DKHR failed. Error code: " << error << std::endl;
            return nullptr;
        }
        m_sharedSurfs[SurfaceKey(surf, nView)] = mem;

        return mem;
    }

    bool OCLEnv::EnqueueAcquireSurfaces(cl_mem* surfaces, int nSurfaces, bool flushAndFinish)
    {
        std::lock_guard<std::mutex> lock(m_sharedSurfMutex);

        cl_command_queue cmdQueue = GetCommandQueue();
        if (!cmdQueue) {
            return false;
        }
        cl_int error = clEnqueueAcquireD3D11ObjectsKHR(cmdQueue, nSurfaces, surfaces, 0, NULL, NULL);
        if (error) {
            printf("clEnqueueAcquireD3D11ObjectsKHR (cmdQueue = %p) failed. Error code: %d\n", cmdQueue, error);
            return false;
        }

        if (flushAndFinish) {
            // flush & finish the command queue
            error = clFlush(cmdQueue);
            if (error) {
                std::cerr << "clFlush failed. Error code: " << error << std::endl;
                return false;
            }
            error = clFinish(cmdQueue);
            if (error) {
                std::cerr << "clFinish failed. Error code: " << error << std::endl;
                return false;
            }
        }

        return true;
    }

    bool OCLEnv::EnqueueReleaseSurfaces(cl_mem* surfaces, int nSurfaces, bool flushAndFinish)
    {
        std::lock_guard<std::mutex> lock(m_sharedSurfMutex);

        cl_command_queue cmdQueue = GetCommandQueue();
        if (!cmdQueue) {
            return false;
        }
        cl_int error = clEnqueueReleaseD3D11ObjectsKHR(cmdQueue, nSurfaces, surfaces, 0, NULL, NULL);
        if (error) {
            std::cerr << "clEnqueueReleaseD3D11ObjectsKHR failed. Error code: " << error << std::endl;
            return false;
        }

        if (flushAndFinish) {
            // flush & finish the command queue
            error = clFlush(cmdQueue);
            if (error) {
                std::cerr << "clFlush failed. Error code: " << error << std::endl;
                return false;
            }
            error = clFinish(cmdQueue);
            if (error) {
                std::cerr << "clFinish failed. Error code: " << error << std::endl;
                return false;
            }
        }
        return true;
    }

    bool OCLEnv::EnqueueAcquireSurfaces(cl_command_queue command_queue,
        cl_uint num_objects,
        const cl_mem* mem_objects,
        cl_uint num_events_in_wait_list,
        const cl_event* event_wait_list,
        cl_event* event) {
        cl_int error = clEnqueueAcquireD3D11ObjectsKHR(command_queue, num_objects, mem_objects, num_events_in_wait_list, event_wait_list, event);

        if (error) {
            std::cerr << "clEnqueueAcquireD3D11ObjectsKHR failed. Error code: " << error << std::endl;
            return false;
        }

        return true;
    }

    bool OCLEnv::EnqueueReleaseSurfaces(cl_command_queue command_queue,
        cl_uint num_objects,
        const cl_mem* mem_objects,
        cl_uint num_events_in_wait_list,
        const cl_event* event_wait_list,
        cl_event* event) {
        cl_int error = clEnqueueReleaseD3D11ObjectsKHR(command_queue, num_objects, mem_objects, num_events_in_wait_list, event_wait_list, event);

        if (error) {
            std::cerr << "clEnqueueReleaseD3D11ObjectsKHR failed. Error code: " << error << std::endl;
            return false;
        }

        return true;
    }

    //bool OCLEnv::EnqueueAcquireAllSurfaces() {
    //    std::vector<cl_mem> surfaces;
    //    printf("EnqueueAcquire: ");
    //    for (auto it = m_sharedSurfs.begin(); it != m_sharedSurfs.end(); ++it) {
    //        surfaces.push_back(it->second);
    //        printf("%p, ", it->second);
    //    }
    //    printf("\n");
    //    return EnqueueAcquireSurfaces(&surfaces[0], surfaces.size(), true);
    //}

    //bool OCLEnv::EnqueueReleaseAllSurfaces() {
    //    std::vector<cl_mem> surfaces;
    //    for (auto it = m_sharedSurfs.begin(); it != m_sharedSurfs.end(); ++it) {
    //        surfaces.push_back(it->second);
    //    }
    //    return EnqueueReleaseSurfaces(&surfaces[0], surfaces.size(), true);
    //}


    OCL::OCL() {}

    bool OCL::Init()
    {
        cl_int error = CL_SUCCESS;

        // Determine the number of installed OpenCL platforms
        cl_uint num_platforms = 0;
        error = clGetPlatformIDs(0, NULL, &num_platforms);
        if (error)
        {
            std::cerr << "OpenCL: Couldn't get platform IDs. \
                        Make sure your platform \
                        supports OpenCL and can find a proper library. Error Code:" << error << std::endl;
            return false;
        }

        // Get all of the handles to the installed OpenCL platforms
        std::vector<cl_platform_id> platforms(num_platforms);
        error = clGetPlatformIDs(num_platforms, &platforms[0], &num_platforms);
        if (error) {
            std::cerr << "OpenCL: Failed to get OCL platform IDs. Error Code: " << error << std::endl;
            return false;
        }

        // Find the platform handle for the installed Gen driver
        const size_t max_string_size = 1024;
        char platform[max_string_size];
        cl_device_id device_ids[2] = { 0 };
        for (unsigned int platform_index = 0; platform_index < num_platforms; platform_index++)
        {
            error = clGetPlatformInfo(platforms[platform_index], CL_PLATFORM_NAME, max_string_size, platform, NULL);
            if (error) {
                std::cerr << "OpenCL: Failed to get platform info. Error Code: " << error << std::endl;
                return false;
            }

            // Choose only GPU devices
            if (clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU,
                sizeof(device_ids) / sizeof(device_ids[0]), device_ids, 0) != CL_SUCCESS)
                continue;

            if (strstr(platform, "Intel")) // Use only Intel platfroms
            {
                std::cout << "OpenCL platform \"" << platform << "\" is used" << std::endl;
                std::shared_ptr<OCLEnv> env(new OCLEnv);
                if (env->Init(platforms[platform_index])) {
                    m_envs.push_back(env);
                }
                else {
                    std::cerr << "Faild to initialize OCL sharing extenstions" << std::endl;
                    return false;
                }
            }
        }
        if (0 == m_envs.size())
        {
            std::cerr << "OpenCLFilter: Didn't find an Intel platform!" << std::endl;
            return false;
        }

        return true;
    }

    std::shared_ptr<OCLEnv> OCL::GetEnv(ID3D11Device* dev) {
        if (!dev) {
            std::cerr << "D3D11Device pointer is invalid." << std::endl;
            return nullptr;
        }

        // Try to find compatible OCL environment
        for (int i = 0; i < m_envs.size(); i++) {
            if (m_envs[i]->SetD3DDevice(dev)) {
                return m_envs[i];
            }
        }
        std::cerr << "No matching OpenCL devices was found for D3D11 device." << std::endl;
        return nullptr;
    }

    // OCLKernelArgSharedSurface methods
    OCLKernelArgSurface::OCLKernelArgSurface() : m_hdl(nullptr) {}
    OCLKernelArgSurface::~OCLKernelArgSurface() {}

    bool OCLKernelArgSurface::Set(cl_kernel kernel) {
        cl_int error = CL_SUCCESS;
        error = clSetKernelArg(kernel, m_idx, sizeof(cl_mem), &m_hdl);
        if (error) {
            std::cerr << "clSetKernelArg failed. Error code: " << error << std::endl;
            return false;
        }
        return true;
    }

    // OCLKernelArgInt methods
    OCLKernelArgInt::OCLKernelArgInt() {}
    OCLKernelArgInt::~OCLKernelArgInt() {}
    bool OCLKernelArgInt::Set(cl_kernel kernel) {
        cl_int error = CL_SUCCESS;
        error = clSetKernelArg(kernel, m_idx, sizeof(cl_int), &m_val);
        if (error) {
            std::cerr << "clSetKernelArg failed. Error code: " << error << std::endl;
            return false;
        }
        return true;
    }

    // OCLKernelArgFloat methods
    OCLKernelArgFloat::OCLKernelArgFloat() {}
    OCLKernelArgFloat::~OCLKernelArgFloat() {}
    bool OCLKernelArgFloat::Set(cl_kernel kernel) {
        cl_int error = CL_SUCCESS;
        error = clSetKernelArg(kernel, m_idx, sizeof(cl_float), &m_val);
        if (error) {
            std::cerr << "clSetKernelArg failed. Error code: " << error << std::endl;
            return false;
        }
        return true;
    }

    // CopyMakeBorder methods
    CopyMakeBorder::CopyMakeBorder(OCLEnv* env) : OCLKernel(env) {
        m_globalWorkSizeY[2] = { 0 };
        m_globalWorkSizeUV[2] = { 0 };
        m_localWorkSizeY[2] = { 0 };
        m_localWorkSizeUV[2] = { 0 };

        m_inSurfY.SetIdx(0); // kernel Y argument 0
        m_argsY.push_back(&m_inSurfY);
        m_outSurfY.SetIdx(1); // kernel Y argument 1 
        m_argsY.push_back(&m_outSurfY);
        m_rowsY.SetIdx(2); // kernel argument 2
        m_argsY.push_back(&m_rowsY);
        m_colsY.SetIdx(3); // kernel argument 3
        m_argsY.push_back(&m_colsY);
        m_borderY.SetIdx(4); // kernel argument 4
        m_argsY.push_back(&m_borderY);

        m_inSurfUV.SetIdx(0); // kernel UV argument 0
        m_argsUV.push_back(&m_inSurfUV);
        m_outSurfUV.SetIdx(1); // kernel UV argument 1 
        m_argsUV.push_back(&m_outSurfUV);
        m_rowsUV.SetIdx(2); // kernel UV argument 2
        m_argsUV.push_back(&m_rowsUV);
        m_colsUV.SetIdx(3); // kernel UV argument 3
        m_argsUV.push_back(&m_colsUV);
        m_borderUV.SetIdx(4); // kernel UV argument 4
        m_argsUV.push_back(&m_borderUV);
    }

    CopyMakeBorder::~CopyMakeBorder() {}

    bool CopyMakeBorder::Create(cl_program program) {
        cl_int error = CL_SUCCESS;

        m_kernelY = clCreateKernel(program, "copyMakeBorder", &error);
        if (error) {
            std::cerr << "OpenCLFilter: clCreateKernel failed. Error code: " << error << std::endl;
            return false;
        }

        m_kernelUV = clCreateKernel(program, "copyMakeBorder", &error);
        if (error) {
            std::cerr << "OpenCLFilter: clCreateKernel failed. Error code: " << error << std::endl;
            return false;
        }
        return true;
    }
    bool CopyMakeBorder::Run() {
        std::vector<cl_mem> sharedSurfaces;
        for (int i = 0; i < m_argsY.size(); i++) {
            if (!(m_argsY[i]->Set(m_kernelY))) {
                return false;
            }

            if (m_argsY[i]->Type() == OCLKernelArg::OCL_KERNEL_ARG_SHARED_SURFACE) {
                cl_mem hdl = dynamic_cast<OCLKernelArgSharedSurface*>(m_argsY[i])->GetHDL();
                sharedSurfaces.push_back(hdl);
            }
        }

        for (int i = 0; i < m_argsUV.size(); i++) {
            if (!(m_argsUV[i]->Set(m_kernelUV))) {
                return false;
            }

            if (m_argsUV[i]->Type() == OCLKernelArg::OCL_KERNEL_ARG_SHARED_SURFACE) {
                cl_mem hdl = dynamic_cast<OCLKernelArgSharedSurface*>(m_argsUV[i])->GetHDL();
                sharedSurfaces.push_back(hdl);
            }
        }

        cl_int error = CL_SUCCESS;
        cl_command_queue cmdQueue = m_env->GetCommandQueue();
        if (!cmdQueue) {
            return false;
        }

        cl_event acquired[1];
        cl_event transformed[2];
        cl_event released[1];

        if (!m_env->EnqueueAcquireSurfaces(cmdQueue, sharedSurfaces.size(), &sharedSurfaces[0], 0, NULL, &acquired[0])) {
            return false;
        }

        error = clEnqueueNDRangeKernel(cmdQueue, m_kernelY, 2, NULL, m_globalWorkSizeY, m_localWorkSizeY, 1, acquired, &transformed[0]);
        if (error) {
            std::cerr << "clEnqueueNDRangeKernel failed. Error code: " << error << std::endl;
            return false;
        }

        error = clEnqueueNDRangeKernel(cmdQueue, m_kernelUV, 2, NULL, m_globalWorkSizeUV, m_localWorkSizeUV, 1, acquired, &transformed[1]);
        if (error) {
            std::cerr << "clEnqueueNDRangeKernel failed. Error code: " << error << std::endl;
            return false;
        }

        if (!m_env->EnqueueReleaseSurfaces(cmdQueue, sharedSurfaces.size(), &sharedSurfaces[0], 2, transformed, &released[0])) {
            return false;
        }

        // flush & finish the command queue
        error = clFlush(cmdQueue);
        if (error) {
            std::cerr << "clFlush failed. Error code: " << error << std::endl;
            return false;
        }

        clWaitForEvents(1, released);

        //error = clFinish(cmdQueue);
        //if (error) {
        //    std::cerr << "clFinish failed. Error code: " << error << std::endl;
        //    return false;
        //}

        return (error == CL_SUCCESS);
    }

    static size_t chooseLocalSize(
        size_t globalSize, // frame width or height
        size_t preferred)  // preferred local size
    {
        size_t ret = 1;
        while ((globalSize % ret == 0) && ret <= preferred)
        {
            ret <<= 1;
        }
        return ret >> 1;
    }

    bool CopyMakeBorder::SetArguments(ID3D11Texture2D* in_surf, ID3D11Texture2D* out_surf, int cols, int rows, int border) {
        if (in_surf == nullptr || out_surf == nullptr || cols == 0 || rows == 0 || border == 0) {
            std::cerr << "CopyMakeBorder: Bad argument" << std::endl;
            return false;
        }

        cl_mem in_hdlY = m_env->CreateSharedSurface(in_surf, 0, true);
        if (!in_hdlY) {
            return false;
        }
        cl_mem in_hdlUV = m_env->CreateSharedSurface(in_surf, 1, true);
        if (!in_hdlUV) {
            return false;
        }

        cl_mem out_hdlY = m_env->CreateSharedSurface(out_surf, 0, false);
        if (!out_hdlY) {
            return false;
        }
        cl_mem out_hdlUV = m_env->CreateSharedSurface(out_surf, 1, false);
        if (!out_hdlUV) {
            return false;
        }

        m_inSurfY.SetHDL(in_hdlY);
        m_outSurfY.SetHDL(out_hdlY);

        m_inSurfUV.SetHDL(in_hdlUV);
        m_outSurfUV.SetHDL(out_hdlUV);

        m_colsY.SetVal(cols);
        m_rowsY.SetVal(rows);
        m_borderY.SetVal(border);

        m_colsUV.SetVal(cols / 2);
        m_rowsUV.SetVal(rows / 2);
        m_borderUV.SetVal(border / 2);

        // Work sizes for Y plane
        m_globalWorkSizeY[0] = cols;
        m_globalWorkSizeY[1] = rows;
        m_localWorkSizeY[0] = chooseLocalSize(m_globalWorkSizeY[0], 8);
        m_localWorkSizeY[1] = chooseLocalSize(m_globalWorkSizeY[1], 8);
        m_globalWorkSizeY[0] = m_localWorkSizeY[0] * (m_globalWorkSizeY[0] / m_localWorkSizeY[0]);
        m_globalWorkSizeY[1] = m_localWorkSizeY[1] * (m_globalWorkSizeY[1] / m_localWorkSizeY[1]);

        // Work size for UV plane
        m_globalWorkSizeUV[0] = cols / 2;
        m_globalWorkSizeUV[1] = rows / 2;
        m_localWorkSizeUV[0] = chooseLocalSize(m_globalWorkSizeUV[0], 8);
        m_localWorkSizeUV[1] = chooseLocalSize(m_globalWorkSizeUV[1], 8);
        m_globalWorkSizeUV[0] = m_localWorkSizeUV[0] * (m_globalWorkSizeUV[0] / m_localWorkSizeUV[0]);
        m_globalWorkSizeUV[1] = m_localWorkSizeUV[1] * (m_globalWorkSizeUV[1] / m_localWorkSizeUV[1]);

        return true;
    }

    //FmtConversion methods
    FmtConversion::FmtConversion(OCLEnv* env) : OCLKernel(env) {
        m_kernelNV12toRGB = nullptr;
        m_kernelRGBtoNV12 = nullptr;

        m_nv12ToRGB = true;

        m_globalWorkSize[2] = { 0 };

        m_surfY.SetIdx(0); // both kernels argument 0
        m_argsNV12toRGB.push_back(&m_surfY);
        m_argsRGBtoNV12.push_back(&m_surfY);

        m_surfUV.SetIdx(1); // both kernels argument 1
        m_argsNV12toRGB.push_back(&m_surfUV);
        m_argsRGBtoNV12.push_back(&m_surfUV);

        m_surfRGB.SetIdx(2); // both kernels argument 2
        m_argsNV12toRGB.push_back(&m_surfRGB);
        m_argsRGBtoNV12.push_back(&m_surfRGB);

        m_cols.SetIdx(3); // both kernels argument 3
        m_argsNV12toRGB.push_back(&m_cols);
        m_argsRGBtoNV12.push_back(&m_cols);

        m_channelSz.SetIdx(4); // both kernels argument 4
        m_argsNV12toRGB.push_back(&m_channelSz);
        m_argsRGBtoNV12.push_back(&m_channelSz);
    }

    FmtConversion::~FmtConversion() {}

    bool FmtConversion::Create(cl_program program) {
        cl_int error = CL_SUCCESS;

        m_kernelNV12toRGB = clCreateKernel(program, "convertNV12ToRGBfloat", &error);
        if (error) {
            std::cerr << "OpenCLFilter: clCreateKernel failed. Error code: " << error << std::endl;
            return false;
        }

        m_kernelRGBtoNV12 = clCreateKernel(program, "convertRGBfloatToNV12", &error);
        if (error) {
            std::cerr << "OpenCLFilter: clCreateKernel failed. Error code: " << error << std::endl;
            return false;
        }
        return true;
    }

    bool FmtConversion::SetArgumentsNV12toRGB(ID3D11Texture2D* in_nv12Surf, cl_mem out_rgbSurf, int cols, int rows) {
        cl_mem in_hdlY = m_env->CreateSharedSurface(in_nv12Surf, 0, true);
        if (!in_hdlY) {
            return false;
        }
        cl_mem in_hdlUV = m_env->CreateSharedSurface(in_nv12Surf, 1, true);
        if (!in_hdlUV) {
            return false;
        }
        m_surfY.SetHDL(in_hdlY);
        m_surfUV.SetHDL(in_hdlUV);

        m_surfRGB.SetHDL(out_rgbSurf);

        m_cols.SetVal(cols);
        m_channelSz.SetVal(cols * rows);

        m_globalWorkSize[0] = cols / 2;
        m_globalWorkSize[1] = rows / 2;

        m_nv12ToRGB = true;

        return true;
    }

    bool FmtConversion::SetArgumentsRGBtoNV12(cl_mem in_rgbSurf, ID3D11Texture2D* out_nv12Surf, int cols, int rows) {
        cl_mem out_hdlY = m_env->CreateSharedSurface(out_nv12Surf, 0, false);
        if (!out_hdlY) {
            return false;
        }
        cl_mem out_hdlUV = m_env->CreateSharedSurface(out_nv12Surf, 1, false);
        if (!out_hdlUV) {
            return false;
        }
        m_surfY.SetHDL(out_hdlY);
        m_surfUV.SetHDL(out_hdlUV);

        m_surfRGB.SetHDL(in_rgbSurf);

        m_cols.SetVal(cols);
        m_channelSz.SetVal(cols * rows);

        m_globalWorkSize[0] = cols / 2;
        m_globalWorkSize[1] = rows / 2;

        m_nv12ToRGB = false;

        return true;
    }

    bool FmtConversion::Run() {
        std::vector<cl_mem> sharedSurfaces;
        std::vector<OCLKernelArg*>& args = m_nv12ToRGB ? m_argsNV12toRGB : m_argsRGBtoNV12;
        cl_kernel& kernel = m_nv12ToRGB ? m_kernelNV12toRGB : m_kernelRGBtoNV12;

        for (int i = 0; i < args.size(); i++) {
            if (!(args[i]->Set(kernel))) {
                return false;
            }

            if (args[i]->Type() == OCLKernelArg::OCL_KERNEL_ARG_SHARED_SURFACE) {
                cl_mem hdl = dynamic_cast<OCLKernelArgSharedSurface*>(args[i])->GetHDL();
                sharedSurfaces.push_back(hdl);
            }
        }

        cl_int error = CL_SUCCESS;
        cl_command_queue cmdQueue = m_env->GetCommandQueue();
        if (!cmdQueue) {
            return false;
        }

        if (!m_env->EnqueueAcquireSurfaces(&sharedSurfaces[0], sharedSurfaces.size(), false)) {
            return false;
        }

        error = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, m_globalWorkSize, NULL, 0, NULL, NULL);
        if (error) {
            std::cerr << "clEnqueueNDRangeKernel failed. Error code: " << error << std::endl;
            return false;
        }

        if (!m_env->EnqueueReleaseSurfaces(&sharedSurfaces[0], sharedSurfaces.size(), false)) {
            return false;
        }

        // flush & finish the command queue
        error = clFlush(cmdQueue);
        if (error) {
            std::cerr << "clFlush failed. Error code: " << error << std::endl;
            return false;
        }
        error = clFinish(cmdQueue);
        if (error) {
            std::cerr << "clFinish failed. Error code: " << error << std::endl;
            return false;
        }

        return (error == CL_SUCCESS);
    }


    //SourceConversion methods
    SourceConversion::SourceConversion(OCLEnv* env) : OCLKernel(env) {
        m_argsRGBtoRGBmem.push_back(&m_surfRGB);
    }

    SourceConversion::~SourceConversion() {
      
    }

    bool SourceConversion::Create(cl_program program) {
        cl_int error = CL_SUCCESS;

        m_kernelRGBtoRGBbuffer = clCreateKernel(program, "convertRGBAToRGBfloat", &error);
        if (error) {
            std::cerr << "OpenCLFilter: clCreateKernel failed. Error code: " << error << std::endl;
            return false;
        }
        return true;
    }
    bool SourceConversion::SetArgumentsRGBtoRGBmem(ID3D11Texture2D* in_rgbSurf, cl_mem out_rgbSurf, int cols, int rows){
        out_rgbSurf = m_env->CreateSharedSurface(in_rgbSurf, 0, true); //rgb surface only has one view,default as 0
        if (!out_rgbSurf) {
            return false;
        }
        m_surfRGB.SetHDL(out_rgbSurf);

        m_globalWorkSize[0] = cols / 2;
        m_globalWorkSize[1] = rows / 2;
        return true;
    }
    bool SourceConversion::Run() {
        std::vector<cl_mem> sharedSurfaces;
        cl_mem hdl = dynamic_cast<OCLKernelArgSharedSurface*>(m_argsRGBtoRGBmem[0])->GetHDL();
        sharedSurfaces.push_back(hdl);

        cl_int error = CL_SUCCESS;
        cl_command_queue cmdQueue = m_env->GetCommandQueue();
        if (!cmdQueue) {
            return false;
        }

        error = clEnqueueNDRangeKernel(cmdQueue, m_kernelRGBtoRGBbuffer, 2, NULL, m_globalWorkSize, NULL, 0, NULL, NULL);
        if (error) {
            std::cerr << "clEnqueueNDRangeKernel failed. Error code: " << error << std::endl;
            return false;
        }

        if (!m_env->EnqueueAcquireSurfaces(&sharedSurfaces[0], sharedSurfaces.size(), false)) {
            return false;
        }

        // test
        /*uint8_t data[100] = {1,2,3,4};
        cl_int err_flag;
        cl_mem t = clCreateBuffer(m_env->GetContext(), CL_MEM_READ_WRITE| CL_MEM_USE_HOST_PTR, 100, data, &err_flag);
        if (err_flag)
        {
            std::cout << "erro" << std::endl;
        }*/
        // test_end
        //printClVector(hdl, 640*480*4, cmdQueue );

        if (!m_env->EnqueueReleaseSurfaces(&sharedSurfaces[0], sharedSurfaces.size(), false)) {
            return false;
        }

        // flush & finish the command queue
        error = clFlush(cmdQueue);
        if (error) {
            std::cerr << "clFlush failed. Error code: " << error << std::endl;
            return false;
        }
        error = clFinish(cmdQueue);
        if (error) {
            std::cerr << "clFinish failed. Error code: " << error << std::endl;
            return false;
        }

        return (error == CL_SUCCESS);
    }

    void SourceConversion::printClVector(cl_mem& clVector, int length, cl_command_queue& commands, int printrowlen )
    {
        uint8_t* tmp = new uint8_t[length];
        //int err = clEnqueueReadBuffer(commands, clVector, CL_TRUE, 0, sizeof(uint8_t) * length, tmp, 0, NULL, NULL);
        size_t origin[3] = {0,0,0};
        size_t region[3] = {640,480,1};
        int err = clEnqueueReadImage(commands, clVector, CL_TRUE, origin,region, 0, 0, tmp, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        if (printrowlen < 0)	// print all as one line
        {
            for (int k = 0; k < length; k++)
            {
                std::cout <<(int)tmp[k] << " ";
            }
            std::cout << std::endl;
        }
        else				// print rows of "printrowlen" length
        {
            for (int k = 0; k < length / printrowlen; k++)
            {
                for (int j = 0; j < printrowlen; j++)
                {
                    std::cout << tmp[k * printrowlen + j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        delete[] tmp;
    }

    //OCLFilterStore methods
    OCLFilterStore::OCLFilterStore(OCLEnv* env) : m_env(env), m_program(env) {}
    OCLFilterStore::~OCLFilterStore() {}

    bool OCLFilterStore::Create(const std::string& programSource) {
        std::string buildOptions = "-I. -Werror -cl-fast-relaxed-math";


        if (!m_program.Build(programSource, buildOptions)) {
            return false;
        }
        return true;
    }

    OCLKernel* OCLFilterStore::CreateKernel(const std::string& name) {
        OCLKernel* kernel = nullptr;
        if (name == "fmtConversion") {
            kernel = new FmtConversion(m_env);
            if (!kernel->Create(m_program.GetHDL())) {
                delete kernel;
                kernel = nullptr;
            }
        }
        else if (name == "copyMakeBorder") {
            kernel = new CopyMakeBorder(m_env);
            if (!kernel->Create(m_program.GetHDL())) {
                delete kernel;
                kernel = nullptr;
            }
        }else if (name == "srcConversion") {
            kernel = new SourceConversion(m_env);
            if (!kernel->Create(m_program.GetHDL())) {
                delete kernel;
                kernel = nullptr;
            }
        }
        return kernel;
    }
}