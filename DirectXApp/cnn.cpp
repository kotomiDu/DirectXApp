// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn.hpp"

#include <chrono>
#include <map>
#include <string>
#include <d3d11.h>
#include <windows.h>
//#include <gpu/gpu_context_api_dx.hpp>
#include <openvino/runtime/intel_gpu/ocl/dx.hpp>


void Cnn::Init(const std::string &model_path,  ID3D11Device*& d3d_device, const cv::Size &new_input_resolution) {
    //// ---------------------------------------------------------------------------------------------------

    //// --------------------------- 1. Reading network ----------------------------------------------------
     ov::Core core;

    auto model = core.read_model(model_path);
    ov::AnyMap config;
    //auto context = core.create_context("GPU", config);
   /* Core ie;
    auto network = ie.ReadNetwork(model_path);*/

    //// --------------------------- Changing input shape if it is needed ----------------------------------
    //InputsDataMap inputInfo(network.getInputsInfo());
    //if (inputInfo.size() != 1) {
    //    THROW_IE_EXCEPTION << "The network should have only one input";
    //}
    //InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;

    //SizeVector input_dims = inputInfoFirst->getInputData()->getTensorDesc().getDims();
    //input_dims[0] = 1;
    //if (new_input_resolution != cv::Size()) {
    //    input_dims[2] = static_cast<size_t>(new_input_resolution.height);
    //    input_dims[3] = static_cast<size_t>(new_input_resolution.width);
    //}

    //std::map<std::string, SizeVector> input_shapes;
    //input_shapes[network.getInputsInfo().begin()->first] = input_dims;
    //network.reshape(input_shapes);

    //// ---------------------------------------------------------------------------------------------------

    //// --------------------------- Configuring input and output ------------------------------------------
    //// ---------------------------   Preparing input blobs -----------------------------------------------
    //input_info_ = network.getInputsInfo().begin()->second;
    //input_name_ = network.getInputsInfo().begin()->first;

    //input_info_->setLayout(Layout::NCHW); //bfyx
    ////input_info->setPrecision(Precision::FP32);
    //input_info_->getPreProcess().setColorFormat(ColorFormat::RGBX);
    //channels_ = input_info_->getTensorDesc().getDims()[1];
    //input_size_ = cv::Size(input_info_->getTensorDesc().getDims()[3], input_info_->getTensorDesc().getDims()[2]);

    //// ---------------------------   Preparing output blobs ----------------------------------------------

    //OutputsDataMap output_info(network.getOutputsInfo());
    //for (auto output : output_info) {
    //    output_names_.emplace_back(output.first);
    //}

    // ---------------------------------------------------------------------------------------------------

    // --------------------------- Loading model to the device -------------------------------------------
    auto compile_model = core.compile_model(model,"GPU");
    auto gpu_context = core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
    //// Extract ocl context handle from RemoteContext
    cl_context context_handle = gpu_context.get();
    //ov::intel_gpu::ocl::D3DContext gpu_context(core, d3d_device);
    //auto exec_net_shared = core.compile_model(model, gpu_context); // change device to RemoteContext
    //remote_context_ = gpu::make_shared_context(ie, "GPU", &d3d_device);
    //ExecutableNetwork executable_network = ie.LoadNetwork(network, remote_context_); // change device to RemoteContext

    // ---------------------------------------------------------------------------------------------------

    //// --------------------------- Creating infer request ------------------------------------------------
    //infer_request_ = executable_network.CreateInferRequest();
    //// ---------------------------------------------------------------------------------------------------

    //is_initialized_ = true;
}


//void Cnn::Infer(ID3D11Texture2D* surface) {
//    auto shared_blob = gpu::make_shared_blob(input_info_->getTensorDesc(), remote_context_, surface);
//    infer_request_.SetBlob(input_name_,shared_blob);
//    infer_request_.Infer();
//
//    // --------------------------- Processing output -----------------------------------------------------
//
//    InferenceEngine::BlobMap blobs;
//    for (const auto& output_name : output_names_) {
//        blobs[output_name] = infer_request_.GetBlob(output_name);
//    }
//
//    auto output_shape = blobs.begin()->second->getTensorDesc().getDims();
//    int length = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];
//    LockedMemory<const void> blobMapped = as<MemoryBlob>(blobs.begin()->second)->rmap();
//    float* output_data_pointer = blobMapped.as<float*>();
//    std::vector<float> output_data(output_data_pointer, output_data_pointer + length);
//
//    std::cout << output_data[0] << std::endl;
//
//}
InferenceEngine::BlobMap Cnn::Infer(const cv::Mat &frame) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    /* Resize manually and copy data from the image to the input blob */
    InferenceEngine::LockedMemory<void> inputMapped =
        InferenceEngine::as<InferenceEngine::MemoryBlob>(infer_request_.GetBlob(input_name_))->wmap();
    float* input_data = inputMapped.as<float *>();

    cv::Mat image;
    if (channels_ == 1) {
         cv::cvtColor(frame, image, cv::COLOR_BGR2GRAY);
    } else {
        image = frame.clone();
    }

    image.convertTo(image, CV_32F);
    cv::resize(image, image, input_size_);
    int image_size = input_size_.area();

    if (channels_ == 3) {
        for (int pid = 0; pid < image_size; ++pid) {
            for (int ch = 0; ch < channels_; ++ch) {
                input_data[ch * image_size + pid] = image.at<cv::Vec3f>(pid)[ch];
            }
        }
    } else if (channels_ == 1) {
        for (int pid = 0; pid < image_size; ++pid) {
            input_data[pid] = image.at<float>(pid);
        }
    }

    // ---------------------------------------------------------------------------------------------------

    // --------------------------- Doing inference -------------------------------------------------------
    /* Running the request synchronously */
    infer_request_.Infer();
    //infer_request_.StartAsync();
    //infer_request_.Wait(IInferRequest::WaitMode::RESULT_READY);
    // ---------------------------------------------------------------------------------------------------

    // --------------------------- Processing output -----------------------------------------------------

    InferenceEngine::BlobMap blobs;
    for (const auto &output_name : output_names_) {
        blobs[output_name] = infer_request_.GetBlob(output_name);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    time_elapsed_ += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    ncalls_++;

    return blobs;
}
