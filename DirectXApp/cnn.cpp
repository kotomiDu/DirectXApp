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
#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/runtime/intel_gpu/ocl/dx.hpp"


void Cnn::Init(const std::string &model_path,  ID3D11Device*& d3d_device, const cv::Size &new_input_resolution) {
    //// ---------------------------------------------------------------------------------------------------

    //// --------------------------- 1. Reading network ----------------------------------------------------
     ov::Core core;

    auto model = core.read_model(model_path);


    ov::preprocess::PrePostProcessor ppp(model);

    ppp.input()
        .tensor()
        .set_element_type(ov::element::f32)
        .set_color_format(ov::preprocess::ColorFormat::RGBX)
        .set_memory_type(ov::intel_gpu::memory_type::surface);

    ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    ppp.input().model().set_layout("NCHW");
    // --------------------------- Loading model to the device -------------------------------------------
    //auto compile_model = core.compile_model(model,"GPU");
    //auto gpu_context = core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
    //// Extract ocl context handle from RemoteContext
    //cl_context context_handle = gpu_context.get();
    ov::intel_gpu::ocl::D3DContext gpu_context(core, d3d_device);
    remote_context = &gpu_context;
    auto exec_net_shared = core.compile_model(model, *remote_context); // change device to RemoteContext

    auto input = model->get_parameters().at(0);
    infer_request = exec_net_shared.create_infer_request();
    //// --------------------------- Creating infer request ------------------------------------------------
    //infer_request_ = executable_network.CreateInferRequest();
    //// ---------------------------------------------------------------------------------------------------

    is_initialized_ = true;
}


void Cnn::Infer(ID3D11Texture2D* surface) {
    ov::Shape input_shape = { 1,640, 480 ,4};
    auto shared_in_blob = remote_context->create_tensor(ov::element::f32, input_shape, surface);
    infer_request.set_input_tensor(shared_in_blob);
    infer_request.infer();
}

