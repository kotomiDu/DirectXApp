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

void Cnn::Init(const std::string& model_path, ID3D11Device*& d3d_device, cv::Mat input_data)
{
    ov::Core core;

    auto model = core.read_model(model_path);

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input()
        .tensor()
        .set_layout("NHWC")
        .set_element_type(ov::element::u8)
        .set_shape({ 1,480,640,3 });


    ppp.input().preprocess()
        .convert_layout("NCHW")
        .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
        .convert_element_type(ov::element::f32);

    ppp.input().model().set_layout("NCHW");

    ppp.output().tensor()
        .set_element_type(ov::element::f32);

    model = ppp.build();

    auto compiled_model = core.compile_model(model,"GPU");
    infer_request = compiled_model.create_infer_request();

    ov::element::Type input_type = ov::element::u8;
    ov::Shape input_shape = { 1, 480, 640, 3 };
    ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, input_data.data);
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
}

void Cnn::Init(const std::string &model_path,  ID3D11Device*& d3d_device, ID3D11Texture2D* input_surface, ID3D11Buffer* output_surface, const cv::Size &new_input_resolution) {
    //// ---------------------------------------------------------------------------------------------------

    //// --------------------------- 1. Reading network ----------------------------------------------------
     ov::Core core;

    auto model = core.read_model(model_path);


    ov::preprocess::PrePostProcessor ppp(model);

    ppp.input().tensor().
        set_element_type(ov::element::u8).
        set_color_format(ov::preprocess::ColorFormat::NV12_SINGLE_PLANE).
        set_layout("NHWC").
        set_spatial_static_shape(480, 640).
        set_memory_type(ov::intel_gpu::memory_type::surface);;
    // 3) Adding explicit preprocessing steps:
    // - convert u8 to f32
    // - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
    ppp.input().preprocess().
        convert_element_type(ov::element::f32).
        convert_color(ov::preprocess::ColorFormat::BGR).
        convert_layout("NCHW").
        resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
        ; 

    ppp.input().model().set_layout("NCHW");

    ppp.output().tensor()
        .set_element_type(ov::element::f32);

    ppp.output().postprocess()
        .convert_layout("NHWC");

    model = ppp.build();

    // output [1,3,720,1280]
   
    // --------------------------- Loading model to the device -------------------------------------------
    //auto compile_model = core.compile_model(model,"GPU");
    //auto gpu_context = core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
    //// Extract ocl context handle from RemoteContext
    //cl_context context_handle = gpu_context.get();
    ov::intel_gpu::ocl::D3DContext gpu_context(core, d3d_device);

    remote_context = &gpu_context;
    auto exec_net_shared = core.compile_model(model, gpu_context); // change device to RemoteContext
    ov::serialize(exec_net_shared.get_runtime_model(),"test_graph.xml");


    auto input = model->get_parameters().at(0);
    infer_request = exec_net_shared.create_infer_request();
    //// --------------------------- Creating infer request ------------------------------------------------
    //infer_request_ = executable_network.CreateInferRequest();
    //// ---------------------------------------------------------------------------------------------------

    ov::Shape input_shape = { 1, 720, 640 ,1};
    ov::Shape output_shape = { 1, 720,1280, 3 };
    auto shared_in_blob = gpu_context.create_tensor(ov::element::u8, input_shape, input_surface);
    //auto shared_output_blob = gpu_context.create_tensor(ov::element::f32, output_shape, output_surface);
    infer_request.set_input_tensor(shared_in_blob);
    //infer_request.set_output_tensor(shared_output_blob);
    infer_request.infer();
    is_initialized_ = true;
}


void Cnn::Infer(ID3D11Texture2D* surface) {
    ov::Shape input_shape = { 1,640, 480 ,4};
    auto shared_in_blob = remote_context->create_tensor(ov::element::f32, input_shape, surface);
    infer_request.set_input_tensor(shared_in_blob);
    infer_request.infer();
}

