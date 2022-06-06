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

#include "openvino/opsets/opset8.hpp"

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

void Cnn::Init(const std::string& model_path, ID3D11Device*& d3d_device, const cv::Size& new_input_resolution) {
    //// ---------------------------------------------------------------------------------------------------

    //// --------------------------- 1. Reading network ----------------------------------------------------
    ov::Core core;
    auto model = core.read_model(model_path);


    //// --------------------------- 2. Preprocess ----------------------------------------------------
    ov::preprocess::PrePostProcessor ppp(model);

    ppp.input().tensor().
        set_layout("NHWC").
        set_element_type(ov::element::u8).
        set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, { "y", "uv" }).
        //
        set_spatial_static_shape(480, 640).
        set_memory_type(ov::intel_gpu::memory_type::surface);

    // 3) Adding explicit preprocessing steps:
    // - convert u8 to f32
    // - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
    ppp.input().preprocess()
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .convert_layout("NCHW")
        .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
        .convert_element_type(ov::element::f32)
        .mean(127.5)
        .scale(127.5);

    ppp.input().model().set_layout("NCHW");

    ppp.output().tensor()
        .set_element_type(ov::element::u8);

     //ppp.output().postprocess()
     //    .convert_layout("NHWC");

    model = ppp.build();

    // --------------------------- Loading model to the device -------------------------------------------
    ov::intel_gpu::ocl::D3DContext gpu_context(core, d3d_device);
    remote_context = gpu_context;
    compile_model = core.compile_model(model, remote_context); // change device to RemoteContext
    //ov::serialize(exec_net_shared.get_runtime_model(), "test_graph.xml");

    //// --------------------------- Creating infer request ------------------------------------------------
    infer_request = compile_model.create_infer_request();
    is_initialized_ = true;
}

void Cnn::Infer(ID3D11Texture2D* input_surface, uint8_t* output_data)
{
    ov::Shape input_shape = { 1, 480, 640 ,1 };
    //auto remote_context = (ov::intel_gpu::ocl::D3DContext)compile_model.get_context();
    auto shared_in_blob = remote_context.create_tensor_nv12(input_shape[1], input_shape[2], input_surface);
    infer_request.set_tensor("inputImage/y", shared_in_blob.first);
    infer_request.set_tensor("inputImage/uv", shared_in_blob.second);
    infer_request.infer();

    const ov::Tensor& output_tensor = infer_request.get_output_tensor(0);
    output_data = output_tensor.data<uint8_t>();

    int rows = 720;
    int cols = 1280;
    cv::Mat outputImage(cv::Size(cols,rows),CV_8UC3, output_data);

    //postprocessing
    //cv::Mat outputImage;
    //if (data_size == rows * cols * 3) // check that the rows and cols match the size of your vector
    //{
    //    //copy vector to mat //NCHW
    //    cv::Mat channelR(rows, cols, CV_8UC1, data); // CV_32FC1
    //    cv::Mat channelG(rows, cols, CV_8UC1, data + cols * rows); // CV_32FC1
    //    cv::Mat channelB(rows, cols, CV_8UC1, data + 2 * cols * rows); // CV_32FC1
    //    // RGB2BGR
    //    std::vector<cv::Mat> channels{ channelB, channelG, channelR };

    //    // Create the output matrix
    //    merge(channels, outputImage);  
    //}
    // normolize  (-1,1) to (0,255)
    //cv::normalize(outputImage, outputImage, 0, 255, cv::NORM_MINMAX);
    //cv::cvtColor(outputImage, outputImage, cv::COLOR_RGB2BGR);
    cv::imwrite("styled.png", outputImage);
}