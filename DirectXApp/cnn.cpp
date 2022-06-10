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
#include <fstream>

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

void Cnn::Init(const std::string &model_path,  ID3D11Device*& d3d_device, cl_context ctx , const cv::Size &new_input_resolution) {
    //// ---------------------------------------------------------------------------------------------------

    //// --------------------------- 1. Reading network ----------------------------------------------------
     ov::Core core;

    auto model = core.read_model(model_path);


    ov::preprocess::PrePostProcessor ppp(model);

    ppp.input().tensor().
        set_layout("NCHW").
        set_element_type(ov::element::u8).
        set_color_format(ov::preprocess::ColorFormat::RGB).
        //set_shape({ 1,3,480,640}).
        set_memory_type(ov::intel_gpu::memory_type::buffer);

    // 3) Adding explicit preprocessing steps:
    // - convert u8 to f32
    // - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
    ppp.input().preprocess()
        //.convert_color(ov::preprocess::ColorFormat::RGB)
        //.convert_layout("NCHW")
        //.resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
        .convert_element_type(ov::element::f32)
        .mean(127.5)
        .scale(127.5);

    ppp.input().model().set_layout("NCHW");

    
    ppp.output().tensor()
        .set_element_type(ov::element::u8);

    model = ppp.build();

    ov::Shape  input_shape = { 1,3,static_cast<size_t>(new_input_resolution.height), static_cast<size_t>(new_input_resolution.width) };
    model->reshape(input_shape);
    
    //for (const auto& op : model->get_ops()) {
    //    //if (op->get_name() != "Transpose_455") continue;
    //    if (!std::dynamic_pointer_cast<ov::opset8::Result>(op) &&
    //        !std::dynamic_pointer_cast<ov::opset8::Parameter>(op) &&
    //        !std::dynamic_pointer_cast<ov::opset8::VariadicSplit>(op) &&
    //        !std::dynamic_pointer_cast<ov::opset8::TopK>(op))
    //    {
    //        //model->add_output(op);
    //        std::cout << op->get_name() << std::endl;
    //    }

    //}
   
    // --------------------------- Loading model to the device -------------------------------------------
    //auto compile_model = core.compile_model(model,"GPU");
    //auto gpu_context = core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
    //// Extract ocl context handle from RemoteContext
    //cl_context context_handle = gpu_context.get();
    auto remote_context = ov::intel_gpu::ocl::ClContext(core, ctx);
    _oclCtx = ctx;
    compiled_model = core.compile_model(model, remote_context); // change device to RemoteContext
    ov::serialize(compiled_model.get_runtime_model(),"test_graph.xml");

    //// --------------------------- Creating infer request ------------------------------------------------
    infer_request = compiled_model.create_infer_request();

    //// Create input and output GPU Blobs
    
    _inputBuffer = cl::Buffer(_oclCtx, CL_MEM_READ_WRITE, input_shape[1] * input_shape[2] * input_shape[3] * sizeof(uint8_t), NULL, NULL);
    _outputBuffer = cl::Buffer(_oclCtx, CL_MEM_READ_WRITE, input_shape[1] * input_shape[2] * input_shape[3] * sizeof(uint8_t), NULL, NULL);
    auto shared_in_blob = remote_context.create_tensor(ov::element::u8, input_shape, _inputBuffer);
    auto shared_output_blob = remote_context.create_tensor(ov::element::u8, input_shape, _outputBuffer);  //style transfer output has the same shape with input
    infer_request.set_input_tensor(shared_in_blob);
    infer_request.set_output_tensor(shared_output_blob);

    is_initialized_ = true;
}


bool Cnn::Infer(StyleTransfer::SourceConversion& srcConversionKrnl, ID3D11Texture2D* input_surface, ID3D11Texture2D* output_surface, const cv::Size& surface_size)
{


    if (!srcConversionKrnl.SetArgumentsRGBtoRGBbuffer(input_surface, _inputBuffer.get(), surface_size.width, surface_size.height)) {
        return false;
    }
    if (!srcConversionKrnl.Run()) {
        return false;
    }


    infer_request.infer();

    if (!srcConversionKrnl.SetArgumentsRGBbuffertoRGBA( _outputBuffer.get(), output_surface, surface_size.width, surface_size.height)) {
        return false;
    }
    if (!srcConversionKrnl.Run()) {
        return false;
    }
    // middle output
    //int idx = 0;
    //for (auto&& output : compiled_model.outputs()) {
    //    const ov::Tensor& output_tensor = infer_request.get_output_tensor(idx);
    //  
    //    auto data = output_tensor.data<uint8_t>();
    //    int rows = 480;
    //    int cols = 640;
    //    cv::Mat outputImage(cv::Size(cols, rows), CV_8UC3, data);
    //    auto data_size = output_tensor.get_size();
    //    if (idx == 1) {
    //        std::cout << std::endl;
    //        continue;
    //    }

    //    std::ofstream myfile;
    //    myfile.open("log.txt", std::ios::out | std::ios::binary);
    //    for (size_t i = 0; i < data_size; i++) {
    //        //std::cout << (int)data[i] << " ";
    //        myfile << data[i];
    //        //myfile << (int)data[i] << " ";
    //        //if (i > 1280*5) break;
    //    }
    //    myfile.close();
    //    std::cout << std::endl;
    //    idx++;
    //}
    ////final  output
 /*   for (auto&& output : compiled_model.outputs()) {
        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        if (name == "NONE")
        {
            continue;
        }

        const ov::Tensor& output_tensor = infer_request.get_tensor(name);

        auto data_size = output_tensor.get_size();
        auto data = output_tensor.data<uint8_t>();
        int rows = 720;
        int cols = 1280;
        cv::Mat outputImage(cv::Size(cols, rows), CV_8UC3, data);
        cv::imwrite("styled.png", outputImage);
    }*/

}