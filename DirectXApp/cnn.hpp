// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <d3d11.h>
#include <windows.h>

#include <opencv2/opencv.hpp>
#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/runtime/intel_gpu/ocl/dx.hpp"
#include "style_transfer_opencl.h"
using namespace InferenceEngine;

class Cnn {
  public:
    Cnn():is_initialized_(false), channels_(0), time_elapsed_(0), ncalls_(0) {}

    void Init(const std::string &model_path,  ID3D11Device*& d3d_device, cl_context ctx ,
              const cv::Size &new_input_resolution = cv::Size());

    void Init(const std::string& model_path, const cv::Size& new_input_resolution = cv::Size());

    bool is_initialized() const {return is_initialized_;}

    size_t ncalls() const {return ncalls_;}
    double time_elapsed() const {return time_elapsed_;}

    const cv::Size& input_size() const {return input_size_;}

    bool Infer(StyleTransfer::SourceConversion& RGBtoRGBfloatKrnl, ID3D11Texture2D* input_surface, ID3D11Texture2D* output_surface, const cv::Size& surface_size);
    void Infer(cv::Mat inputdata, cv::Mat& outputdata, const cv::Size& new_input_resolution);

  private:
    bool is_initialized_;
    cv::Size input_size_;
    int channels_;
    std::string input_name_;
    std::vector<std::string> output_names_;

    double time_elapsed_;
    size_t ncalls_;

    ov::InferRequest infer_request;
    ov::CompiledModel compiled_model;
    cl::Buffer                          _inputBuffer;

    cl::Buffer                          _outputBuffer;
    cl::Context							_oclCtx;

};
