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

using namespace InferenceEngine;

class Cnn {
  public:
    Cnn():is_initialized_(false), channels_(0), time_elapsed_(0), ncalls_(0) {}

    void Init(const std::string &model_path,  ID3D11Device*& d3d_device, ID3D11Buffer* input_surface, ID3D11Buffer* output_surface,
              const cv::Size &new_input_resolution = cv::Size());

    void Init(const std::string& model_path, ID3D11Device*& d3d_device, ID3D11Texture2D* input_surface, ID3D11Texture2D* output_surface,
        const cv::Size& new_input_resolution = cv::Size());

    void Init(const std::string& model_path, ID3D11Device*& d3d_device, cv::Mat input_data);

    bool is_initialized() const {return is_initialized_;}

    size_t ncalls() const {return ncalls_;}
    double time_elapsed() const {return time_elapsed_;}

    const cv::Size& input_size() const {return input_size_;}

    void Infer(ID3D11Texture2D* surface);

  private:
    bool is_initialized_;
    cv::Size input_size_;
    int channels_;
    std::string input_name_;
    std::vector<std::string> output_names_;

    double time_elapsed_;
    size_t ncalls_;

    ov::InferRequest infer_request;
    ov::intel_gpu::ocl::D3DContext* remote_context;

};
