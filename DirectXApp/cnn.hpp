// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <d3d11.h>
#include <windows.h>

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>

using namespace InferenceEngine;

class Cnn {
  public:
    Cnn():is_initialized_(false), channels_(0), time_elapsed_(0), ncalls_(0) {}

    void Init(const std::string &model_path,  ID3D11Device*& d3d_device,
              const cv::Size &new_input_resolution = cv::Size());

    InferenceEngine::BlobMap Infer(const cv::Mat &frame);

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
    InferRequest infer_request_;
    std::vector<std::string> output_names_;

    double time_elapsed_;
    size_t ncalls_;

    RemoteContext::Ptr remote_context_;
    InputInfo::Ptr input_info_;
};
