// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_UTILS_CV_H_
#define DRAGON_UTILS_CV_H_

#include <vector>
#include <cstdint>
#include <opencv2/opencv.hpp>

typedef int64_t TIndex;

cv::Mat ReadImageToCVMat(const std::string& filename,
                         const int height = 0,
                         const int width = 0,
                         const bool is_color = true);

template <typename T>
void ConvertMats(const std::vector<cv::Mat*>& mats,
                 const std::vector<TIndex>& shape,
                 T* data);

#endif    // DRAGON_UTILS_CV_H_