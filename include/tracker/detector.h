// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_DETECTOR_H_
#define DRAGON_DETECTOR_H_

#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include "core/blob.h"
#include "utils/bbox.h"
#include "core/model.h"


class Detector : public Model {
public:
    Detector(const std::string& graph_file,
             const dragon::Device& device,
             dragon::Workspace* ws) : Model(graph_file, device, ws) {
        scale_ = 600;
        max_size_ = 1000;
        nms_ = 0.3f;
        thresh_ = 0.5f;
        this->Init();
    }

    void CreateVariables() final {
        CreateVariable("data");
        CreateVariable("im_info");
    }

    void FeedVariables() final {
        FeedVariable("data", data);
        FeedVariable("im_info", im_info);
    }

    void GetImageBlob(const cv::Mat& mat);

    void NetForward();

    void DrawAndShow(const std::string& filename, bool vis);
    void DrawAndShow(cv::Mat src, cv::Mat& dst, bool vis);
    void VideoDemo(const std::string& filename);

    inline void set_scale(int scale) { scale_ = scale; }
    inline void set_max_size(int max_size) { scale_ = max_size; }
    inline void set_nms(float nms) { nms_ = nms; }
    inline void set_thresh(float thresh) { thresh_ = thresh; }

    std::vector<BoundBox> bndbox;

    struct metatxt{
	// ZF
        std::string rois = "Tensor_21";
        std::string bbox_pred = "Tensor_30";
        std::string cls_prob = "Tensor_29";
	// RESNET
        //std::string rois = "Tensor_190";
        //std::string bbox_pred = "Tensor_236";
        //std::string cls_prob = "Tensor_235";
    } meta;

protected:
    int scale_, max_size_;
    float nms_, thresh_;
    Blob<float> data, im_info;
    Blob<float> rois, cls_prob, bbox_pred;
    float im_scale;
    int mat_rows;
    int mat_cols;
};

#endif    // DRAGON_DETECTOR_H_
