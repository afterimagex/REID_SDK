// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_REIDENTITY_H_
#define DRAGON_REIDENTITY_H_

#include <string>
#include <opencv2/opencv.hpp>
#include <vector>

#include "core/blob.h"
#include "core/model.h"
#include "tracker/reidentity.h"

using std::vector;

class Reidentity : public Model {
public:
    Reidentity(const std::string& graph_file,
               const dragon::Device& device,
               dragon::Workspace* ws) : Model(graph_file, device, ws) {
        resize_w = 64;
        resize_h = 128;
        this->Init();
    }

    void CreateVariables() final {
        CreateVariable("data");
    }

    void FeedVariables() final {
        FeedVariable("data", data);
    }

    vector<vector<float >> predict(vector<cv::Mat> imgs); // 提取图片特征向量



//    void GetImageBlob(vector<cv::Mat>& images);
//
//    void NetForward();

//    vector<vector<float>> feature;

    float feature_dis(vector<float> feat1, vector<float> feat2); //计算两个特征向量之间的相似度

    vector<vector<float>> feature_dis(vector<vector<float>> feats); //计算特征向量之间的距离

    vector<vector<float>> img_dis(vector<cv::Mat> images); //计算图片之间的距离
    
    float img_dis(cv::Mat image1, cv::Mat image2);  // 计算两张图片的距离

    struct metatxt{
        std::string feature = "Tensor_99";
    } meta;

protected:
    int resize_w;
    int resize_h;
    Blob<float> data;
    Blob<float> feat;

};

#endif    // DRAGON_REIDENTITY_H_
