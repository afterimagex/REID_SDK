// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_BENCHMARKS_ATTRIBUTE_H_
#define DRAGON_BENCHMARKS_ATTRIBUTE_H_

#include <string>
#include <opencv2/opencv.hpp>

#include "core/blob.h"
#include "core/model.h"

#include <map>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
    
class Attribute : public Model {
 public:
    Attribute(const std::string& graph_file,
                 const dragon::Device& device,
                 dragon::Workspace* ws)
         : Model(graph_file, device, ws) {
        scale = 224;
        label_num = 86;
        this->Init();
        this->InitLabelDict();
    }

    void CreateVariables() final {
        CreateVariable("data");
    }

    void FeedVariables() final {
        FeedVariable("data", data);
    }

    void InitLabelDict();

    void GetImageBlob(const cv::Mat& mat);
    void GetImageBlob(const vector<cv::Mat>& images);

    void NetForward();

    map<string, string> LabelConvert(std::vector<int>& label);
    vector<map<string, string>> LabelConvert(std::vector<std::vector<int>>& label);

    void RunDemo(const std::string& filename, bool vis=true);

    vector<vector<int>> GetOnes() { return ones; }

    int vector_max_index(std::vector<int>& vec);

 protected:
    int scale;
    int label_num;
    Blob<float> data;
    Blob<float> output;
    vector<vector<int>> ones;

    map<string, vector<string>> str_map;
    map<string, vector<int>> int_map;
    map<string, vector<int>> car_int_map;
    map<string, vector<int>> people_int_map;
    map<string, vector<int>> cp_int_map;

    struct metatxt{
        std::string output = "Tensor_79";
    } meta;
};

#endif    // DRAGON_BENCHMARKS_ATTRIBUTE_H_
