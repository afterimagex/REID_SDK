#include <ctime>
#include <cstdio>
#include <thread>
#include <string>

#include "dragon.h"
#include <vector>
#include "core/blob.h"
#include "utils/string.h"
#include <opencv2/opencv.hpp>
#include <iostream>

#include "tracker/detector.h"

#include "tracker/attribute.h"
#include "tracker/tracker.h"


using dragon::Workspace;
using dragon::Device;
using dragon::CreateWorkspace;
using dragon::MoveWorkspace;
using dragon::ReleaseWorkspace;
using dragon::CreateTensor;
using dragon::LoadCaffemodel;
using dragon::SetLogLevel;


Detector* init_rcnn(const std::string metatxt, const std::string caffemodel){
    Device dev("GPU", 0);
    Workspace* ws_model = CreateWorkspace("rcnn/model");
    LoadCaffemodel(caffemodel, ws_model);
    Workspace* ws_ins = CreateWorkspace("rcnn/ins");
    MoveWorkspace(ws_ins, ws_model);
    Detector* detector = new Detector(metatxt, dev, ws_ins);
    return detector;
}



Attribute* init_attr(const std::string metatxt, const std::string caffemodel){
    Device dev("GPU", 0);
    Workspace* ws_model = CreateWorkspace("attr/model");
    LoadCaffemodel(caffemodel, ws_model);
    Workspace* ws_ins = CreateWorkspace("attr/ins");
    MoveWorkspace(ws_ins, ws_model);
    Attribute* attribute = new Attribute(metatxt, dev, ws_ins);
    return attribute;
}


int main(){
    //ZF
    Detector* Rcnn = init_rcnn("../meta/RCNN/ZFP.metatxt", "../meta/RCNN/ZF_12_01_iter_410000.caffemodel");
    Rcnn->meta.rois = "Tensor_21";
    Rcnn->meta.bbox_pred = "Tensor_30";
    Rcnn->meta.cls_prob = "Tensor_29";
    //RESNET50
    //Detector* Rcnn = init_rcnn("../meta/RCNN/RESNET50.metatxt", "../meta/RCNN/RESNET50_1214_finetune_iter_77000.caffemodel");
    // Attribute* Attr = init_attr("../meta/ATTR/ATTR10.metatxt", "../meta/ATTR/RESNET10_iter_200000_20171204.caffemodel");
    Tracker tkTracker("../meta/REID/REID.metatxt", "../meta/REID/REID.caffemodel");

    const std::string video_file = "../meta/DEMO/video.avi";
    cv::VideoCapture capture(video_file);
    std::cout << video_file << std::endl;

    if (!capture.isOpened())
    {
        std::cout << "Can not open video file" << std::endl;
        return 0;
    }
    cv::Mat mat;
    clock_t  start_time, end_time;
    while (true){
         if (!capture.read(mat)){
             break;
         }
        start_time = clock();
        // detect
        Rcnn->GetImageBlob(mat);
        Rcnn->NetForward();
        // track & reid
        vector<TrackingBox> res = tkTracker.update_vis(mat, Rcnn->bndbox);

        end_time = clock();
        char buf[16] = {0};
        sprintf(buf, "%f", CLOCKS_PER_SEC/double(end_time-start_time));
        cv::resize(mat, mat, cv::Size(1200, 800));
        cv::putText(mat, buf, cv::Point(10, 10), 1, 1, (0, 0, 0), 1);
        //cv::imshow("demo", mat);
        int key = 0xff & cv::waitKey(30);

        if (key == 27){
            break;
        }
    }


}
