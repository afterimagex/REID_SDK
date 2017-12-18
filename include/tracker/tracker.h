#ifndef DRAGON_TRACKER_H_
#define DRAGON_TRACKER_H_

// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------


#include <string>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <tracker/Hungarian.h>
#include <tracker/KalmanTracker.h>
#include "tracker/reidentity.h"
#include "utils/bbox.h"
#include "dragon.h"


typedef struct TrackingBox
{
    int id; //tracker id
    int cls; // obj class
    Rect_<float> box; // bnd
    vector<float> feature; //feature

}TrackingBox;


class Tracker {
public:
    Tracker(string model, string weight);
    vector<KalmanTracker> trackers; // kalmantracker to estimate new location
    vector<TrackingBox> update(cv::Mat img, vector<BoundBox> bnds); // update tracker's state
    vector<TrackingBox> update_vis(cv::Mat img, vector<BoundBox> bnds); // update and show result
private:
    cv::Scalar get_color_id(int idx);
    vector<cv::Scalar> id_color_list;
    double iou(Rect_<float> bnd0, Rect_<float> bnd1);
    double IOU_THRESHOLD = 0.3;
    Reidentity * reid_model;
    int frame_count;
};


#endif    // DRAGON_TRACKER_H_
