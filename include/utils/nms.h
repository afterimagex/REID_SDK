// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_UTILS_NMS_H_
#define DRAGON_UTILS_NMS_H_

#include <unordered_map>
#include <unordered_set>

#include "utils/bbox.h"

inline void ApplyNMS(const float nms_thresh,
                     const std::vector<BoundingBox>& boxes,
                     std::vector<int>& keep) {
    std::unordered_set<int> suppressed;
    std::unordered_map<int, float> area;
    float ix1, iy1, ix2, iy2;
    float xx1, yy1, xx2, yy2;
    float w, h;
    float inter, ovr;
    //  compute area
    for (int i = 0; i < boxes.size(); i++) {
        area[i] = (boxes[i].x2() - boxes[i].x1() + 1) *
                  (boxes[i].y2() - boxes[i].y1() + 1);
        if (area[i] < 0) suppressed.insert(i);
    }
    //  apply nms
    for (int i = 0; i < boxes.size(); i++) {
        if (suppressed.count(i)) continue;
        keep.push_back(i);
        ix1 = boxes[i].x1();
        iy1 = boxes[i].y1();
        ix2 = boxes[i].x2();
        iy2 = boxes[i].y2();
        for (int j = i + 1; j < boxes.size(); j++) {
            if (suppressed.count(j)) continue;
            xx1 = std::max(ix1, (float)boxes[j].x1());
            yy1 = std::max(iy1, (float)boxes[j].y1());
            xx2 = std::min(ix2, (float)boxes[j].x2());
            yy2 = std::min(iy2, (float)boxes[j].y2());
            w = std::max(0.f, xx2 - xx1 + 1);
            h = std::max(0.f, yy2 - yy1 + 1);
            inter = w * h;
            ovr = inter / (area[i] + area[j] - inter);
            if (ovr > nms_thresh) suppressed.insert(j);
        }
    }
}

#endif    // DRAGON_UTILS_NMS_H_
