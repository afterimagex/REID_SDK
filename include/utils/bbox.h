// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_UTILS_BBOX_H_
#define DRAGON_UTILS_BBOX_H_

#include <cmath>
#include <sstream>
#include <vector>
#include <algorithm>

class BoundingBox {
 public:
    BoundingBox(const std::vector<float>& anchor,
                const std::vector<float>& delta, float conf) {
        float width = anchor[2] - anchor[0];
        float height = anchor[3] - anchor[1];
        float ctr_x = anchor[0] + 0.5f * width;
        float ctr_y = anchor[1] + 0.5f * height;
        float pred_ctr_x = delta[0] * width + ctr_x;
        float pred_ctr_y = delta[1] * height + ctr_y;
        float pred_w = exp(delta[2]) * width;
        float pred_h = exp(delta[3]) * height;
        box_.push_back(int(pred_ctr_x - 0.5f * pred_w));
        box_.push_back(int(pred_ctr_y - 0.5f * pred_h));
        box_.push_back(int(pred_ctr_x + 0.5f * pred_w));
        box_.push_back(int(pred_ctr_y + 0.5f * pred_h));
        conf_ = conf;
    }

    BoundingBox(const int x1, const int y1,
                const int x2, const int y2,
                const float conf) {
        box_.push_back(x1);
        box_.push_back(y1);
        box_.push_back(x2);
        box_.push_back(y2);
        conf_ = conf;
    }

    void Crop(const int h, const int w) {
      box_[0] = std::max(0, std::min(box_[0], w - 1));
      box_[1] = std::max(0, std::min(box_[1], h - 1));
      box_[2] = std::max(0, std::min(box_[2], w - 1));
      box_[3] = std::max(0, std::min(box_[3], h - 1));
    }

    std::string debug_string() {
        std::stringstream ss;
        ss << "[";
        for (int i = 0; i < box_.size() - 1; i++) ss << box_[i] << ", ";
        ss << box_[box_.size() - 1] << "]";
        return ss.str();
    }

    const float conf() const { return conf_; }
    const std::vector<int> box() const { return box_; }

    const int x1() const { return box_[0]; }
    const int y1() const { return box_[1]; }
    const int x2() const { return box_[2]; }
    const int y2() const { return box_[3]; }

    bool operator < (const BoundingBox & a) const { return conf_ > a.conf_;}

 protected:
    std::vector<int> box_;
    float conf_;
};

class BoundBox {
public:

    BoundBox(const int x1, const int y1,
             const int x2, const int y2,
             const int cls,
             const float conf) {
        box_.push_back(x1);
        box_.push_back(y1);
        box_.push_back(x2);
        box_.push_back(y2);
        cls_ = cls;
        conf_ = conf;
    }

    std::string debug_string() {
        std::stringstream ss;
        ss << "[";
        for (int i = 0; i < box_.size() - 1; i++) ss << box_[i] << ", ";
        ss << box_[box_.size() - 1] << "]";
        return ss.str();
    }

    const float conf() const { return conf_; }

    const std::vector<int> box() const { return box_; }

    const int x1() const { return box_[0]; }
    const int y1() const { return box_[1]; }
    const int x2() const { return box_[2]; }
    const int y2() const { return box_[3]; }
    const int w() const { return box_[2] - box_[0]; }
    const int h() const { return box_[3] - box_[1]; }
    const int cls() const { return cls_; }

    bool operator < (const BoundBox & a) const { return conf_ > a.conf_;}

protected:
    std::vector<int> box_;
    int cls_;
    float conf_;
};

class NormalizedBBox {
 public:
    NormalizedBBox(const std::vector<float>& priorbox,
                   const std::vector<float>& delta, float conf) {
        float width = priorbox[2] - priorbox[0];
        float height = priorbox[3] - priorbox[1];
        float ctr_x = priorbox[0] + 0.5f * width;
        float ctr_y = priorbox[1] + 0.5f * height;
        //  scale delta by (0.1, 0.1, 0.2, 0.2)
        float pred_ctr_x = delta[0] * 0.1f * width + ctr_x;
        float pred_ctr_y = delta[1] * 0.1f * height + ctr_y;
        float pred_w = exp(delta[2] * 0.2f) * width;
        float pred_h = exp(delta[3] * 0.2f) * height;
        box_ = std::vector<float>({ pred_ctr_x - 0.5f * pred_w,
                                    pred_ctr_y - 0.5f * pred_h,
                                    pred_ctr_x + 0.5f * pred_w,
                                    pred_ctr_y + 0.5f * pred_h });
        this->conf_ = conf;
    }

    BoundingBox ToBoundingBox(const int h, const int w) {
        int x1 = std::max(0, std::min(int(box_[0] * w), w - 1));
        int y1 = std::max(0, std::min(int(box_[1] * h), h - 1));
        int x2 = std::max(0, std::min(int(box_[2] * w), w - 1));
        int y2 = std::max(0, std::min(int(box_[3] * h), h - 1));
        return BoundingBox(x1, y1, x2, y2, conf_);
    }

    const float conf() const { return conf_; }
    const std::vector<float> box() const { return box_; }

    const float x1() const { return box_[0]; }
    const float y1() const { return box_[1]; }
    const float x2() const { return box_[2]; }
    const float y2() const { return box_[3]; }

    bool operator < (const NormalizedBBox& a) const { return conf_ > a.conf_; }
    std::vector<float> box_;
    float conf_;
};

#endif    // DRAGON_UTILS_BBOX_H_
