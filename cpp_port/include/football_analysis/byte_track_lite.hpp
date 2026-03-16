#pragma once

#include <vector>

#include <opencv2/core/types.hpp>

namespace fa {

struct TrackableDetection {
    int trackId{-1};
    int classId{-1};
    float confidence{0.0f};
    cv::Rect box{};
    bool hasBall{false};
};

class ByteTrackLite {
public:
    ByteTrackLite(float highThresh, float lowThresh, float iouHighThresh, float iouLowThresh, int maxLost);

    std::vector<TrackableDetection> update(const std::vector<TrackableDetection>& detections);

private:
    struct Track {
        int id{-1};
        int classId{-1};
        cv::Rect box{};
        float score{0.0f};
        int lost{0};
    };

    float iou(const cv::Rect& a, const cv::Rect& b) const;

    float highThresh_{0.45f};
    float lowThresh_{0.10f};
    float iouHighThresh_{0.30f};
    float iouLowThresh_{0.20f};
    int maxLost_{30};
    int nextTrackId_{1};
    std::vector<Track> tracks_;
};

}  // namespace fa
