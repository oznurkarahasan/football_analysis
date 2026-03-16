#include "football_analysis/byte_track_lite.hpp"

#include <algorithm>

namespace fa {

ByteTrackLite::ByteTrackLite(float highThresh, float lowThresh, float iouHighThresh, float iouLowThresh, int maxLost)
    : highThresh_(highThresh),
      lowThresh_(lowThresh),
      iouHighThresh_(iouHighThresh),
      iouLowThresh_(iouLowThresh),
      maxLost_(maxLost) {}

float ByteTrackLite::iou(const cv::Rect& a, const cv::Rect& b) const {
    const int interX1 = std::max(a.x, b.x);
    const int interY1 = std::max(a.y, b.y);
    const int interX2 = std::min(a.x + a.width, b.x + b.width);
    const int interY2 = std::min(a.y + a.height, b.y + b.height);

    const int interW = std::max(0, interX2 - interX1);
    const int interH = std::max(0, interY2 - interY1);
    const float interArea = static_cast<float>(interW * interH);
    const float unionArea = static_cast<float>(a.area() + b.area()) - interArea;

    if (unionArea <= 1e-6f) {
        return 0.0f;
    }
    return interArea / unionArea;
}

std::vector<TrackableDetection> ByteTrackLite::update(const std::vector<TrackableDetection>& detections) {
    std::vector<TrackableDetection> out = detections;
    std::vector<int> highDetIdx;
    std::vector<int> lowDetIdx;

    for (int i = 0; i < static_cast<int>(out.size()); ++i) {
        if (out[static_cast<std::size_t>(i)].confidence >= highThresh_) {
            highDetIdx.push_back(i);
        } else if (out[static_cast<std::size_t>(i)].confidence >= lowThresh_) {
            lowDetIdx.push_back(i);
        }
    }

    std::vector<bool> detUsed(out.size(), false);
    std::vector<bool> trackMatched(tracks_.size(), false);

    auto matchStage = [&](const std::vector<int>& candidateDetIdx, float iouThresh) {
        for (std::size_t t = 0; t < tracks_.size(); ++t) {
            if (trackMatched[t]) {
                continue;
            }

            float bestIou = 0.0f;
            int bestDet = -1;
            for (int detIdx : candidateDetIdx) {
                if (detUsed[static_cast<std::size_t>(detIdx)]) {
                    continue;
                }

                const auto& det = out[static_cast<std::size_t>(detIdx)];
                if (det.classId != tracks_[t].classId) {
                    continue;
                }

                const float score = iou(tracks_[t].box, det.box);
                if (score > bestIou) {
                    bestIou = score;
                    bestDet = detIdx;
                }
            }

            if (bestDet != -1 && bestIou >= iouThresh) {
                auto& track = tracks_[t];
                auto& det = out[static_cast<std::size_t>(bestDet)];
                track.box = det.box;
                track.score = det.confidence;
                track.lost = 0;

                det.trackId = track.id;
                detUsed[static_cast<std::size_t>(bestDet)] = true;
                trackMatched[t] = true;
            }
        }
    };

    matchStage(highDetIdx, iouHighThresh_);
    matchStage(lowDetIdx, iouLowThresh_);

    for (int detIdx : highDetIdx) {
        if (detUsed[static_cast<std::size_t>(detIdx)]) {
            continue;
        }

        auto& det = out[static_cast<std::size_t>(detIdx)];
        Track track{};
        track.id = nextTrackId_++;
        track.classId = det.classId;
        track.box = det.box;
        track.score = det.confidence;
        track.lost = 0;
        tracks_.push_back(track);

        det.trackId = track.id;
        detUsed[static_cast<std::size_t>(detIdx)] = true;
    }

    if (trackMatched.size() < tracks_.size()) {
        trackMatched.resize(tracks_.size(), false);
    }

    for (std::size_t t = 0; t < tracks_.size(); ++t) {
        const bool matchedNow = (t < trackMatched.size()) ? trackMatched[t] : false;
        if (!matchedNow) {
            tracks_[t].lost += 1;
        }
    }

    tracks_.erase(
        std::remove_if(
            tracks_.begin(),
            tracks_.end(),
            [&](const Track& tr) { return tr.lost > maxLost_; }),
        tracks_.end());

    std::vector<TrackableDetection> tracked;
    tracked.reserve(out.size());
    for (const auto& det : out) {
        if (det.trackId != -1) {
            tracked.push_back(det);
        }
    }

    return tracked;
}

}  // namespace fa
