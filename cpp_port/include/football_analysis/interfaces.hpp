#pragma once

#include "football_analysis/types.hpp"

namespace fa {

class IDetector {
public:
    virtual ~IDetector() = default;
    virtual std::vector<std::vector<Detection>> detectFrames(const std::vector<Frame>& frames) = 0;
};

class ITrackIdAssigner {
public:
    virtual ~ITrackIdAssigner() = default;
    virtual TracksTimeline assignTrackIds(
        const std::vector<std::vector<Detection>>& detectionsPerFrame,
        const std::vector<Frame>& frames) = 0;
};

class ICameraMotionEstimator {
public:
    virtual ~ICameraMotionEstimator() = default;
    virtual std::vector<Point2f> estimate(const std::vector<Frame>& frames) = 0;
};

class IHomographyTransformer {
public:
    virtual ~IHomographyTransformer() = default;
    virtual std::optional<Point2f> toWorld(const Point2f& adjustedPoint) = 0;
};

class ITeamClassifier {
public:
    virtual ~ITeamClassifier() = default;
    virtual void fit(const Frame& frame, const IdTracks& playersInFirstFrame) = 0;
    virtual int predictTeam(const Frame& frame, const TrackState& playerState, int playerId) = 0;
};

}  // namespace fa
