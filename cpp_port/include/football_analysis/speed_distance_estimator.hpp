#pragma once

#include "football_analysis/types.hpp"

namespace fa {

class SpeedDistanceEstimator {
public:
    SpeedDistanceEstimator(int frameWindow, float fps);

    void apply(TracksTimeline& timeline) const;

private:
    int frameWindow_{5};
    float fps_{24.0f};
};

}  // namespace fa
