#pragma once

#include "football_analysis/types.hpp"

namespace fa {

class BallTrajectoryFiller {
public:
    void interpolateMissingBallBoxes(TracksTimeline& timeline) const;
};

}  // namespace fa
