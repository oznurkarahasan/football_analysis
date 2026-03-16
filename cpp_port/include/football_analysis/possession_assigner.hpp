#pragma once

#include "football_analysis/types.hpp"

namespace fa {

class PossessionAssigner {
public:
    explicit PossessionAssigner(float maxDistancePx);

    int assignBallToPlayer(const IdTracks& players, const BBox& ballBBox) const;
    TeamBallControl assignTimeline(TracksTimeline& timeline) const;

private:
    float maxDistancePx_{70.0f};
};

}  // namespace fa
