#include "football_analysis/ball_trajectory_filler.hpp"

namespace fa {

namespace {

bool hasBall(const FrameTracks& frameTracks) {
    return frameTracks.ball.find(1) != frameTracks.ball.end();
}

}  // namespace

void BallTrajectoryFiller::interpolateMissingBallBoxes(TracksTimeline& timeline) const {
    if (timeline.empty()) {
        return;
    }

    int firstValid = -1;
    for (std::size_t i = 0; i < timeline.size(); ++i) {
        if (hasBall(timeline[i])) {
            firstValid = static_cast<int>(i);
            break;
        }
    }

    if (firstValid < 0) {
        return;
    }

    for (int i = 0; i < firstValid; ++i) {
        timeline[static_cast<std::size_t>(i)].ball[1] = timeline[static_cast<std::size_t>(firstValid)].ball.at(1);
    }

    int prevValid = firstValid;
    for (std::size_t i = static_cast<std::size_t>(firstValid + 1); i < timeline.size(); ++i) {
        if (hasBall(timeline[i])) {
            const int nextValid = static_cast<int>(i);
            const int gap = nextValid - prevValid;

            if (gap > 1) {
                const auto& b0 = timeline[static_cast<std::size_t>(prevValid)].ball.at(1).bbox;
                const auto& b1 = timeline[static_cast<std::size_t>(nextValid)].ball.at(1).bbox;

                for (int k = 1; k < gap; ++k) {
                    const float t = static_cast<float>(k) / static_cast<float>(gap);
                    TrackState state{};
                    state.bbox.x1 = b0.x1 + (b1.x1 - b0.x1) * t;
                    state.bbox.y1 = b0.y1 + (b1.y1 - b0.y1) * t;
                    state.bbox.x2 = b0.x2 + (b1.x2 - b0.x2) * t;
                    state.bbox.y2 = b0.y2 + (b1.y2 - b0.y2) * t;
                    timeline[static_cast<std::size_t>(prevValid + k)].ball[1] = state;
                }
            }

            prevValid = nextValid;
        }
    }

    for (std::size_t i = static_cast<std::size_t>(prevValid + 1); i < timeline.size(); ++i) {
        timeline[i].ball[1] = timeline[static_cast<std::size_t>(prevValid)].ball.at(1);
    }
}

}  // namespace fa
