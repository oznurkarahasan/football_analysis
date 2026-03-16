#include "football_analysis/possession_assigner.hpp"

#include <limits>

namespace fa {

PossessionAssigner::PossessionAssigner(float maxDistancePx)
    : maxDistancePx_(maxDistancePx) {}

int PossessionAssigner::assignBallToPlayer(const IdTracks& players, const BBox& ballBBox) const {
    const Point2f ballCenter = centerOfBBox(ballBBox);

    float minDistance = std::numeric_limits<float>::max();
    int assignedPlayer = -1;

    for (const auto& [playerId, playerState] : players) {
        const BBox& box = playerState.bbox;
        const Point2f leftFoot{box.x1, box.y2};
        const Point2f rightFoot{box.x2, box.y2};

        const float d = std::min(distance(leftFoot, ballCenter), distance(rightFoot, ballCenter));
        if (d < maxDistancePx_ && d < minDistance) {
            minDistance = d;
            assignedPlayer = playerId;
        }
    }

    return assignedPlayer;
}

TeamBallControl PossessionAssigner::assignTimeline(TracksTimeline& timeline) const {
    TeamBallControl teamControl;
    teamControl.reserve(timeline.size());

    int lastTeam = 0;

    for (auto& frameTracks : timeline) {
        for (auto& [_, playerState] : frameTracks.players) {
            playerState.hasBall = false;
        }

        const auto ballIt = frameTracks.ball.find(1);
        if (ballIt == frameTracks.ball.end()) {
            teamControl.push_back(lastTeam);
            continue;
        }

        const int playerId = assignBallToPlayer(frameTracks.players, ballIt->second.bbox);
        if (playerId != -1) {
            auto pIt = frameTracks.players.find(playerId);
            if (pIt != frameTracks.players.end()) {
                pIt->second.hasBall = true;
                if (pIt->second.teamId != 0) {
                    lastTeam = pIt->second.teamId;
                }
            }
        }

        teamControl.push_back(lastTeam);
    }

    return teamControl;
}

}  // namespace fa
