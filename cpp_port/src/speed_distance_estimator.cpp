#include "football_analysis/speed_distance_estimator.hpp"

#include <unordered_map>

namespace fa {

SpeedDistanceEstimator::SpeedDistanceEstimator(int frameWindow, float fps)
    : frameWindow_(frameWindow), fps_(fps) {}

void SpeedDistanceEstimator::apply(TracksTimeline& timeline) const {
    if (timeline.empty() || frameWindow_ <= 0 || fps_ <= 0.0f) {
        return;
    }

    std::unordered_map<int, float> totalDistanceByPlayer;
    const std::size_t n = timeline.size();

    for (std::size_t start = 0; start < n; start += static_cast<std::size_t>(frameWindow_)) {
        const std::size_t end = std::min(start + static_cast<std::size_t>(frameWindow_), n - 1);
        const float dt = static_cast<float>(end - start) / fps_;
        if (dt <= 0.0f) {
            continue;
        }

        for (const auto& [playerId, startState] : timeline[start].players) {
            auto endIt = timeline[end].players.find(playerId);
            if (endIt == timeline[end].players.end()) {
                continue;
            }

            if (!startState.positionWorld.has_value() || !endIt->second.positionWorld.has_value()) {
                continue;
            }

            const float covered = distance(startState.positionWorld.value(), endIt->second.positionWorld.value());
            totalDistanceByPlayer[playerId] += covered;
            const float speedKmh = (covered / dt) * 3.6f;

            for (std::size_t i = start; i < end; ++i) {
                auto inWindow = timeline[i].players.find(playerId);
                if (inWindow == timeline[i].players.end()) {
                    continue;
                }
                inWindow->second.speedKmh = speedKmh;
                inWindow->second.distanceMeters = totalDistanceByPlayer[playerId];
            }
        }
    }
}

}  // namespace fa
