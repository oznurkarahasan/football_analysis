#include "football_analysis/pipeline.hpp"

namespace fa {

Pipeline::Pipeline(
    PipelineConfig config,
    std::shared_ptr<IDetector> detector,
    std::shared_ptr<ITrackIdAssigner> trackAssigner,
    std::shared_ptr<ICameraMotionEstimator> cameraEstimator,
    std::shared_ptr<IHomographyTransformer> homography,
    std::shared_ptr<ITeamClassifier> teamClassifier)
    : config_(config),
      detector_(std::move(detector)),
      trackAssigner_(std::move(trackAssigner)),
      cameraEstimator_(std::move(cameraEstimator)),
      homography_(std::move(homography)),
      teamClassifier_(std::move(teamClassifier)) {}

Pipeline::Output Pipeline::run(const std::vector<Frame>& frames) {
    Output output;
    if (frames.empty()) {
        return output;
    }

    const auto detectionsPerFrame = detector_->detectFrames(frames);
    output.timeline = trackAssigner_->assignTrackIds(detectionsPerFrame, frames);

    addBasePositions(output.timeline);

    const auto cameraMotion = cameraEstimator_->estimate(frames);
    addCameraAdjustedPositions(output.timeline, cameraMotion);

    addWorldPositions(output.timeline);
    ballFiller_.interpolateMissingBallBoxes(output.timeline);

    SpeedDistanceEstimator speedEstimator(config_.speedFrameWindow, config_.fps);
    speedEstimator.apply(output.timeline);

    assignTeams(output.timeline, frames);

    PossessionAssigner possession(config_.ballPossessionDistanceThresholdPx);
    output.teamBallControl = possession.assignTimeline(output.timeline);

    return output;
}

void Pipeline::addBasePositions(TracksTimeline& timeline) const {
    for (auto& frameTracks : timeline) {
        for (auto& [_, state] : frameTracks.players) {
            state.position = footPosition(state.bbox);
        }
        for (auto& [_, state] : frameTracks.referees) {
            state.position = footPosition(state.bbox);
        }
        for (auto& [_, state] : frameTracks.ball) {
            state.position = centerOfBBox(state.bbox);
        }
    }
}

void Pipeline::addCameraAdjustedPositions(TracksTimeline& timeline, const std::vector<Point2f>& cameraMotion) const {
    const std::size_t n = std::min(timeline.size(), cameraMotion.size());
    for (std::size_t i = 0; i < n; ++i) {
        const Point2f motion = cameraMotion[i];
        auto adjustContainer = [&](IdTracks& tracks) {
            for (auto& [_, state] : tracks) {
                state.positionAdjusted = Point2f{state.position.x - motion.x, state.position.y - motion.y};
            }
        };
        adjustContainer(timeline[i].players);
        adjustContainer(timeline[i].referees);
        adjustContainer(timeline[i].ball);
    }
}

void Pipeline::addWorldPositions(TracksTimeline& timeline) const {
    for (auto& frameTracks : timeline) {
        auto transformContainer = [&](IdTracks& tracks) {
            for (auto& [_, state] : tracks) {
                state.positionWorld = homography_->toWorld(state.positionAdjusted);
            }
        };
        transformContainer(frameTracks.players);
        transformContainer(frameTracks.referees);
        transformContainer(frameTracks.ball);
    }
}

void Pipeline::assignTeams(TracksTimeline& timeline, const std::vector<Frame>& frames) const {
    if (timeline.empty() || frames.empty()) {
        return;
    }

    teamClassifier_->fit(frames.front(), timeline.front().players);

    const std::size_t n = std::min(timeline.size(), frames.size());
    for (std::size_t i = 0; i < n; ++i) {
        for (auto& [playerId, state] : timeline[i].players) {
            state.teamId = teamClassifier_->predictTeam(frames[i], state, playerId);
        }
    }
}

}  // namespace fa
