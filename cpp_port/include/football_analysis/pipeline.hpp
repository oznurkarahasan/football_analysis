#pragma once

#include <memory>

#include "football_analysis/ball_trajectory_filler.hpp"
#include "football_analysis/interfaces.hpp"
#include "football_analysis/possession_assigner.hpp"
#include "football_analysis/speed_distance_estimator.hpp"

namespace fa {

class Pipeline {
public:
    Pipeline(
        PipelineConfig config,
        std::shared_ptr<IDetector> detector,
        std::shared_ptr<ITrackIdAssigner> trackAssigner,
        std::shared_ptr<ICameraMotionEstimator> cameraEstimator,
        std::shared_ptr<IHomographyTransformer> homography,
        std::shared_ptr<ITeamClassifier> teamClassifier);

    struct Output {
        TracksTimeline timeline;
        TeamBallControl teamBallControl;
    };

    Output run(const std::vector<Frame>& frames);

private:
    void addBasePositions(TracksTimeline& timeline) const;
    void addCameraAdjustedPositions(TracksTimeline& timeline, const std::vector<Point2f>& cameraMotion) const;
    void addWorldPositions(TracksTimeline& timeline) const;
    void assignTeams(TracksTimeline& timeline, const std::vector<Frame>& frames) const;

private:
    PipelineConfig config_{};
    std::shared_ptr<IDetector> detector_;
    std::shared_ptr<ITrackIdAssigner> trackAssigner_;
    std::shared_ptr<ICameraMotionEstimator> cameraEstimator_;
    std::shared_ptr<IHomographyTransformer> homography_;
    std::shared_ptr<ITeamClassifier> teamClassifier_;

    BallTrajectoryFiller ballFiller_;
};

}  // namespace fa
