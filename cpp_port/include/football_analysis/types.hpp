#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace fa {

struct Point2f {
    float x{0.0f};
    float y{0.0f};
};

struct BBox {
    float x1{0.0f};
    float y1{0.0f};
    float x2{0.0f};
    float y2{0.0f};
};

inline Point2f centerOfBBox(const BBox& box) {
    return Point2f{(box.x1 + box.x2) * 0.5f, (box.y1 + box.y2) * 0.5f};
}

inline Point2f footPosition(const BBox& box) {
    return Point2f{(box.x1 + box.x2) * 0.5f, box.y2};
}

inline float distance(const Point2f& a, const Point2f& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

enum class ObjectClass {
    Player,
    Referee,
    Ball,
    Goalkeeper,
    Unknown
};

struct Detection {
    BBox bbox{};
    ObjectClass cls{ObjectClass::Unknown};
    float confidence{0.0f};
};

struct Frame {
    int index{0};
    int width{0};
    int height{0};
};

struct TrackState {
    BBox bbox{};
    Point2f position{};
    Point2f positionAdjusted{};
    std::optional<Point2f> positionWorld{};
    int teamId{0};
    bool hasBall{false};
    float speedKmh{0.0f};
    float distanceMeters{0.0f};
};

using IdTracks = std::unordered_map<int, TrackState>;

struct FrameTracks {
    IdTracks players;
    IdTracks referees;
    IdTracks ball;
};

using TracksTimeline = std::vector<FrameTracks>;

using TeamBallControl = std::vector<int>;

struct PipelineConfig {
    float ballPossessionDistanceThresholdPx{70.0f};
    int speedFrameWindow{5};
    float fps{24.0f};
};

}  // namespace fa
