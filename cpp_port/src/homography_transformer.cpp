#include "football_analysis/homography_transformer.hpp"

#include <vector>

#include <opencv2/imgproc.hpp>  // getPerspectiveTransform, pointPolygonTest, perspectiveTransform

namespace fa {

HomographyTransformer::HomographyTransformer() {
    // Python pixel_vertices:
    //   [[110, 1035], [265, 275], [910, 260], [1640, 915]]
    const std::array<cv::Point2f, 4> pixelVertices = {
        cv::Point2f(110.0f, 1035.0f),
        cv::Point2f(265.0f, 275.0f),
        cv::Point2f(910.0f, 260.0f),
        cv::Point2f(1640.0f, 915.0f),
    };

    // Python target_vertices (metre):
    //   court_width=68, court_length=23.32
    //   [[0, 68], [0, 0], [23.32, 0], [23.32, 68]]
    const std::array<cv::Point2f, 4> targetVertices = {
        cv::Point2f(0.0f, 68.0f),
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(23.32f, 0.0f),
        cv::Point2f(23.32f, 68.0f),
    };

    buildMatrix(pixelVertices, targetVertices);
}

HomographyTransformer::HomographyTransformer(
    const std::array<cv::Point2f, 4>& pixelVertices,
    const std::array<cv::Point2f, 4>& targetVertices) {
    buildMatrix(pixelVertices, targetVertices);
}

void HomographyTransformer::buildMatrix(
    const std::array<cv::Point2f, 4>& pixelVertices,
    const std::array<cv::Point2f, 4>& targetVertices) {
    pixelVertices_ = pixelVertices;

    const std::vector<cv::Point2f> src(pixelVertices.begin(), pixelVertices.end());
    const std::vector<cv::Point2f> dst(targetVertices.begin(), targetVertices.end());
    homographyMatrix_ = cv::getPerspectiveTransform(src, dst);
}

std::optional<Point2f> HomographyTransformer::toWorld(const Point2f& adjustedPoint) {
    const cv::Point2f cvPt(adjustedPoint.x, adjustedPoint.y);

    // Python: cv2.pointPolygonTest ile nokta polygon icinde mi kontrol et
    // >= 0 → icinde veya kenarda, < 0 → disinda
    const std::vector<cv::Point2f> polygon(pixelVertices_.begin(), pixelVertices_.end());
    if (cv::pointPolygonTest(polygon, cvPt, false) < 0.0) {
        return std::nullopt;
    }

    // Python: reshaped_point.reshape(-1,1,2) → cv2.perspectiveTransform karsiligi
    const std::vector<cv::Point2f> src = {cvPt};
    std::vector<cv::Point2f> dst;
    cv::perspectiveTransform(src, dst, homographyMatrix_);

    return Point2f{dst[0].x, dst[0].y};
}

}  // namespace fa
