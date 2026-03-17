#include "football_analysis/camera_motion_estimator.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace fa {

CameraMotionEstimator::CameraMotionEstimator(
    int leftStripWidth,
    int rightStripStart,
    int rightStripEnd,
    float minMovementPx)
    : leftStripWidth_(leftStripWidth),
      rightStripStart_(rightStripStart),
      rightStripEnd_(rightStripEnd),
      minMovementPx_(minMovementPx) {}

std::vector<cv::Point2f> CameraMotionEstimator::detectFeatures(const cv::Mat& gray) const {
    // Python: cv2.goodFeaturesToTrack ile kenar seritlerinde feature tespit
    // mask: sol serit [0, leftStripWidth] ve sag serit [rightStripStart, rightStripEnd]
    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);

    const int w = gray.cols;
    const int h = gray.rows;

    // Sol serit
    const int leftEnd = std::min(leftStripWidth_, w);
    cv::rectangle(mask, cv::Point(0, 0), cv::Point(leftEnd, h), cv::Scalar(255), cv::FILLED);

    // Sag serit
    const int rStart = std::min(rightStripStart_, w);
    const int rEnd = std::min(rightStripEnd_, w);
    if (rStart < rEnd) {
        cv::rectangle(mask, cv::Point(rStart, 0), cv::Point(rEnd, h), cv::Scalar(255), cv::FILLED);
    }

    // Python parametreleri: maxCorners=100, qualityLevel=0.3, minDistance=3, blockSize=7
    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(gray, points, 100, 0.3, 3.0, mask, 7);
    return points;
}

std::vector<cv::Point2f> CameraMotionEstimator::estimate(const std::vector<cv::Mat>& frames) {
    const std::size_t n = frames.size();
    std::vector<cv::Point2f> motion(n, cv::Point2f(0.0f, 0.0f));

    if (n < 2) {
        return motion;
    }

    // Python Lucas-Kanade parametreleri:
    //   winSize=(15,15), maxLevel=2
    //   criteria=(COUNT|EPS, 10, 0.03)
    const cv::Size winSize(15, 15);
    const int maxLevel = 2;
    const cv::TermCriteria termCriteria(
        cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03);

    cv::Mat prevGray;
    cv::cvtColor(frames[0], prevGray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> prevPoints = detectFeatures(prevGray);

    for (std::size_t i = 1; i < n; ++i) {
        cv::Mat currGray;
        cv::cvtColor(frames[i], currGray, cv::COLOR_BGR2GRAY);

        // Feature kalmadiysa yeniden tespit et
        if (prevPoints.empty()) {
            prevPoints = detectFeatures(prevGray);
        }

        if (prevPoints.empty()) {
            prevGray = currGray;
            continue;
        }

        std::vector<cv::Point2f> currPoints;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(
            prevGray, currGray,
            prevPoints, currPoints,
            status, err,
            winSize, maxLevel,
            termCriteria);

        // Python mantigi: en buyuk deplasmanli feature'i bul
        // Bu, kamera hareketinin en iyi tahminini verir
        float maxDist = 0.0f;
        cv::Point2f bestMotion(0.0f, 0.0f);

        for (std::size_t j = 0; j < status.size(); ++j) {
            if (status[j] == 0) {
                continue;
            }
            const float dx = currPoints[j].x - prevPoints[j].x;
            const float dy = currPoints[j].y - prevPoints[j].y;
            const float dist = std::sqrt(dx * dx + dy * dy);
            if (dist > maxDist) {
                maxDist = dist;
                bestMotion = cv::Point2f(dx, dy);
            }
        }

        if (maxDist > minMovementPx_) {
            // Kamera hareket etti: deplasmayi kaydet ve feature'lari yenile
            motion[i] = bestMotion;
            prevPoints = detectFeatures(currGray);
        } else {
            // Kamera sabit: sadece takip edilen noktalar kullanilmaya devam eder
            motion[i] = cv::Point2f(0.0f, 0.0f);
            std::vector<cv::Point2f> validPoints;
            validPoints.reserve(status.size());
            for (std::size_t j = 0; j < status.size(); ++j) {
                if (status[j]) {
                    validPoints.push_back(currPoints[j]);
                }
            }
            prevPoints = std::move(validPoints);
        }

        prevGray = currGray;
    }

    return motion;
}

}  // namespace fa
