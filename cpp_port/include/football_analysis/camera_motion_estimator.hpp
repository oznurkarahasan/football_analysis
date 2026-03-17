#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace fa {

// Python camera_movement_estimator.py portudur.
// Lucas-Kanade optical flow ile frame'ler arasi kamera hareketini tahmin eder.
// Sadece frame'in sol (0..leftStripWidth) ve sag (rightStripStart..rightStripEnd)
// kenar seritlerindeki featurelar kullanilir; merkezdeki oyuncu/top karismasin diye.
class CameraMotionEstimator {
public:
    // Parametreler Python implementasyonuyla ayni:
    //   leftStripWidth   : sol kenar seridi genisligi (piksel), Python'da 20
    //   rightStripStart  : sag kenar seridinin baslangici, Python'da 900
    //   rightStripEnd    : sag kenar seridinin sonu, Python'da 1050
    //   minMovementPx    : hareket esigi; altindaysa kamera hareketi yok sayilir, Python'da 5
    explicit CameraMotionEstimator(
        int leftStripWidth = 20,
        int rightStripStart = 900,
        int rightStripEnd = 1050,
        float minMovementPx = 5.0f);

    // Verilen frame dizisi icin her frame'in kamera deplasmanini (dx, dy) dondurur.
    // motion[0] her zaman (0,0)'dir (onceki frame yok).
    std::vector<cv::Point2f> estimate(const std::vector<cv::Mat>& frames);

private:
    // Verilen gritonlu frame'de kenar seritlerinde feature noktalari tespit eder.
    std::vector<cv::Point2f> detectFeatures(const cv::Mat& gray) const;

    int leftStripWidth_;
    int rightStripStart_;
    int rightStripEnd_;
    float minMovementPx_;
};

}  // namespace fa
