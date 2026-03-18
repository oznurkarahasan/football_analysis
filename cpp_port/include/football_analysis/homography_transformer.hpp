#pragma once

#include <array>

#include <opencv2/core.hpp>

#include "football_analysis/interfaces.hpp"

namespace fa {

// Python view_transformer/view_transformer.py portudur.
// cv::getPerspectiveTransform ile pixel koordinatlarini gercek dunya
// koordinatlarina (metre) donusturur.
//
// Varsayilan degerleri Python ile birebir aynidir:
//   pixel_vertices : frame uzerindeki 4 referans noktasi
//   target_vertices: karsilik gelen saha koordinatlari (metre)
class HomographyTransformer : public IHomographyTransformer {
public:
    // Python pixel_vertices / target_vertices ile ayni varsayilan degerler.
    // Farkli bir kamera acisi/sahasi icin dis parametreler gecirilebilir.
    HomographyTransformer();

    HomographyTransformer(
        const std::array<cv::Point2f, 4>& pixelVertices,
        const std::array<cv::Point2f, 4>& targetVertices);

    // Kamera-kompanze edilmis pixel konumunu saha koordinatina (metre) donusturur.
    // Nokta pixel_vertices polygon'unun disindaysa std::nullopt doner.
    std::optional<Point2f> toWorld(const Point2f& adjustedPoint) override;

private:
    void buildMatrix(
        const std::array<cv::Point2f, 4>& pixelVertices,
        const std::array<cv::Point2f, 4>& targetVertices);

    cv::Mat homographyMatrix_;
    std::array<cv::Point2f, 4> pixelVertices_;
};

}  // namespace fa
