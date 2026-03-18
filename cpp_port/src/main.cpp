#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "football_analysis/byte_track_lite.hpp"
#include "football_analysis/camera_motion_estimator.hpp"
#include "football_analysis/homography_transformer.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
};

using HumanDetection = fa::TrackableDetection;

struct LetterboxInfo {
    float ratio{1.0f};
    float dw{0.0f};
    float dh{0.0f};
};

struct ClassMapping {
    int ballId = 0;
    int goalkeeperId = 1;
    int playerId = 2;
    int refereeId = 3;
};

bool isHumanClass(int classId, const ClassMapping& mapping) {
    return classId == mapping.playerId || classId == mapping.goalkeeperId || classId == mapping.refereeId;
}

std::string classLabel(int classId, const ClassMapping& mapping) {
    if (classId == mapping.ballId) {
        return "ball";
    }
    if (classId == mapping.playerId) {
        return "player";
    }
    if (classId == mapping.goalkeeperId) {
        return "goalkeeper";
    }
    if (classId == mapping.refereeId) {
        return "referee";
    }
    return "other";
}

cv::Scalar classColor(int classId, const ClassMapping& mapping) {
    if (classId == mapping.ballId) {
        return cv::Scalar(0, 255, 0);
    }
    if (classId == mapping.playerId) {
        return cv::Scalar(0, 0, 255);
    }
    if (classId == mapping.goalkeeperId) {
        return cv::Scalar(255, 0, 0);
    }
    if (classId == mapping.refereeId) {
        return cv::Scalar(0, 255, 255);
    }
    return cv::Scalar(200, 200, 200);
}

void drawBallTriangle(cv::Mat& frame, const cv::Rect& box, const cv::Scalar& color) {
    const int x = box.x + (box.width / 2);
    const int y = box.y;
    std::vector<cv::Point> triangle = {
        cv::Point(x, y),
        cv::Point(x - 10, y - 20),
        cv::Point(x + 10, y - 20)};
    const std::vector<std::vector<cv::Point>> contours = {triangle};
    cv::drawContours(frame, contours, 0, color, cv::FILLED);
    cv::drawContours(frame, contours, 0, cv::Scalar(0, 0, 0), 2);
}

void drawPlayerEllipse(cv::Mat& frame, const cv::Rect& box, const cv::Scalar& color, int trackId) {
    const int y2 = box.y + box.height;
    const int xCenter = box.x + box.width / 2;
    const int width = box.width;

    cv::ellipse(
        frame,
        cv::Point(xCenter, y2),
        cv::Size(std::max(1, width), std::max(1, static_cast<int>(0.35f * static_cast<float>(width)))),
        0.0,
        -45,
        235,
        color,
        2,
        cv::LINE_4);

    const int rectW = 40;
    const int rectH = 20;
    const int x1 = xCenter - rectW / 2;
    const int x2 = xCenter + rectW / 2;
    const int y1 = y2 + 5;
    const int y2r = y1 + rectH;

    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2r), color, cv::FILLED);
    cv::putText(frame, std::to_string(trackId), cv::Point(x1 + 8, y1 + 15), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 2);
}

cv::Point2f rectCenter(const cv::Rect& box) {
    return cv::Point2f(
        static_cast<float>(box.x + box.width / 2),
        static_cast<float>(box.y + box.height / 2));
}

float pointDistance(const cv::Point2f& a, const cv::Point2f& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

int assignBallToClosestPlayer(const std::vector<HumanDetection>& humans, const cv::Rect& ballBox, int playerClassId, float maxDistancePx) {
    const cv::Point2f ballCenter(
        static_cast<float>(ballBox.x + ballBox.width / 2),
        static_cast<float>(ballBox.y + ballBox.height / 2));

    float minDistance = 1e9f;
    int assignedTrack = -1;
    for (const auto& human : humans) {
        if (human.classId != playerClassId) {
            continue;
        }

        const cv::Point2f leftFoot(static_cast<float>(human.box.x), static_cast<float>(human.box.y + human.box.height));
        const cv::Point2f rightFoot(static_cast<float>(human.box.x + human.box.width), static_cast<float>(human.box.y + human.box.height));
        const float d = std::min(pointDistance(leftFoot, ballCenter), pointDistance(rightFoot, ballCenter));
        if (d < maxDistancePx && d < minDistance) {
            minDistance = d;
            assignedTrack = human.trackId;
        }
    }

    return assignedTrack;
}


cv::Mat letterboxImage(const cv::Mat& image, int targetSize, LetterboxInfo& info) {
    const int srcW = image.cols;
    const int srcH = image.rows;
    const float ratio = std::min(
        static_cast<float>(targetSize) / static_cast<float>(srcW),
        static_cast<float>(targetSize) / static_cast<float>(srcH));

    const int newW = static_cast<int>(std::round(static_cast<float>(srcW) * ratio));
    const int newH = static_cast<int>(std::round(static_cast<float>(srcH) * ratio));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    const int padW = targetSize - newW;
    const int padH = targetSize - newH;
    const int left = padW / 2;
    const int right = padW - left;
    const int top = padH / 2;
    const int bottom = padH - top;

    cv::Mat out;
    cv::copyMakeBorder(
        resized,
        out,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114));

    info.ratio = ratio;
    info.dw = static_cast<float>(left);
    info.dh = static_cast<float>(top);
    return out;
}

void interpolateBallDetections(std::vector<std::optional<Detection>>& balls) {
    if (balls.empty()) {
        return;
    }

    int firstValid = -1;
    for (std::size_t i = 0; i < balls.size(); ++i) {
        if (balls[i].has_value()) {
            firstValid = static_cast<int>(i);
            break;
        }
    }

    if (firstValid < 0) {
        return;
    }

    for (int i = 0; i < firstValid; ++i) {
        balls[static_cast<std::size_t>(i)] = balls[static_cast<std::size_t>(firstValid)];
    }

    int prevValid = firstValid;
    for (std::size_t i = static_cast<std::size_t>(firstValid + 1); i < balls.size(); ++i) {
        if (!balls[i].has_value()) {
            continue;
        }

        const int nextValid = static_cast<int>(i);
        const int gap = nextValid - prevValid;
        if (gap > 1) {
            const cv::Rect b0 = balls[static_cast<std::size_t>(prevValid)]->box;
            const cv::Rect b1 = balls[static_cast<std::size_t>(nextValid)]->box;
            const float c0 = balls[static_cast<std::size_t>(prevValid)]->confidence;
            const float c1 = balls[static_cast<std::size_t>(nextValid)]->confidence;

            for (int k = 1; k < gap; ++k) {
                const float t = static_cast<float>(k) / static_cast<float>(gap);
                Detection interp{};
                interp.classId = balls[static_cast<std::size_t>(prevValid)]->classId;
                interp.confidence = c0 + (c1 - c0) * t;
                interp.box.x = static_cast<int>(std::round(static_cast<float>(b0.x) + (static_cast<float>(b1.x - b0.x) * t)));
                interp.box.y = static_cast<int>(std::round(static_cast<float>(b0.y) + (static_cast<float>(b1.y - b0.y) * t)));
                interp.box.width = std::max(1, static_cast<int>(std::round(static_cast<float>(b0.width) + (static_cast<float>(b1.width - b0.width) * t))));
                interp.box.height = std::max(1, static_cast<int>(std::round(static_cast<float>(b0.height) + (static_cast<float>(b1.height - b0.height) * t))));
                balls[static_cast<std::size_t>(prevValid + k)] = interp;
            }
        }

        prevValid = nextValid;
    }

    for (std::size_t i = static_cast<std::size_t>(prevValid + 1); i < balls.size(); ++i) {
        balls[i] = balls[static_cast<std::size_t>(prevValid)];
    }
}

std::vector<Detection> decodeYoloOutput(
    const cv::Mat& output,
    float confThreshold,
    float nmsThreshold,
    const LetterboxInfo& letterbox,
    int frameWidth,
    int frameHeight) {
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    if (output.dims != 3) {
        return {};
    }

    const int dim1 = output.size[1];
    const int dim2 = output.size[2];
    const int rows = std::max(dim1, dim2);
    const int cols = std::min(dim1, dim2);
    const bool rowsAreDetections = (dim1 == rows);

    if (cols < 6) {
        return {};
    }

    const cv::Mat reshaped = output.reshape(1, dim1);
    cv::Mat detections = reshaped;
    if (!rowsAreDetections) {
        cv::transpose(reshaped, detections);
    }

    for (int i = 0; i < detections.rows; ++i) {
        const float* data = detections.ptr<float>(i);
        const float cx = data[0];
        const float cy = data[1];
        const float w = data[2];
        const float h = data[3];

        int bestClass = -1;
        float bestScore = 0.0f;

        int bestClassV8 = -1;
        float bestScoreV8 = 0.0f;
        for (int c = 4; c < detections.cols; ++c) {
            if (data[c] > bestScoreV8) {
                bestScoreV8 = data[c];
                bestClassV8 = c - 4;
            }
        }

        int bestClassV5 = -1;
        float bestScoreV5 = 0.0f;
        if (detections.cols >= 6) {
            const float objectness = data[4];
            for (int c = 5; c < detections.cols; ++c) {
                const float score = objectness * data[c];
                if (score > bestScoreV5) {
                    bestScoreV5 = score;
                    bestClassV5 = c - 5;
                }
            }
        }

        if (bestScoreV5 > bestScoreV8) {
            bestScore = bestScoreV5;
            bestClass = bestClassV5;
        } else {
            bestScore = bestScoreV8;
            bestClass = bestClassV8;
        }

        if (bestScore < confThreshold || bestClass < 0) {
            continue;
        }

        const float x1 = (cx - 0.5f * w - letterbox.dw) / letterbox.ratio;
        const float y1 = (cy - 0.5f * h - letterbox.dh) / letterbox.ratio;
        const float x2 = (cx + 0.5f * w - letterbox.dw) / letterbox.ratio;
        const float y2 = (cy + 0.5f * h - letterbox.dh) / letterbox.ratio;

        int left = static_cast<int>(std::round(x1));
        int top = static_cast<int>(std::round(y1));
        int width = static_cast<int>(std::round(x2 - x1));
        int height = static_cast<int>(std::round(y2 - y1));

        left = std::clamp(left, 0, frameWidth - 1);
        top = std::clamp(top, 0, frameHeight - 1);
        width = std::clamp(width, 1, frameWidth - left);
        height = std::clamp(height, 1, frameHeight - top);

        boxes.emplace_back(left, top, width, height);
        scores.push_back(bestScore);
        classIds.push_back(bestClass);
    }

    std::vector<Detection> result;
    std::unordered_map<int, std::vector<int>> classToIndices;
    for (int i = 0; i < static_cast<int>(classIds.size()); ++i) {
        classToIndices[classIds[static_cast<std::size_t>(i)]].push_back(i);
    }

    for (const auto& [cls, indices] : classToIndices) {
        std::vector<cv::Rect> clsBoxes;
        std::vector<float> clsScores;
        clsBoxes.reserve(indices.size());
        clsScores.reserve(indices.size());

        for (int idx : indices) {
            clsBoxes.push_back(boxes[static_cast<std::size_t>(idx)]);
            clsScores.push_back(scores[static_cast<std::size_t>(idx)]);
        }

        std::vector<int> keptLocal;
        cv::dnn::NMSBoxes(clsBoxes, clsScores, confThreshold, nmsThreshold, keptLocal);
        for (int local : keptLocal) {
            const int original = indices[static_cast<std::size_t>(local)];
            result.push_back(Detection{cls, scores[static_cast<std::size_t>(original)], boxes[static_cast<std::size_t>(original)]});
        }
    }

    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Kullanim: " << argv[0]
                  << " <model.onnx> <input_video> [output_video] [ball_id player_id goalkeeper_id referee_id]\n";
        return 1;
    }

    const std::string modelPath = argv[1];
    const std::string inputVideoPath = argv[2];
    const std::string outputVideoPath = (argc >= 4) ? argv[3] : "output_test_cpp.mp4";

    ClassMapping classMapping{};
    if (argc >= 8) {
        try {
            classMapping.ballId = std::stoi(argv[4]);
            classMapping.playerId = std::stoi(argv[5]);
            classMapping.goalkeeperId = std::stoi(argv[6]);
            classMapping.refereeId = std::stoi(argv[7]);
        } catch (const std::exception&) {
            std::cerr << "Sinif id argumanlari sayi olmali.\n";
            return 1;
        }
    }

    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "Model bulunamadi: " << modelPath << "\n";
        return 1;
    }

    if (!std::filesystem::exists(inputVideoPath)) {
        std::cerr << "Video bulunamadi: " << inputVideoPath << "\n";
        return 1;
    }

    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(modelPath);
    } catch (const std::exception& ex) {
        std::cerr << "Model yuklenemedi: " << ex.what() << "\n";
        return 1;
    }

    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) {
        std::cerr << "Video acilamadi: " << inputVideoPath << "\n";
        return 1;
    }

    const int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double fps = cap.get(cv::CAP_PROP_FPS) > 0.0 ? cap.get(cv::CAP_PROP_FPS) : 25.0;

    const int inputSize = 640;
    const float confThreshold = 0.10f;
    const float nmsThreshold = 0.45f;

    cv::VideoWriter writer(
        outputVideoPath,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(frameWidth, frameHeight));

    if (!writer.isOpened()) {
        std::cerr << "Cikti videosu olusturulamadi: " << outputVideoPath << "\n";
        return 1;
    }

    // --- 1. GECiS: Tum frame'leri oku + YOLO tespitlerini topla (ByteTrack yok) ---
    // Kamera hareketi tahmin edilmeden once tracking yapilmamali.
    // Python da ayni sekilde once tum frame'leri alip sonra isliyordu.

    struct RawFrameDetections {
        std::vector<HumanDetection> humans;  // trackId=-1, ham tespitler
        std::optional<Detection> ball;
    };

    std::vector<cv::Mat> frames;
    std::vector<RawFrameDetections> rawPerFrame;

    cv::Mat frame;
    int totalBallDetections = 0;
    int totalHumanDetections = 0;

    while (cap.read(frame)) {
        LetterboxInfo letterbox{};
        const cv::Mat padded = letterboxImage(frame, inputSize, letterbox);
        cv::Mat blob = cv::dnn::blobFromImage(
            padded,
            1.0 / 255.0,
            cv::Size(inputSize, inputSize),
            cv::Scalar(),
            true,
            false);

        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        RawFrameDetections raw{};

        if (!outputs.empty()) {
            const auto detections = decodeYoloOutput(
                outputs[0],
                confThreshold,
                nmsThreshold,
                letterbox,
                frame.cols,
                frame.rows);

            for (const auto& det : detections) {
                int mappedClassId = det.classId;
                if (mappedClassId == classMapping.goalkeeperId) {
                    mappedClassId = classMapping.playerId;
                }

                if (isHumanClass(mappedClassId, classMapping)) {
                    ++totalHumanDetections;
                    raw.humans.push_back(HumanDetection{-1, mappedClassId, det.confidence, det.box, false});
                } else if (mappedClassId == classMapping.ballId) {
                    if (!raw.ball.has_value() || det.confidence > raw.ball->confidence) {
                        raw.ball = Detection{mappedClassId, det.confidence, det.box};
                    }
                }
            }

            if (raw.ball.has_value()) {
                ++totalBallDetections;
            }
        }

        frames.push_back(frame.clone());
        rawPerFrame.push_back(std::move(raw));
    }

    // --- 2. GECiS: Kamera hareketi tahmini ---
    // Python'daki camera_movement_estimator.get_camera_movement() karsiligi.
    // Lucas-Kanade optical flow ile frame kenarlarindan kamera deplasmanini hesapla.
    std::cout << "Kamera hareketi tahmin ediliyor (" << frames.size() << " frame)...\n";
    fa::CameraMotionEstimator cameraEstimator;
    const std::vector<cv::Point2f> cameraMotion = cameraEstimator.estimate(frames);

    // --- 3. GECiS: Kamera kompanzasyonu + ByteTrack + Homografi + top sahipligi ---
    // Tespit kutularini kamera hareketine gore duzelttikten sonra tracker'a ver.
    // ByteTrack kamera sapması olmadan tutarli IOU eslesmeleri yapabilsin.
    // Goruntu cizimi icin kutular orijinal koordinatlara geri cevrilir.
    // Homografi ile saha (dunya) koordinatlari hesaplanir.

    std::vector<std::vector<HumanDetection>> humansPerFrame;
    std::vector<std::optional<Detection>> ballPerFrame;

    // Per-frame world position haritasi: trackId → saha koordinati (metre)
    // std::nullopt: oyuncu saha polygon'u disinda
    using WorldPosMap = std::unordered_map<int, std::optional<fa::Point2f>>;
    std::vector<WorldPosMap> worldPosPerFrame;
    std::vector<std::optional<fa::Point2f>> ballWorldPerFrame;

    fa::ByteTrackLite humanTracker(0.45f, 0.10f, 0.30f, 0.20f, 30);
    fa::HomographyTransformer homography;

    for (std::size_t i = 0; i < frames.size(); ++i) {
        auto humans = rawPerFrame[i].humans;
        auto ball = rawPerFrame[i].ball;

        // Kamera deplasmanini tespit kutularindan cikar (piksel olarak yuvarla)
        const int dxi = static_cast<int>(std::round(cameraMotion[i].x));
        const int dyi = static_cast<int>(std::round(cameraMotion[i].y));

        if (dxi != 0 || dyi != 0) {
            for (auto& h : humans) {
                h.box.x -= dxi;
                h.box.y -= dyi;
            }
            if (ball.has_value()) {
                ball->box.x -= dxi;
                ball->box.y -= dyi;
            }
        }

        // ByteTrack: kompanze edilmis koordinatlar uzerinde calistir
        auto tracked = humanTracker.update(humans);

        // Goruntu cizimleri icin orijinal frame koordinatlarına geri al
        if (dxi != 0 || dyi != 0) {
            for (auto& h : tracked) {
                h.box.x += dxi;
                h.box.y += dyi;
            }
            if (ball.has_value()) {
                ball->box.x += dxi;
                ball->box.y += dyi;
            }
        }

        // Homografi: her oyuncu icin saha koordinatini hesapla.
        // Python: position_adjusted = position - camera_movement
        // Burada position_adjusted = orijinal konum - cameraMotion[i] (float degerle)
        WorldPosMap worldMap;
        for (const auto& h : tracked) {
            const float footX = static_cast<float>(h.box.x + h.box.width / 2);
            const float footY = static_cast<float>(h.box.y + h.box.height);
            const fa::Point2f adjustedPos{footX - cameraMotion[i].x, footY - cameraMotion[i].y};
            worldMap[h.trackId] = homography.toWorld(adjustedPos);
        }
        worldPosPerFrame.push_back(std::move(worldMap));

        // Top icin saha koordinati (merkez noktasi kullanilir)
        std::optional<fa::Point2f> ballWorld;
        if (ball.has_value()) {
            const float cx = static_cast<float>(ball->box.x + ball->box.width / 2);
            const float cy = static_cast<float>(ball->box.y + ball->box.height / 2);
            const fa::Point2f adjustedPos{cx - cameraMotion[i].x, cy - cameraMotion[i].y};
            ballWorld = homography.toWorld(adjustedPos);
        }
        ballWorldPerFrame.push_back(ballWorld);

        // Top sahipligi atamasi (orijinal koordinatlarla)
        if (ball.has_value()) {
            const int assignedTrack = assignBallToClosestPlayer(tracked, ball->box, classMapping.playerId, 70.0f);
            if (assignedTrack != -1) {
                for (auto& h : tracked) {
                    if (h.trackId == assignedTrack) {
                        h.hasBall = true;
                        break;
                    }
                }
            }
        }

        humansPerFrame.push_back(std::move(tracked));
        ballPerFrame.push_back(ball);
    }

    interpolateBallDetections(ballPerFrame);

    int totalWorldPositions = 0;

    for (std::size_t i = 0; i < frames.size(); ++i) {
        auto& outFrame = frames[i];
        const WorldPosMap& worldMap = worldPosPerFrame[i];

        for (const auto& human : humansPerFrame[i]) {
            const cv::Scalar color = classColor(human.classId, classMapping);
            drawPlayerEllipse(outFrame, human.box, color, human.trackId);
            if (human.hasBall) {
                drawBallTriangle(outFrame, human.box, cv::Scalar(0, 0, 255));
            }

            // Saha koordinatini oyuncunun altina yaz (saha icindeyse)
            const auto it = worldMap.find(human.trackId);
            if (it != worldMap.end() && it->second.has_value()) {
                ++totalWorldPositions;
                const fa::Point2f& wp = it->second.value();
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.1f,%.1f", wp.x, wp.y);
                const int xCenter = human.box.x + human.box.width / 2;
                const int y2 = human.box.y + human.box.height;
                cv::putText(
                    outFrame,
                    buf,
                    cv::Point(xCenter - 18, y2 + 42),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.40,
                    cv::Scalar(0, 255, 255),  // sari-yesil (cyan)
                    1);
            }
        }

        if (ballPerFrame[i].has_value()) {
            drawBallTriangle(outFrame, ballPerFrame[i]->box, classColor(classMapping.ballId, classMapping));

            // Top saha koordinatini topun yanina yaz
            if (ballWorldPerFrame[i].has_value()) {
                const fa::Point2f& wp = ballWorldPerFrame[i].value();
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.1f,%.1f", wp.x, wp.y);
                const int bx = ballPerFrame[i]->box.x + ballPerFrame[i]->box.width / 2;
                const int by = ballPerFrame[i]->box.y;
                cv::putText(
                    outFrame,
                    buf,
                    cv::Point(bx + 8, by - 8),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.40,
                    cv::Scalar(0, 255, 0),  // yesil
                    1);
            }
        }

        writer.write(outFrame);
    }

    std::cout << "Islenen frame sayisi: " << frames.size() << "\n";
    std::cout << "Top tespit sayisi: " << totalBallDetections << "\n";
    std::cout << "Insan tespit sayisi: " << totalHumanDetections << "\n";
    std::cout << "Dunya koordinati hesaplanan: " << totalWorldPositions << " tespit\n";
    std::cout << "Cikti videosu: " << outputVideoPath << "\n";
    return 0;
}
