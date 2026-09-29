// FaceStitche intermediate demo (standalone C++17 implementation).
//
// The camera acquisition part is intentionally omitted.  The program reads
// the recorded ROI images and mesh from Input/, then writes the results to
// Output/. Calibration and run-specific transforms are embedded below.
//
// OpenCV is used for the image and colour operations.  The small PLY/PCD
// readers below keep this demo independent of Qt, the camera SDK and PCL
// while preserving the same data flow and matrix conventions.

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using Point3 = cv::Point3f;
using Face = std::vector<int>;

struct CameraCalibration {
    cv::Matx33f K{};
    std::array<float, 5> Kc{};
    cv::Matx33f R{};
    cv::Vec3f T{};
};

struct RoiMapping {
    float lx = 0.f, ly = 0.f, rx = 0.f, ry = 0.f;
    float width = 0.f, height = 0.f;
};

struct Mesh {
    std::vector<Point3> vertices;
    std::vector<Face> faces;
};

struct OrganizedCloud {
    int width = 0, height = 0;
    std::vector<Point3> points;
};

struct DepthFilterStats {
    size_t rightRawValid = 0, rightRawInvalid = 0, rightPixelOut = 0;
    size_t leftCandidates = 0, checked = 0, kept = 0;
    size_t rejectedPredInvalid = 0, rejectedRightPixelOut = 0;
    size_t rejectedNoObservation = 0, rejectedC2c = 0, rejectedDepth = 0;
};

static std::string readText(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void recordedParameters(const cv::Size& roiImageSize, cv::Matx44f& sac,
                               cv::Matx44f& icp, cv::Matx33f& colour,
                               CameraCalibration& leftCamera, CameraCalibration& rightCamera,
                               RoiMapping& roi, int& rgbWidth) {
    // Values recorded in the same capture as Input/mesh.ply and the two ROI PNGs.
    // A different capture needs its own matching calibration and transforms.
    if (roiImageSize != cv::Size(839, 768))
        throw std::runtime_error("ROI image shape does not match the embedded capture parameters (839x768)");
    sac = cv::Matx44f(0.92006f, 0.291485f, 0.261777f, -131.759f,
                     -0.389089f, 0.601661f, 0.697577f, -400.253f,
                     0.0458319f, -0.743667f, 0.666978f, 194.712f,
                     0.f, 0.f, 0.f, 1.f);
    icp = cv::Matx44f(0.998841f, 0.0458283f, 0.0152881f, -10.2782f,
                      -0.0473837f, 0.867569f, 0.495066f, -330.809f,
                      0.00942494f, -0.49521f, 0.868727f, 91.3583f,
                      0.f, 0.f, 0.f, 1.f);
    colour = cv::Matx33f(0.882276f, 0.166891f, -0.0624152f,
                         0.196258f, 0.909857f, -0.130364f,
                         0.530117f, -0.379741f, 0.843794f);
    leftCamera.K = cv::Matx33f(2043.24f, 0.f, 959.333f,
                               0.f, 2043.59f, 1243.07f,
                               0.f, 0.f, 1.f);
    leftCamera.Kc = {0.14914f, -0.350025f, 0.000820168f, 0.000929513f, 0.f};
    leftCamera.R = cv::Matx33f(0.998912f, 0.00839277f, 0.0458813f,
                               -0.00795956f, 0.999922f, -0.0096167f,
                               -0.0459584f, 0.00924104f, 0.998901f);
    leftCamera.T = cv::Vec3f(-55.9109f, 0.902611f, -12.9045f);
    rightCamera.K = cv::Matx33f(2049.48f, 0.f, 964.197f,
                                0.f, 2049.29f, 1342.38f,
                                0.f, 0.f, 1.f);
    rightCamera.Kc = {0.160907f, -0.324665f, -0.0015037f, -0.000438364f, 0.f};
    rightCamera.R = cv::Matx33f(0.998339f, -0.00415198f, 0.057459f,
                                0.00424261f, 0.99999f, -0.00145538f,
                                -0.0574524f, 0.00169674f, 0.998347f);
    rightCamera.T = cv::Vec3f(-55.7382f, -0.071412f, -13.6707f);
    rgbWidth = 1944;
    // The logged ROI origins were (922,636) and (793,652); the saved PNGs
    // include 120 pixels of padding on each side.
    roi = RoiMapping{802.f, 516.f, 673.f, 532.f, 839.f, 768.f};
}

static OrganizedCloud readBinaryXyzPcd(const fs::path& path) {
    static_assert(sizeof(Point3) == 3 * sizeof(float), "PCD XYZ layout requires three packed floats");
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    OrganizedCloud cloud;
    std::string line, fields, sizes, types, counts, data;
    size_t pointCount = 0;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream row(line);
        std::string key;
        row >> key;
        if (key == "WIDTH") row >> cloud.width;
        else if (key == "HEIGHT") row >> cloud.height;
        else if (key == "POINTS") row >> pointCount;
        else if (key == "FIELDS") fields = line;
        else if (key == "SIZE") sizes = line;
        else if (key == "TYPE") types = line;
        else if (key == "COUNT") counts = line;
        else if (key == "DATA") { row >> data; break; }
    }
    if (cloud.width <= 0 || cloud.height <= 0 ||
        pointCount != static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height) ||
        fields != "FIELDS x y z" || sizes != "SIZE 4 4 4" ||
        types != "TYPE F F F" || counts != "COUNT 1 1 1" || data != "binary")
        throw std::runtime_error("expected organized binary XYZ float32 PCD: " + path.string());
    cloud.points.resize(pointCount);
    const auto bytes = static_cast<std::streamsize>(pointCount * sizeof(Point3));
    in.read(reinterpret_cast<char*>(cloud.points.data()), bytes);
    if (in.gcount() != bytes) throw std::runtime_error("truncated PCD data: " + path.string());
    return cloud;
}

static Mesh readBinaryPly(const fs::path& path) {
    const std::string raw = readText(path); // binary-safe string
    const auto endHeader = raw.find("end_header");
    if (endHeader == std::string::npos) throw std::runtime_error("invalid PLY header");
    size_t data = endHeader + std::string("end_header").size();
    while (data < raw.size() && (raw[data] == '\n' || raw[data] == '\r' || raw[data] == ' ')) ++data;
    const std::string header = raw.substr(0, data);
    std::smatch match;
    std::regex vertexRegex(R"(element\s+vertex\s+(\d+))");
    std::regex faceRegex(R"(element\s+face\s+(\d+))");
    if (!std::regex_search(header, match, vertexRegex)) throw std::runtime_error("PLY vertex count missing");
    const size_t vertexCount = static_cast<size_t>(std::stoull(match[1].str()));
    if (!std::regex_search(header, match, faceRegex)) throw std::runtime_error("PLY face count missing");
    const size_t faceCount = static_cast<size_t>(std::stoull(match[1].str()));
    if (data + vertexCount * 5u * sizeof(float) > raw.size()) throw std::runtime_error("truncated PLY vertices");
    Mesh mesh;
    mesh.vertices.resize(vertexCount);
    const auto* bytes = reinterpret_cast<const unsigned char*>(raw.data());
    for (size_t i = 0; i < vertexCount; ++i) {
        const auto* p = reinterpret_cast<const float*>(bytes + data + i * 5u * sizeof(float));
        mesh.vertices[i] = Point3(p[0], p[1], p[2]);
    }
    size_t pos = data + vertexCount * 5u * sizeof(float);
    mesh.faces.reserve(faceCount);
    for (size_t i = 0; i < faceCount; ++i) {
        if (pos + sizeof(std::int32_t) > raw.size()) throw std::runtime_error("truncated PLY faces");
        std::int32_t count = 0;
        std::memcpy(&count, bytes + pos, sizeof(count));
        pos += sizeof(count);
        if (count < 0 || pos + static_cast<size_t>(count) * sizeof(std::int32_t) > raw.size())
            throw std::runtime_error("invalid PLY face");
        Face face(static_cast<size_t>(count));
        for (int j = 0; j < count; ++j) {
            std::int32_t index = 0;
            std::memcpy(&index, bytes + pos, sizeof(index));
            pos += sizeof(index);
            face[static_cast<size_t>(j)] = static_cast<int>(index);
        }
        mesh.faces.push_back(std::move(face));
    }
    return mesh;
}

static Point3 transform(const cv::Matx44f& m, const Point3& p) {
    const cv::Vec4f q = m * cv::Vec4f(p.x, p.y, p.z, 1.f);
    return Point3(q[0] / q[3], q[1] / q[3], q[2] / q[3]);
}

static cv::Mat rotateRight(const cv::Mat& image) {
    cv::Mat out;
    cv::rotate(image, out, cv::ROTATE_90_CLOCKWISE);
    return out;
}

static cv::Mat applyColourMatrix(const cv::Mat& image, const cv::Matx33f& matrix) {
    cv::Mat out(image.size(), CV_8UC3);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            const cv::Vec3b p = image.at<cv::Vec3b>(y, x);
            cv::Vec3f v(p[0] / 255.f, p[1] / 255.f, p[2] / 255.f);
            cv::Vec3f q = matrix * v;
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(
                cv::saturate_cast<uchar>(std::clamp(q[0], 0.f, 1.f) * 255.f),
                cv::saturate_cast<uchar>(std::clamp(q[1], 0.f, 1.f) * 255.f),
                cv::saturate_cast<uchar>(std::clamp(q[2], 0.f, 1.f) * 255.f));
        }
    }
    return out;
}

static cv::Mat bgrSamplesToLab(const std::vector<cv::Vec3b>& colours) {
    cv::Mat bgr(static_cast<int>(colours.size()), 1, CV_32FC3);
    for (int i = 0; i < bgr.rows; ++i) {
        bgr.at<cv::Vec3f>(i, 0) = cv::Vec3f(
            colours[static_cast<size_t>(i)][0] / 255.f,
            colours[static_cast<size_t>(i)][1] / 255.f,
            colours[static_cast<size_t>(i)][2] / 255.f);
    }
    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    return lab;
}

static cv::Mat reinhardL(const cv::Mat& image, const cv::Mat& sourceLab, const cv::Mat& targetLab) {
    cv::Scalar muS, sigmaS, muT, sigmaT;
    cv::meanStdDev(sourceLab.reshape(1), muS, sigmaS);
    cv::meanStdDev(targetLab.reshape(1), muT, sigmaT);
    cv::Mat source;
    image.convertTo(source, CV_32FC3, 1.0 / 255.0);
    cv::Mat lab;
    cv::cvtColor(source, lab, cv::COLOR_BGR2Lab);
    const float s = static_cast<float>(sigmaS[0]);
    const float scale = s > 1e-6f ? static_cast<float>(sigmaT[0]) / s : 1.f;
    lab.forEach<cv::Vec3f>([&](cv::Vec3f& p, const int*) {
        p[0] = std::clamp((p[0] - static_cast<float>(muS[0])) * scale + static_cast<float>(muT[0]), 0.f, 100.f);
    });
    cv::Mat bgr;
    cv::cvtColor(lab, bgr, cv::COLOR_Lab2BGR);
    cv::Mat result;
    bgr.convertTo(result, CV_8UC3, 255.0);
    return result;
}

static float meanDistance(const std::vector<cv::Vec3b>& a, const std::vector<cv::Vec3b>& b) {
    if (a.empty() || a.size() != b.size()) return 0.f;
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const cv::Vec3f d = cv::Vec3f(a[i]) - cv::Vec3f(b[i]);
        sum += std::sqrt(d.dot(d));
    }
    return static_cast<float>(sum / static_cast<double>(a.size()));
}

static float meanLabDistance(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || a.rows != b.rows) return 0.f;
    double sum = 0.0;
    for (int i = 0; i < a.rows; ++i) {
        const cv::Vec3f d = a.at<cv::Vec3f>(i, 0) - b.at<cv::Vec3f>(i, 0);
        sum += std::sqrt(d.dot(d));
    }
    return static_cast<float>(sum / std::max(1, a.rows));
}

static cv::Point2f projectCamera(const Point3& p, const CameraCalibration& c, bool& valid) {
    const cv::Vec3f cam = c.R * cv::Vec3f(p.x, p.y, p.z) + c.T;
    valid = std::isfinite(cam[0]) && std::isfinite(cam[1]) && std::isfinite(cam[2]) && std::abs(cam[2]) > 1e-6f;
    const float z = valid ? cam[2] : 1.f;
    const float u = cam[0] / z, v = cam[1] / z;
    const float r = u * v + v * v; // same expression as FaceStitche::Point2RGBXY
    const float dx1 = 2.f * c.Kc[2] * u * v + c.Kc[3] * (r + 2.f * u * u);
    const float dx2 = c.Kc[2] * (r + 2.f * v * v) + 2.f * c.Kc[3] * u * v;
    const float a1 = (1.f + c.Kc[0] * r + c.Kc[1] * r * r + c.Kc[4] * u * u * u) * u + dx1;
    const float a2 = (1.f + c.Kc[0] * r + c.Kc[1] * r * r + c.Kc[4] * v * v * v) * v + dx2;
    return cv::Point2f(c.K(0,0) * a1 + c.K(0,1) * a2 + c.K(0,2),
                       c.K(1,0) * a1 + c.K(1,1) * a2 + c.K(1,2));
}

static cv::Point2f toStitchedUv(const cv::Point2f& uv, bool left, const RoiMapping& roi, int rgbWidth) {
    if (left) return cv::Point2f(uv.x - (rgbWidth - roi.height - roi.ly), uv.y - roi.lx);
    return cv::Point2f(uv.x - (rgbWidth - roi.height - roi.ry), uv.y + roi.width - roi.rx);
}

static bool validDepthPoint(const Point3& p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
           !(p.x == 0.f && p.y == 0.f && p.z == 0.f);
}

static bool projectedStitchedPixel(const Point3& point, const CameraCalibration& camera,
                                  bool leftSide, const RoiMapping& roi, int rgbWidth,
                                  const cv::Size& roiImageSize, cv::Point& pixel) {
    bool valid = false;
    const cv::Point2f uv = toStitchedUv(projectCamera(point, camera, valid), leftSide, roi, rgbWidth);
    if (!valid || !std::isfinite(uv.x) || !std::isfinite(uv.y) ||
        std::abs(uv.x) > 1e6f || std::abs(uv.y) > 1e6f) return false;
    pixel = cv::Point(cvRound(uv.x), cvRound(uv.y));
    return pixel.x >= 0 && pixel.x < roiImageSize.height &&
           pixel.y >= (leftSide ? 0 : roiImageSize.width) &&
           pixel.y < (leftSide ? roiImageSize.width : 2 * roiImageSize.width);
}

static DepthFilterStats selectDepthConsistentPairs(
    const OrganizedCloud& leftCloud, const OrganizedCloud& rightCloud,
    const cv::Matx44f& icp, const CameraCalibration& leftCamera,
    const CameraCalibration& rightCamera, const RoiMapping& roi, int rgbWidth,
    const cv::Mat& leftImage, const cv::Mat& rightImage, const fs::path& csvPath,
    std::vector<cv::Vec3b>& rawLeft, std::vector<cv::Vec3b>& rawRight,
    std::vector<cv::Point>& rawPositions) {
    // These are the right depth ROI coordinates recorded for this capture.
    constexpr int rightU0 = 563, rightV0 = 123, rightW = 348, rightH = 418;
    constexpr size_t maxToCheck = 300;
    constexpr int halfWindow = 1;
    constexpr float maxC2cDistance = 1.f, maxDepthDifference = 1.f;
    if (leftCloud.width != 1224 || leftCloud.height != 1024 ||
        rightCloud.width != 1224 || rightCloud.height != 1024)
        throw std::runtime_error("PCD shape does not match the embedded capture parameters (1224x1024)");
    DepthFilterStats stats;
    std::unordered_map<int, std::vector<size_t>> rightPixelToRawIndices;
    rightPixelToRawIndices.reserve(static_cast<size_t>(rightW * rightH));
    for (int y = rightV0; y < rightV0 + rightH; ++y) {
        for (int x = rightU0; x < rightU0 + rightW; ++x) {
            const size_t index = static_cast<size_t>(y) * rightCloud.width + x;
            const Point3& observed = rightCloud.points[index];
            if (!validDepthPoint(observed)) { ++stats.rightRawInvalid; continue; }
            ++stats.rightRawValid;
            cv::Point pixel;
            if (!projectedStitchedPixel(observed, rightCamera, false, roi, rgbWidth,
                                       leftImage.size(), pixel)) {
                ++stats.rightPixelOut;
                continue;
            }
            const int key = pixel.y * leftImage.rows + pixel.x;
            rightPixelToRawIndices[key].push_back(index);
        }
    }

    // The saved demo has RGB ROI images but no gray ROI landmark coordinates.
    // Use left raw depth points that actually project into the saved left ROI.
    std::vector<std::pair<size_t, cv::Point>> candidates;
    for (size_t i = 0; i < leftCloud.points.size(); ++i) {
        const Point3& p = leftCloud.points[i];
        if (!validDepthPoint(p)) continue;
        cv::Point pixel;
        if (projectedStitchedPixel(p, leftCamera, true, roi, rgbWidth,
                                  leftImage.size(), pixel)) candidates.emplace_back(i, pixel);
    }
    stats.leftCandidates = candidates.size();
    std::mt19937 generator(20250331);
    std::shuffle(candidates.begin(), candidates.end(), generator);
    stats.checked = std::min(maxToCheck, candidates.size());

    std::ofstream csv(csvPath);
    if (!csv) throw std::runtime_error("cannot write " + csvPath.string());
    csv << "status,left_index,left_x,left_y,left_z,pred_right_x,pred_right_y,pred_right_z,"
           "left_pixel_x,left_pixel_y,pred_right_pixel_x,pred_right_pixel_y,"
           "obs_x,obs_y,obs_z,obs_pixel_x,obs_pixel_y,c2c_distance,depth_difference\n";
    csv << std::setprecision(8);
    const Point3 zero(0.f, 0.f, 0.f);
    const cv::Point absent(-1, -1);
    auto record = [&](const char* status, size_t index, const Point3& leftPoint,
                      const Point3& predicted, const cv::Point& leftPixel,
                      const cv::Point& predictedPixel, const Point3& observed,
                      const cv::Point& observedPixel, float distance, float depthDifference) {
        csv << status << ',' << index << ',' << leftPoint.x << ',' << leftPoint.y << ',' << leftPoint.z
            << ',' << predicted.x << ',' << predicted.y << ',' << predicted.z
            << ',' << leftPixel.x << ',' << leftPixel.y
            << ',' << predictedPixel.x << ',' << predictedPixel.y
            << ',' << observed.x << ',' << observed.y << ',' << observed.z
            << ',' << observedPixel.x << ',' << observedPixel.y
            << ',' << distance << ',' << depthDifference << '\n';
    };
    for (size_t k = 0; k < stats.checked; ++k) {
        const auto [index, leftPixel] = candidates[k];
        const Point3& leftPoint = leftCloud.points[index];
        const Point3 predicted = transform(icp, leftPoint);
        if (!validDepthPoint(predicted)) {
            ++stats.rejectedPredInvalid;
            record("PRED_RIGHT_INVALID", index, leftPoint, predicted, leftPixel,
                   absent, zero, absent, -1.f, -1.f);
            continue;
        }
        cv::Point predictedPixel;
        if (!projectedStitchedPixel(predicted, rightCamera, false, roi, rgbWidth,
                                   leftImage.size(), predictedPixel)) {
            ++stats.rejectedRightPixelOut;
            record("PRED_RIGHT_PIXEL_OUT", index, leftPoint, predicted, leftPixel,
                   absent, zero, absent, -1.f, -1.f);
            continue;
        }
        bool found = false;
        float bestDistance = std::numeric_limits<float>::max(), bestDz = 0.f;
        Point3 bestObserved = zero;
        cv::Point bestPixel = absent;
        for (int dv = -halfWindow; dv <= halfWindow; ++dv) {
            for (int du = -halfWindow; du <= halfWindow; ++du) {
                const cv::Point pixel(predictedPixel.x + du, predictedPixel.y + dv);
                if (pixel.x < 0 || pixel.x >= leftImage.rows ||
                    pixel.y < leftImage.cols || pixel.y >= 2 * leftImage.cols) continue;
                const auto it = rightPixelToRawIndices.find(pixel.y * leftImage.rows + pixel.x);
                if (it == rightPixelToRawIndices.end()) continue;
                for (size_t observedIndex : it->second) {
                    const Point3& observed = rightCloud.points[observedIndex];
                    const float dx = predicted.x - observed.x;
                    const float dy = predicted.y - observed.y;
                    const float dz = predicted.z - observed.z;
                    const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (distance < bestDistance) {
                        found = true;
                        bestDistance = distance;
                        bestDz = dz;
                        bestObserved = observed;
                        bestPixel = pixel;
                    }
                }
            }
        }
        if (!found) {
            ++stats.rejectedNoObservation;
            record("NO_VALID_3X3_OBS", index, leftPoint, predicted, leftPixel,
                   predictedPixel, zero, absent, -1.f, -1.f);
            continue;
        }
        if (bestDistance > maxC2cDistance) {
            ++stats.rejectedC2c;
            record("PROJ_C2C_TOO_LARGE", index, leftPoint, predicted, leftPixel,
                   predictedPixel, bestObserved, bestPixel, bestDistance, bestDz);
            continue;
        }
        if (std::abs(bestDz) > maxDepthDifference) {
            ++stats.rejectedDepth;
            record("PROJ_DEPTH_TOO_LARGE", index, leftPoint, predicted, leftPixel,
                   predictedPixel, bestObserved, bestPixel, bestDistance, bestDz);
            continue;
        }
        ++stats.kept;
        record("KEEP", index, leftPoint, predicted, leftPixel,
               predictedPixel, bestObserved, bestPixel, bestDistance, bestDz);
        const cv::Point leftPosition(leftPixel.y, leftImage.rows - 1 - leftPixel.x);
        const cv::Point rightPosition(bestPixel.y - leftImage.cols,
                                      rightImage.rows - 1 - bestPixel.x);
        const cv::Vec3b l = leftImage.at<cv::Vec3b>(leftPosition);
        const cv::Vec3b r = rightImage.at<cv::Vec3b>(rightPosition);
        const float brightness = (l[0] + l[1] + l[2] + r[0] + r[1] + r[2]) / 6.f;
        if (brightness < 245.f) {
            rawLeft.push_back(l);
            rawRight.push_back(r);
            rawPositions.push_back(leftPosition);
        }
    }
    if (rawLeft.empty()) throw std::runtime_error("depth verification found no usable colour pairs");
    return stats;
}

static cv::Vec3b sampleClamped(const cv::Mat& image, cv::Point2f uv) {
    const int x = std::clamp(cvRound(uv.x), 0, image.cols - 1);
    const int y = std::clamp(cvRound(uv.y), 0, image.rows - 1);
    return image.at<cv::Vec3b>(y, x);
}

static std::vector<cv::Vec3f> meshNormals(const Mesh& mesh) {
    std::vector<cv::Vec3f> normals(mesh.vertices.size(), cv::Vec3f(0, 0, 0));
    for (const Face& face : mesh.faces) {
        if (face.size() < 3) continue;
        const Point3 a = mesh.vertices[static_cast<size_t>(face[0])];
        const Point3 b = mesh.vertices[static_cast<size_t>(face[1])];
        const Point3 c = mesh.vertices[static_cast<size_t>(face[2])];
        const cv::Vec3f u(b.x - a.x, b.y - a.y, b.z - a.z);
        const cv::Vec3f v(c.x - a.x, c.y - a.y, c.z - a.z);
        const cv::Vec3f n = u.cross(v);
        for (int index : face) normals[static_cast<size_t>(index)] += n;
    }
    for (auto& n : normals) {
        const float length = std::sqrt(n.dot(n));
        n = length > 1e-8f ? n / length : cv::Vec3f(0, 0, 1);
    }
    return normals;
}

static void writeObj(const fs::path& objPath, const fs::path& mtlPath,
                     const Mesh& mesh, const std::vector<cv::Vec3b>& colours,
                     const std::vector<cv::Vec3f>& normals) {
    std::ofstream mtl(mtlPath);
    mtl << "# FaceStitche demo material\nnewmtl VertexColorMaterial\n"
        << "Ka 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nNs 0\nillum 0\n";
    std::ofstream obj(objPath);
    obj << "# FaceStitche demo vertex-colour mesh\n"
        << "# Vertices: " << mesh.vertices.size() << "\n"
        << "# Faces: " << mesh.faces.size() << "\n"
        << "mtllib " << mtlPath.filename().string() << "\n"
        << "usemtl VertexColorMaterial\n";
    obj << std::setprecision(8);
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        const Point3& p = mesh.vertices[i];
        const cv::Vec3b& c = colours[i];
        obj << "v " << p.x << ' ' << p.y << ' ' << p.z << ' '
            << c[2] / 255.f << ' ' << c[1] / 255.f << ' ' << c[0] / 255.f << '\n';
    }
    for (const auto& n : normals) obj << "vn " << n[0] << ' ' << n[1] << ' ' << n[2] << '\n';
    for (const Face& face : mesh.faces) {
        obj << "f";
        for (int index : face) obj << ' ' << index + 1 << "//" << index + 1;
        obj << '\n';
    }
}

static std::string cameraJson(const CameraCalibration& c) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(8)
        << "{\"K\": [[" << c.K(0,0) << ", " << c.K(0,1) << ", " << c.K(0,2)
        << "], [" << c.K(1,0) << ", " << c.K(1,1) << ", " << c.K(1,2)
        << "], [0, 0, 1]], \"Kc\": [" << c.Kc[0] << ", " << c.Kc[1] << ", " << c.Kc[2]
        << ", " << c.Kc[3] << ", " << c.Kc[4] << "], \"R\": [["
        << c.R(0,0) << ", " << c.R(0,1) << ", " << c.R(0,2) << "], ["
        << c.R(1,0) << ", " << c.R(1,1) << ", " << c.R(1,2) << "], ["
        << c.R(2,0) << ", " << c.R(2,1) << ", " << c.R(2,2) << "]], \"T\": ["
        << c.T[0] << ", " << c.T[1] << ", " << c.T[2] << "]}";
    return out.str();
}

int main(int argc, char** argv) {
    try {
        fs::path root = fs::current_path();
        if (argc >= 3 && std::string(argv[1]) == "--root") root = fs::path(argv[2]);
        const fs::path input = root / "Input";
        const fs::path output = root / "Output";
        const cv::Mat left = cv::imread((input / "rgb_face_roi_l.png").string(), cv::IMREAD_COLOR);
        const cv::Mat right = cv::imread((input / "rgb_face_roi_r.png").string(), cv::IMREAD_COLOR);
        if (left.empty() || right.empty() || left.size() != right.size())
            throw std::runtime_error("cannot read equal-sized ROI images from Input");

        cv::Matx44f sac = cv::Matx44f::eye(), icp = cv::Matx44f::eye();
        cv::Matx33f colour = cv::Matx33f::eye();
        CameraCalibration leftCamera, rightCamera;
        int rgbWidth = 0;
        RoiMapping roi;
        recordedParameters(left.size(), sac, icp, colour, leftCamera, rightCamera, roi, rgbWidth);
        const OrganizedCloud leftCloud = readBinaryXyzPcd(input / "point_lift.pcd");
        const OrganizedCloud rightCloud = readBinaryXyzPcd(input / "point_right.pcd");
        fs::create_directories(output);
        for (const auto& entry : fs::directory_iterator(output))
            if (entry.is_regular_file()) fs::remove(entry.path());

        std::vector<cv::Vec3b> rawLeft, rawRight;
        std::vector<cv::Point> rawPositions;
        rawLeft.reserve(300); rawRight.reserve(300); rawPositions.reserve(300);
        const DepthFilterStats depth = selectDepthConsistentPairs(
            leftCloud, rightCloud, icp, leftCamera, rightCamera, roi, rgbWidth,
            left, right, output / "depth_consistency.csv", rawLeft, rawRight, rawPositions);
        std::vector<float> diffs(rawLeft.size());
        float meanDiff = 0.f;
        for (size_t i = 0; i < rawLeft.size(); ++i) {
            const cv::Vec3f d = cv::Vec3f(rawLeft[i]) - cv::Vec3f(rawRight[i]);
            diffs[i] = std::sqrt(d.dot(d)); meanDiff += diffs[i];
        }
        meanDiff /= std::max<size_t>(1, diffs.size());
        std::vector<cv::Vec3b> filteredLeft, filteredRight;
        std::vector<cv::Point> filteredPositions;
        for (size_t i = 0; i < diffs.size(); ++i)
            if (diffs[i] <= 2.f * std::max(meanDiff, 1e-6f)) {
                filteredLeft.push_back(rawLeft[i]); filteredRight.push_back(rawRight[i]);
                filteredPositions.push_back(rawPositions[i]);
            }

        const cv::Mat matrixImage = applyColourMatrix(left, colour);
        std::vector<cv::Vec3b> matrixSamples;
        matrixSamples.reserve(filteredLeft.size());
        for (const auto& c : filteredLeft) {
            const cv::Vec3f q = colour * cv::Vec3f(c[0] / 255.f, c[1] / 255.f, c[2] / 255.f);
            matrixSamples.emplace_back(cv::saturate_cast<uchar>(std::clamp(q[0], 0.f, 1.f) * 255.f), cv::saturate_cast<uchar>(std::clamp(q[1], 0.f, 1.f) * 255.f), cv::saturate_cast<uchar>(std::clamp(q[2], 0.f, 1.f) * 255.f));
        }
        const cv::Mat finalImage = reinhardL(matrixImage, bgrSamplesToLab(matrixSamples), bgrSamplesToLab(filteredRight));
        cv::imwrite((output / "corrected_left.png").string(), finalImage);
        cv::Mat rawStitched, matrixStitched, finalStitched;
        cv::hconcat(left, right, rawStitched); cv::hconcat(matrixImage, right, matrixStitched); cv::hconcat(finalImage, right, finalStitched);
        cv::imwrite((output / "stitched_raw.png").string(), rotateRight(rawStitched));
        cv::imwrite((output / "stitched_matrix.png").string(), rotateRight(matrixStitched));
        cv::imwrite((output / "stitched_reinhard.png").string(), rotateRight(finalStitched));

        Mesh mesh = readBinaryPly(input / "mesh.ply");
        cv::Mat stitched = rotateRight(finalStitched);
        std::vector<cv::Vec3b> vertexColours(mesh.vertices.size());
        const cv::Matx44f invIcp = icp.inv();
        const float seam = -60.f, seamWidth = 5.f;
        size_t leftInBounds = 0, rightInBounds = 0;
        for (size_t i = 0; i < mesh.vertices.size(); ++i) {
            const Point3 rightPoint = mesh.vertices[i];
            const Point3 leftPoint = transform(invIcp, rightPoint);
            bool leftValid = false, rightValid = false;
            const cv::Point2f leftUv = toStitchedUv(projectCamera(leftPoint, leftCamera, leftValid), true, roi, rgbWidth);
            const cv::Point2f rightUv = toStitchedUv(projectCamera(rightPoint, rightCamera, rightValid), false, roi, rgbWidth);
            if (leftValid && leftUv.x >= 0.f && leftUv.x < roi.height && leftUv.y >= 0.f && leftUv.y < roi.width) ++leftInBounds;
            if (rightValid && rightUv.x >= 0.f && rightUv.x < roi.height && rightUv.y >= roi.width && rightUv.y < 2.f * roi.width) ++rightInBounds;
            const cv::Vec3b lc = sampleClamped(stitched, leftUv);
            const cv::Vec3b rc = sampleClamped(stitched, rightUv);
            const float w = std::clamp((seam + seamWidth - rightPoint.y) / (2.f * seamWidth), 0.f, 1.f);
            cv::Vec3b c;
            for (int k = 0; k < 3; ++k) c[k] = cv::saturate_cast<uchar>(lc[k] * w + rc[k] * (1.f - w));
            if (rightPoint.y < seam - seamWidth) c = lc;
            if (rightPoint.y > seam + seamWidth) c = rc;
            vertexColours[i] = c;
        }
        writeObj(output / "colorfacemesh.obj", output / "colorfacemesh.mtl", mesh, vertexColours, meshNormals(mesh));

        const float rawRgb = meanDistance(filteredLeft, filteredRight);
        const float matrixRgb = meanDistance(matrixSamples, filteredRight);
        std::vector<cv::Vec3b> finalSamples;
        finalSamples.reserve(filteredLeft.size());
        for (const cv::Point& p : filteredPositions) finalSamples.push_back(finalImage.at<cv::Vec3b>(p));
        const float finalRgb = meanDistance(finalSamples, filteredRight);
        const cv::Mat rawLab = bgrSamplesToLab(filteredLeft), refLab = bgrSamplesToLab(filteredRight), matrixLab = bgrSamplesToLab(matrixSamples), finalLab = bgrSamplesToLab(finalSamples);
        std::ofstream metrics(output / "metrics.json");
        metrics << std::fixed << std::setprecision(6)
            << "{\n  \"roi_image_shape\": [" << left.cols << ", " << left.rows << "],\n"
            << "  \"sac_transform\": [\n    [" << sac(0,0) << ", " << sac(0,1) << ", " << sac(0,2) << ", " << sac(0,3) << "],\n    [" << sac(1,0) << ", " << sac(1,1) << ", " << sac(1,2) << ", " << sac(1,3) << "],\n    [" << sac(2,0) << ", " << sac(2,1) << ", " << sac(2,2) << ", " << sac(2,3) << "],\n    [0, 0, 0, 1]\n  ],\n"
            << "  \"icp_transform\": [\n    [" << icp(0,0) << ", " << icp(0,1) << ", " << icp(0,2) << ", " << icp(0,3) << "],\n    [" << icp(1,0) << ", " << icp(1,1) << ", " << icp(1,2) << ", " << icp(1,3) << "],\n    [" << icp(2,0) << ", " << icp(2,1) << ", " << icp(2,2) << ", " << icp(2,3) << "],\n    [0, 0, 0, 1]\n  ],\n"
            << "  \"projection\": \"SDK Point2RGBXY with separate left/right K/Kc/R/T embedded in demo.cpp\",\n"
            << "  \"camera_parameters\": {\"left\": " << cameraJson(leftCamera) << ", \"right\": " << cameraJson(rightCamera) << ", \"rgb_width\": " << rgbWidth << "},\n"
            << "  \"roi_mapping\": {\"lx\": " << roi.lx << ", \"ly\": " << roi.ly << ", \"rx\": " << roi.rx << ", \"ry\": " << roi.ry << ", \"width\": " << roi.width << ", \"height\": " << roi.height << "},\n"
            << "  \"projection_in_bounds\": {\"left\": " << leftInBounds << ", \"right\": " << rightInBounds << "},\n"
            << "  \"depth_consistency\": {\"left_candidates\": " << depth.leftCandidates
            << ", \"checked\": " << depth.checked << ", \"kept\": " << depth.kept
            << ", \"right_raw_valid\": " << depth.rightRawValid
            << ", \"right_raw_invalid\": " << depth.rightRawInvalid
            << ", \"right_pixel_out\": " << depth.rightPixelOut
            << ", \"rejected_pred_invalid\": " << depth.rejectedPredInvalid
            << ", \"rejected_right_pixel_out\": " << depth.rejectedRightPixelOut
            << ", \"rejected_no_observation\": " << depth.rejectedNoObservation
            << ", \"rejected_c2c\": " << depth.rejectedC2c
            << ", \"rejected_depth\": " << depth.rejectedDepth
            << ", \"half_window\": 1, \"max_c2c_distance\": 1.0, \"max_depth_difference\": 1.0},\n"
            << "  \"colour_matrix\": [[" << colour(0,0) << ", " << colour(0,1) << ", " << colour(0,2) << "], [" << colour(1,0) << ", " << colour(1,1) << ", " << colour(1,2) << "], [" << colour(2,0) << ", " << colour(2,1) << ", " << colour(2,2) << "]],\n"
            << "  \"sac_parameters\": {\"correspondence_randomness\": 3, \"number_of_samples\": 3, \"max_correspondence_distance\": 3.0, \"maximum_iterations\": 500},\n"
            << "  \"icp_parameters\": {\"maximum_iterations\": 10000, \"max_correspondence_distance\": 4.0, \"transformation_epsilon\": 1e-10, \"euclidean_fitness_epsilon\": 1e-4},\n"
            << "  \"sample_count\": " << filteredLeft.size() << ",\n"
            << "  \"raw_rgb_mean\": " << rawRgb << ",\n  \"matrix_rgb_mean\": " << matrixRgb << ",\n  \"reinhard_rgb_mean\": " << finalRgb << ",\n"
            << "  \"raw_lab_de76_mean\": " << meanLabDistance(rawLab, refLab) << ",\n  \"matrix_lab_de76_mean\": " << meanLabDistance(matrixLab, refLab) << ",\n  \"reinhard_lab_de76_mean\": " << meanLabDistance(finalLab, refLab) << "\n}\n";
        std::cout << "wrote " << mesh.vertices.size() << " vertices and " << mesh.faces.size() << " faces\n";
        std::cout << "Output: " << output << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FaceStitche demo failed: " << e.what() << '\n';
        return 1;
    }
}
