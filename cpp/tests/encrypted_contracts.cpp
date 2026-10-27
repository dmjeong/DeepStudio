#include "vision_runtime_c.h"
#include "model_crypto.h"
// A host can use the BW8 example and the native SDK in the same translation unit.
#include "../../example/cpp/model_crypto.h"
#include <iostream>
#include <algorithm>

static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class T> static void equal(const T* a, const T* b, size_t n) {
    if (n) require(a && b && std::memcmp(a, b, n*sizeof(T)) == 0, "Output bytes changed after encryption");
}
static void compare(const dv_result& a, const dv_result& b) {
    require(a.kind == b.kind && a.class_id == b.class_id && a.confidence == b.confidence &&
            a.probability_count == b.probability_count && a.detection_count == b.detection_count &&
            a.mask_width == b.mask_width && a.mask_height == b.mask_height && a.mask_stride_bytes == b.mask_stride_bytes &&
            a.anomaly_map_width == b.anomaly_map_width && a.anomaly_map_height == b.anomaly_map_height &&
            a.anomaly_map_stride_bytes == b.anomaly_map_stride_bytes && a.anomaly_score == b.anomaly_score &&
            a.anomaly_threshold == b.anomaly_threshold && a.anomalous == b.anomalous, "Result metadata changed");
    equal(a.probabilities, b.probabilities, a.probability_count);
    equal(a.mask, b.mask, size_t(a.mask_stride_bytes)*a.mask_height);
    equal(a.detections, b.detections, a.detection_count);
    equal(a.anomaly_map, b.anomaly_map, size_t(a.anomaly_map_width)*a.anomaly_map_height);
    if (a.class_name_utf8 || b.class_name_utf8)
        require(a.class_name_utf8 && b.class_name_utf8 && std::string(a.class_name_utf8) == b.class_name_utf8, "Class name changed");
}
using Session = std::unique_ptr<dv_session, decltype(&dv_close_session)>;
using Result = std::unique_ptr<dv_result, decltype(&dv_release_result)>;
using Context = std::unique_ptr<dv_image_context, decltype(&dv_release_image_context)>;

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Fixture directory required");
        const auto root = std::filesystem::u8path(argv[1]);
        auto key = dvs_crypto::ReadKey(root / "example-only.key");
        const auto list = nlohmann::json::parse(std::ifstream(root / "encrypted-tests.json"));
        for (const auto& row : list) {
            const std::string name = row.get<std::string>();
            const auto plain_path = (root / (name + ".json")).u8string();
            const auto encrypted_path = (root / (name + ".dvsenc")).u8string();
            dv_session *a = nullptr, *b = nullptr;
            // Default -1 preserves JSON verified optimizer and thread settings.
            require(dv_create_session(plain_path.c_str(), nullptr, &a) == DV_STATUS_OK, dv_last_error(nullptr));
            Session plain(a, dv_close_session);
            require(dv_create_session_encrypted(encrypted_path.c_str(), key.data(), 32, nullptr, &b) == DV_STATUS_OK, dv_last_error(nullptr));
            Session encrypted(b, dv_close_session);
            const auto config = nlohmann::json::parse(std::ifstream(root / (name + ".json")));
            for (int sample = 0; sample < 5; ++sample) {
                // Padded rows and different grayscale values exercise real preprocessing.
                std::vector<uint8_t> pixels(37*24, static_cast<uint8_t>(sample*63));
                dv_image_view image{sizeof(dv_image_view), DV_ABI_VERSION, pixels.data(), 32, 24, 1, 37};
                dv_result *x = nullptr, *y = nullptr;
                if (config.value("backend", std::string()) == "sam2") {
                    dv_image_context *p = nullptr, *q = nullptr;
                    require(dv_sam_encode(a, &image, &p) == DV_STATUS_OK, dv_last_error(a));
                    Context pc(p, dv_release_image_context);
                    require(dv_sam_encode(b, &image, &q) == DV_STATUS_OK, dv_last_error(b));
                    Context qc(q, dv_release_image_context);
                    const float point[]{16, 12}, box[]{4, 4, 24, 20}, mask[]{1,0,0,1};
                    const int32_t label[]{1};
                    dv_sam_prompt prompt{sizeof(dv_sam_prompt), DV_ABI_VERSION, point, label, 1,
                                         sample == 1 ? box : nullptr, sample == 2 ? mask : nullptr,
                                         sample == 2 ? 2u : 0u, sample == 2 ? 2u : 0u};
                    require(dv_sam_segment(a, p, &prompt, &x) == DV_STATUS_OK, dv_last_error(a));
                    require(dv_sam_segment(b, q, &prompt, &y) == DV_STATUS_OK, dv_last_error(b));
                } else {
                    require(dv_infer(a, &image, &x) == DV_STATUS_OK, dv_last_error(a));
                    require(dv_infer(b, &image, &y) == DV_STATUS_OK, dv_last_error(b));
                }
                Result xp(x, dv_release_result), yp(y, dv_release_result);
                compare(*x, *y);
            }
            std::cout << "PASS byte-exact: " << name << '\n';
        }
        const auto path = (root / "classify.dvsenc").u8string();
        key[0] ^= 1;
        dv_session* rejected = nullptr;
        require(dv_create_session_encrypted(path.c_str(), key.data(), 32, nullptr, &rejected) != DV_STATUS_OK && !rejected,
                "Wrong key accepted");
        key[0] ^= 1;
        const auto invalid_list = nlohmann::json::parse(std::ifstream(root / "encrypted-invalid-tests.json"));
        for (const auto& bad : invalid_list) {
            const auto broken = (root / (bad.get<std::string>() + ".dvsenc")).u8string();
            require(dv_create_session_encrypted(broken.c_str(), key.data(), 32, nullptr, &rejected) != DV_STATUS_OK && !rejected,
                    "Malformed encrypted model accepted");
        }
        require(dv_create_session_encrypted(path.c_str(), key.data(), 31, nullptr, &rejected) == DV_STATUS_INVALID_ARGUMENT,
                "Short key accepted");
        dvs_crypto::Wipe(key.data(), key.size());
        std::cout << "PASS: encrypted C ABI + SAM2 + authentication negative tests\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
