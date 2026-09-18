#pragma once

/* Stable C ABI for the in-process C++17 ONNX runtime. */
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(VISION_RUNTIME_BUILD)
#  define DV_API __declspec(dllexport)
#elif defined(_WIN32)
#  define DV_API __declspec(dllimport)
#else
#  define DV_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DV_ABI_VERSION 1u

typedef enum dv_status {
    DV_STATUS_OK = 0,
    DV_STATUS_INVALID_ARGUMENT = 1,
    DV_STATUS_NOT_READY = 2,
    DV_STATUS_RUNTIME_ERROR = 3,
    DV_STATUS_OUT_OF_MEMORY = 4,
    DV_STATUS_INTERNAL_ERROR = 5
} dv_status;

typedef enum dv_result_kind {
    DV_RESULT_CLASSIFICATION = 1,
    DV_RESULT_SEGMENTATION = 2,
    DV_RESULT_DETECTION = 3,
    DV_RESULT_ANOMALY = 4
} dv_result_kind;

typedef struct dv_detection {
    float x1;
    float y1;
    float x2;
    float y2;
    int32_t class_id;
    float confidence;
} dv_detection;

typedef struct dv_session_options {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* runtime_utf8; /* NULL means onnxruntime. */
    int32_t num_threads;      /* -1 means use the JSON/default value. */
} dv_session_options;

typedef struct dv_image_view {
    uint32_t struct_size;
    uint32_t abi_version;
    const uint8_t* data;
    int32_t width;
    int32_t height;
    int32_t channels;          /* 1, 3, or 4; BGR/BGRA for color input. */
    int32_t stride_bytes;      /* 0 means tightly packed. */
} dv_image_view;

typedef struct dv_sam_prompt {
    uint32_t struct_size;
    uint32_t abi_version;
    const float* point_coords_xy; /* point_count pairs in original image pixels. */
    const int32_t* point_labels;  /* 1 positive, 0 negative, -1 no point, 2/3 box corners. */
    uint32_t point_count;
    const float* box_xyxy;        /* optional four values in original image pixels. */
    const float* mask_input;      /* optional row-major FP32 low-resolution mask. */
    uint32_t mask_width;
    uint32_t mask_height;
} dv_sam_prompt;

typedef struct dv_result {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t kind;
    int32_t class_id;
    float confidence;
    float* probabilities;
    uint32_t probability_count;
    char* class_name_utf8;
    uint8_t* mask;
    uint32_t mask_width;
    uint32_t mask_height;
    uint32_t mask_stride_bytes;
    uint32_t mask_classes;
    double total_ms;
    double preprocess_ms;
    double model_ms;
    double postprocess_ms;
    dv_detection* detections;
    uint32_t detection_count;
    float anomaly_score;
    float anomaly_threshold;
    uint32_t anomalous;
    float* anomaly_map;
    uint32_t anomaly_map_width;
    uint32_t anomaly_map_height;
    uint32_t anomaly_map_stride_bytes;
} dv_result;

typedef struct dv_session dv_session;
typedef struct dv_image_context dv_image_context;

DV_API uint32_t dv_abi_version(void);
DV_API const char* dv_status_name(dv_status status);
DV_API dv_status dv_create_session(const char* config_path_utf8,
                                   const dv_session_options* options,
                                   dv_session** out_session);
DV_API dv_status dv_infer(dv_session* session, const dv_image_view* image,
                          dv_result** out_result);
DV_API dv_status dv_sam_encode(dv_session* session, const dv_image_view* image,
                               dv_image_context** out_context);
DV_API dv_status dv_sam_segment(dv_session* session, const dv_image_context* context,
                                const dv_sam_prompt* prompt, dv_result** out_result);
DV_API const char* dv_last_error(const dv_session* session);
DV_API void dv_release_result(dv_result* result);
DV_API void dv_release_image_context(dv_image_context* context);
DV_API void dv_close_session(dv_session* session);

#ifdef __cplusplus
}
#endif
