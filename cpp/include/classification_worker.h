#pragma once
#include "vision_inference.h"
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

struct BackgroundClassification
{
    uint64_t frame_id = 0;
    ClassifyResult prediction;
    double queue_ms = 0;       // Submit entry through worker dequeue (includes owned frame copy)
    double end_to_end_ms = 0;  // Submit entry through inference completion, excludes caller wakeup
};

// In-process component, not an OS service. One worker exclusively owns the model.
// Capacity bounds waiting frames; one additional frame may be executing.
// Stop producers before destruction. Shutdown drains every accepted frame and joins.
class ClassificationWorker
{
public:
    using Predictor = std::function<ClassifyResult(const cv::Mat&)>;
    using Factory = std::function<Predictor()>;

    // num_threads=-1 reads num_threads from JSON; 0 uses the runtime's automatic count.
    ClassificationWorker(const std::string& config_path, const std::string& runtime = "onnxruntime",
                         int num_threads = 4, size_t max_pending = 2, int warmup = 10);
    // Factory executes on the worker thread; useful for embedding an existing predictor.
    explicit ClassificationWorker(Factory factory, size_t max_pending = 2);
    ~ClassificationWorker();
    ClassificationWorker(const ClassificationWorker&) = delete;
    ClassificationWorker& operator=(const ClassificationWorker&) = delete;

    std::shared_future<void> Ready() const { return m_ready; }
    // Clones image before returning. Rejects empty input, a full queue, or shutdown.
    std::future<BackgroundClassification> Submit(uint64_t frame_id, const cv::Mat& image);
    void Stop(); // Call on an owner/producer thread, never on the worker itself.

private:
    struct Job
    {
        uint64_t id;
        cv::Mat image;
        std::chrono::steady_clock::time_point submitted;
        std::promise<BackgroundClassification> promise;
    };
    void Run(Factory factory);
    const size_t m_capacity;
    std::mutex m_mutex, m_stop_mutex;
    std::condition_variable m_changed;
    std::deque<Job> m_jobs;
    bool m_stopping = false;
    std::promise<void> m_ready_promise;
    std::shared_future<void> m_ready;
    std::thread m_thread;
};
