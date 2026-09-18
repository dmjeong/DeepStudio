#include "classification_worker.h"
#include <stdexcept>
#include <utility>

namespace {
ClassificationWorker::Factory ModelFactory(std::string path, std::string runtime, int threads, int warmup)
{
    if (threads < -1 || warmup < 0) throw std::invalid_argument("Threads must be >= -1 and warmup nonnegative.");
    return [path = std::move(path), runtime = std::move(runtime), threads, warmup]() {
        auto model = std::make_shared<VisionInference>();
        if (!model->InitializeFromJson(path, runtime, threads))
            throw std::runtime_error("Background model initialization failed: " + path);
        const auto& config = model->GetConfig();
        if (config.task != "classify") throw std::invalid_argument("Classification model required.");
        cv::Mat image(std::max(config.input_height, config.crop_height),
                      std::max(config.input_width, config.crop_width),
                      CV_MAKETYPE(CV_8U, config.input_channels), cv::Scalar::all(127));
        for (int i = 0; i < warmup; ++i) model->Classify(image);
        return ClassificationWorker::Predictor([model](const cv::Mat& pixels) { return model->Classify(pixels); });
    };
}
}

ClassificationWorker::ClassificationWorker(const std::string& path, const std::string& runtime,
                                           int threads, size_t capacity, int warmup)
    : ClassificationWorker(ModelFactory(path, runtime, threads, warmup), capacity) {}

ClassificationWorker::ClassificationWorker(Factory factory, size_t capacity)
    : m_capacity(capacity), m_ready(m_ready_promise.get_future().share())
{
    if (!capacity || !factory) throw std::invalid_argument("A factory and positive queue capacity are required.");
    m_thread = std::thread([this, factory = std::move(factory)]() mutable { Run(std::move(factory)); });
}

ClassificationWorker::~ClassificationWorker() { Stop(); }

std::future<BackgroundClassification> ClassificationWorker::Submit(uint64_t id, const cv::Mat& image)
{
    const auto submitted = std::chrono::steady_clock::now();
    if (image.empty()) throw std::invalid_argument("Cannot submit an empty image.");
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_stopping) throw std::runtime_error("Classification worker is stopped.");
    if (m_jobs.size() >= m_capacity) throw std::runtime_error("Classification queue is full.");
    Job job{id, image.clone(), submitted, {}};
    auto future = job.promise.get_future();
    m_jobs.push_back(std::move(job));
    m_changed.notify_one();
    return future;
}

void ClassificationWorker::Stop()
{
    std::lock_guard<std::mutex> stop_lock(m_stop_mutex);
    if (m_thread.joinable() && std::this_thread::get_id() == m_thread.get_id())
        throw std::logic_error("Stop must be called from the owning thread.");
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stopping = true;
    }
    m_changed.notify_all();
    if (m_thread.joinable()) m_thread.join();
}

void ClassificationWorker::Run(Factory factory)
{
    Predictor predict;
    try
    {
        predict = factory();
        if (!predict) throw std::runtime_error("Predictor factory returned an empty function.");
        m_ready_promise.set_value();
    }
    catch (...)
    {
        const auto error = std::current_exception();
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stopping = true;
        m_ready_promise.set_exception(error);
        for (auto& job : m_jobs) job.promise.set_exception(error);
        m_jobs.clear();
        return;
    }
    for (;;)
    {
        Job job;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_changed.wait(lock, [this] { return m_stopping || !m_jobs.empty(); });
            if (m_jobs.empty()) break;
            job = std::move(m_jobs.front());
            m_jobs.pop_front();
        }
        const auto started = std::chrono::steady_clock::now();
        try
        {
            auto prediction = predict(job.image);
            const auto finished = std::chrono::steady_clock::now();
            BackgroundClassification result;
            result.frame_id = job.id;
            result.prediction = std::move(prediction);
            result.queue_ms = std::chrono::duration<double, std::milli>(started - job.submitted).count();
            result.end_to_end_ms = std::chrono::duration<double, std::milli>(finished - job.submitted).count();
            job.promise.set_value(std::move(result));
        }
        catch (...) { job.promise.set_exception(std::current_exception()); }
    }
}
