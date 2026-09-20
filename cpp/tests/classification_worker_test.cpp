#include "classification_worker.h"
#include <atomic>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Call> bool Throws(Call call) {
    try { call(); } catch (const std::exception&) { return true; }
    return false;
}

int main(int argc, char** argv) {
    try {
        Require(argc == 2, "Fixture directory required.");
        const auto main_thread = std::this_thread::get_id();
        std::promise<void> entered, release;
        auto entered_future = entered.get_future();
        auto gate = release.get_future().share();
        std::atomic<int> calls{0};
        std::atomic<bool> background{false};
        ClassificationWorker worker([&] {
            return ClassificationWorker::Predictor([&](const cv::Mat& image) {
                background = std::this_thread::get_id() != main_thread;
                if (calls.fetch_add(1) == 0) { entered.set_value(); gate.wait(); }
                ClassifyResult result;
                result.class_id = image.at<uchar>(0,0);
                return result;
            });
        }, 1);
        worker.Ready().get();
        cv::Mat frame(2,3,CV_8UC1,cv::Scalar(7));
        auto first = worker.Submit(1, frame);
        frame.setTo(99);
        const bool started = entered_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        if (!started) { release.set_value(); throw std::runtime_error("Worker failed to start."); }
        auto second = worker.Submit(2, cv::Mat(2,3,CV_8UC1,cv::Scalar(11)));
        const bool full = Throws([&] { worker.Submit(3, frame); });
        release.set_value();
        worker.Stop(); // Must drain both accepted frames and join.
        const auto a = first.get(), b = second.get();
        Require(full && background && calls == 2, "Queue bound or worker thread contract failed.");
        Require(a.frame_id == 1 && a.prediction.class_id == 7 && b.frame_id == 2 && b.prediction.class_id == 11,
                "Frame copy, FIFO order or result ownership failed.");
        Require(a.queue_ms >= 0 && a.end_to_end_ms >= a.queue_ms, "Invalid background timings.");
        Require(Throws([&] { worker.Submit(4, frame); }), "Submit after stop accepted.");
        worker.Stop();

        ClassificationWorker errors([] {
            return ClassificationWorker::Predictor([](const cv::Mat& image) {
                if (image.depth() != CV_8U) throw std::invalid_argument("bad frame");
                ClassifyResult result; result.class_id = 0; return result;
            });
        });
        errors.Ready().get();
        auto failed = errors.Submit(1, cv::Mat(2,3,CV_32FC1,cv::Scalar(0)));
        Require(Throws([&] { failed.get(); }), "Inference error not delivered through future.");
        Require(errors.Submit(2, frame).get().prediction.class_id == 0, "One bad frame killed the worker.");
        Require(Throws([&] { errors.Submit(3, cv::Mat{}); }), "Empty frame accepted.");
        errors.Stop();

        std::promise<void> fail_start;
        auto fail_gate = fail_start.get_future().share();
        ClassificationWorker startup([fail_gate]() -> ClassificationWorker::Predictor {
            fail_gate.wait(); throw std::runtime_error("expected initialization failure");
        });
        auto pending = startup.Submit(5, frame);
        fail_start.set_value();
        Require(Throws([&] { startup.Ready().get(); }), "Startup failure not propagated.");
        Require(Throws([&] { pending.get(); }), "Queued future stranded after startup failure.");
        startup.Stop();
        Require(Throws([&] { startup.Submit(6, frame); }), "Failed worker accepted a frame.");

        // Concurrent producers receive independent futures; no accepted frame is lost.
        ClassificationWorker concurrent([] {
            return ClassificationWorker::Predictor([](const cv::Mat&) { ClassifyResult r; r.class_id=1; return r; });
        }, 200);
        concurrent.Ready().get();
        std::vector<std::vector<std::future<BackgroundClassification>>> jobs(4);
        std::vector<std::thread> producers;
        for (uint64_t p=0;p<4;++p) producers.emplace_back([&,p] {
            for(uint64_t i=0;i<40;++i) jobs[p].push_back(concurrent.Submit(p*100+i,frame));
        });
        for(auto& producer:producers) producer.join();
        concurrent.Stop();
        for(uint64_t p=0;p<4;++p) for(uint64_t i=0;i<40;++i)
            Require(jobs[p][i].get().frame_id==p*100+i,"Concurrent submission lost a result.");

        // Also execute the actual exported fixture through the public model constructor.
        const auto path = (std::filesystem::u8path(argv[1]) / "classify.json").u8string();
        std::vector<std::string> runtimes{"onnxruntime"};
#ifdef VISION_WITH_OPENVINO
        runtimes.push_back("openvino");
#endif
        for (const auto& runtime : runtimes) {
            VisionInference direct;
            Require(direct.InitializeFromJson(path, runtime, 1), "Real model load failed.");
            const auto expected = direct.Classify(frame);
            ClassificationWorker actual(path, runtime, -1, 2, 2); // Preserve JSON thread count.
            actual.Ready().get();
            const auto got = actual.Submit(9, frame).get().prediction;
            Require(got.class_id == expected.class_id, "Background class mismatch.");
            for(size_t i=0;i<got.probabilities.size();++i)
                Require(std::abs(got.probabilities[i]-expected.probabilities[i])<1e-5,"Background probabilities mismatch.");
            actual.Stop();
        }
        const auto numerical_path = (std::filesystem::u8path(argv[1]) / "runtime_optimization.json").u8string();
        ClassificationWorker numerical(numerical_path);
        numerical.Ready().get();
        const auto stable = numerical.Submit(10, cv::Mat(2, 3, CV_8UC1, cv::Scalar(255))).get().prediction;
        Require(std::abs(stable.confidence - 1.0 / (1.0 + std::exp(-2.0))) < 1e-6,
                "Background worker ignored the verified runtime settings.");
        numerical.Stop();
        std::cout << "Background ownership, FIFO, errors, shutdown, concurrent producers and actual runtime contracts passed.\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
