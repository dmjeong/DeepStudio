// Minimal example of integrating inference as one function in an existing C++ app.
#include "classification_worker.h"
#include <iostream>

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: efficientnet_background_demo model.json image.png [onnxruntime|openvino]\n";
        return 1;
    }
    try {
        cv::setNumThreads(1);
        ClassificationWorker inference(argv[1], argc == 4 ? argv[3] : "onnxruntime", -1, 2, 30);
        // Loading/compilation/warmup also execute on the worker. Your UI remains here.
        size_t main_loop_ticks = 0;
        auto ready = inference.Ready();
        while (ready.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            ++main_loop_ticks; // Replace with the application's event loop, not inference.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ready.get(); // Propagates initialization errors; model is now warm.
        auto frame = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
        auto pending = inference.Submit(42, frame);
        frame.release(); // Submit owns a copy, so the caller can recycle its camera buffer.
        while (pending.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            ++main_loop_ticks;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const auto result = pending.get(); // Receive result/errors on the caller/UI thread.
        std::cout << "frame=" << result.frame_id << " class=" << result.prediction.class_name
                  << " confidence=" << result.prediction.confidence
                  << " pipeline_ms=" << result.prediction.inference_ms
                  << " queue_ms=" << result.queue_ms << " submit_to_result_ms=" << result.end_to_end_ms
                  << " main_loop_ticks=" << main_loop_ticks << '\n';
        inference.Stop(); // Drains accepted frames and joins; destructor also does this.
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
