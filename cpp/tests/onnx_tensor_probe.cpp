// Numerical regression helper, not an image preprocessing example.
// onnx_tensor_probe config.json|model.dvsenc input.f32 output.f32 key
// Input is one preprocessed NCHW FP32 tensor matching the deployment JSON.
#include "model_crypto.h"
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::invalid_argument("Expected config, input.f32, output.f32, key");
        const auto path = std::filesystem::u8path(argv[1]);
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "tensor-regression");
        env.DisableTelemetryEvents();
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        nlohmann::json config;
        std::unique_ptr<Ort::Session> session;
        if (path.extension() == ".dvsenc") {
            auto key = dvs_crypto::ReadKey(std::filesystem::u8path(argv[4]));
            dvs_crypto::Package package(path, key);
            dvs_crypto::Wipe(key.data(), key.size());
            config = package.Config();
            session = package.Session(env, config.at("model_path").get<std::string>(), options);
        } else {
            std::ifstream file(path);
            file >> config;
            const auto model = path.parent_path() / std::filesystem::u8path(config.at("model_path").get<std::string>());
            session = std::make_unique<Ort::Session>(env, model.c_str(), options);
        }
        std::array<int64_t, 4> shape{{1, config.at("input_channels").get<int64_t>(),
                                     config.at("input_height").get<int64_t>(),
                                     config.at("input_width").get<int64_t>()}};
        size_t count = 1;
        for (auto size : shape) {
            if (size <= 0 || size > 16384 || count > 100000000 / static_cast<size_t>(size))
                throw std::invalid_argument("Input shape too large or invalid");
            count *= static_cast<size_t>(size);
        }
        std::vector<float> data(count);
        std::ifstream file(std::filesystem::u8path(argv[2]), std::ios::binary);
        if (!file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(float))) ||
            file.peek() != std::char_traits<char>::eof()) throw std::invalid_argument("Input tensor length mismatch");
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(memory, data.data(), count, shape.data(), shape.size());
        const auto input = config.at("input_name").get<std::string>();
        const auto output = config.at("output_name").get<std::string>();
        const char* in[] = {input.c_str()};
        const char* out[] = {output.c_str()};
        auto result = session->Run(Ort::RunOptions{nullptr}, in, &tensor, 1, out, 1);
        const auto info = result[0].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Public output must be FP32");
        const auto size = info.GetElementCount();
        std::ofstream dest(std::filesystem::u8path(argv[3]), std::ios::binary);
        if (!dest.write(reinterpret_cast<const char*>(result[0].GetTensorData<float>()),
                        static_cast<std::streamsize>(size * sizeof(float))))
            throw std::runtime_error("Cannot write output tensor");
        std::cout << "ORT " << OrtGetApiBase()->GetVersionString() << " outputs " << size << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
