#include "csv_model_manager.h"
#include "sevision_csv_example.h"
#include "../example_paths.h"
#include <iostream>

namespace {
using namespace std::chrono_literals;
using dvs_bw8::CsvModelManager;
void Require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class Predicate> void Wait(Predicate check) {
    const auto end = std::chrono::steady_clock::now() + 8s;
    while (!check()) {
        if (std::chrono::steady_clock::now() > end) throw std::runtime_error("Filename CSV test timed out");
        std::this_thread::sleep_for(10ms);
    }
}
template<class Function> void Reject(Function function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    Require(rejected, message);
}
struct Temp {
    std::filesystem::path path;
    Temp() {
        for (int i = 0; i < 100; ++i) {
            auto candidate = std::filesystem::temp_directory_path() /
                ("dvs-filename-csv-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            if (std::filesystem::create_directory(candidate)) { path = candidate; return; }
        }
        throw std::runtime_error("Cannot create test directory");
    }
    ~Temp() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};
void Replace(const std::filesystem::path& from, const std::filesystem::path& to) {
#ifdef _WIN32
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const auto error = GetLastError();
        if ((error != ERROR_SHARING_VIOLATION && error != ERROR_LOCK_VIOLATION && error != ERROR_ACCESS_DENIED)
                || std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("Cannot publish file");
        std::this_thread::sleep_for(10ms);
    }
#else
    std::filesystem::rename(from, to);
#endif
}
void Text(const std::filesystem::path& path, const std::string& text) {
    auto pending = path; pending += ".pending";
    { std::ofstream stream(pending, std::ios::binary);
      stream << text; if (!stream) throw std::runtime_error("Cannot write CSV"); }
    Replace(pending, path);
}
void Model(const std::filesystem::path& source, const std::filesystem::path& path) {
    auto pending = path; pending += ".pending";
    std::filesystem::copy_file(source, pending, std::filesystem::copy_options::overwrite_existing);
    Replace(pending, path);
}
std::string Top(const std::string& hour) { return "sevision_ai_261126_" + hour + ".dvsenc"; }
std::string Side(const std::string& hour) { return "sevision_side_261126_" + hour + ".dvsenc"; }
std::string Rows(const std::string& top = "16H", const std::string& side = "15H", bool reverse = false) {
    const auto a = "0,TOP,classify," + Top(top) + "\r\n";
    const auto b = "7,SIDE,classify," + Side(side) + "\r\n";
    return "\xef\xbb\xbfindex,camtype,task,name\r\n" + (reverse ? b + a : a + b);
}
int Revision(CsvModelManager& manager, int index, const std::string& cam) {
    auto model = manager.Snapshot(index, cam);
    return model ? std::stoi(model->ModelVersion().substr(21, 1)) : 0;
}
struct Readers {
    std::atomic<bool> running{true};
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    explicit Readers(CsvModelManager& manager) : Readers([&manager](int i) {
        return manager.Snapshot(i % 2 ? 7 : 0, i % 2 ? "SIDE" : "TOP");
    }) {}
    explicit Readers(CSeVision& vision) : Readers([&vision](int i) {
        return std::atomic_load(&vision.thd_classifier.at(i % 2 ? 7 : 0));
    }) {}
    explicit Readers(std::function<std::shared_ptr<dvs_bw8::Classifier>(int)> snapshot) {
        for (int i = 0; i < 4; ++i) threads.emplace_back([this, snapshot, i] {
            const std::vector<uint8_t> frame(240 * 224, 255);
            while (running.load()) {
                try {
                    auto model = snapshot(i);
                    if (!model) throw std::runtime_error("Missing live model");
                    auto result = model->InferBW8(frame.data(), 224, 224, 240);
                    if (result.class_id != 0 || result.class_name != "white_v" + model->ModelVersion().substr(21, 1)) ++errors;
                } catch (...) { ++errors; }
                std::this_thread::yield();
            }
        });
    }
    void Stop() { running.store(false); for (auto& thread : threads) if (thread.joinable()) thread.join(); }
    ~Readers() { Stop(); }
};
} // namespace

int main() {
    try {
        using namespace dvs_bw8::filename_csv_detail;
        Spec spec;
        spec.name = Top("16H"); ParseFilename(spec);
        Require(spec.family == "sevision_ai" && spec.version == "20261126_160000_000000", "Hour parsing failed");
        spec.name = Top("163025"); ParseFilename(spec);
        Require(spec.version == "20261126_163025_000000", "Seconds parsing failed");
        spec.name = "model_000229_00H.dvsenc"; ParseFilename(spec);
        for (const std::string name : {"x_260230_16H.dvsenc", "x_260229_16H.dvsenc", "x_261126_24H.dvsenc",
                "x_261126_166000.dvsenc", "x_261126_1H.dvsenc", "x_261126_16H.onnx",
                "../x_261126_16H.dvsenc", "D:\\x_261126_16H.dvsenc", "_261126_16H.dvsenc"})
            Reject([&] { spec.name = name; ParseFilename(spec); }, "Bad filename accepted");

        const auto assets = example::ExecutableDirectory() / "assets";
        Temp temp;
        const auto csv = temp.path / "models.csv";
        const auto directory = temp.path / std::filesystem::u8path(u8"모델 폴더");
        std::filesystem::create_directory(directory);
        Text(csv, "name,task,index,camtype\r\n\"검사, 모델_261126_16H.dvsenc\",classify,0,TOP\r\n");
        Require(Read(csv).at(0).name == u8"검사, 모델_261126_16H.dvsenc", "CSV UTF-8/quoting/column order failed");
        for (const auto& invalid : {
                std::string("index,camtype,task,name\n-1,TOP,classify,") + Top("16H"),
                std::string("index,camtype,task,name\n2147483648,TOP,classify,") + Top("16H"),
                std::string("index,camtype,task,name\n0,TOP,segment,") + Top("16H"),
                Rows() + "0,TOP,classify," + Top("16H"),
                std::string("index,camtype,task,name\n0,TOP,classify,\"unfinished"),
                std::string("index,camtype,task,name\n")}) {
            Text(csv, invalid); Reject([&] { Read(csv); }, "Bad CSV accepted");
        }

        Model(assets / "reload/model.dvsenc", directory / Top("16H"));
        Model(assets / "reload/model.dvsenc", directory / Side("15H"));
        Text(csv, Rows());
        // Initial startup failure publishes none of the models, even if the first loaded.
        CsvModelManager failed(csv, directory, [&](int index) {
            if (index == 7) throw std::runtime_error("Key missing");
            return dvs_crypto::ReadKey(assets / "example-only.key");
        }, 50ms);
        Reject([&] { failed.Start(); }, "Partial initial load succeeded");
        Require(!failed.Snapshot(0, "TOP"), "Partial initial table published");

        std::atomic<int> reads{0};
        std::atomic<bool> wrong_key{false};
        CsvModelManager manager(csv, std::filesystem::u8path(u8"모델 폴더"), [&](int) {
            ++reads;
            auto key = dvs_crypto::ReadKey(assets / "example-only.key");
            if (wrong_key.load()) key[0] ^= 1;
            return key;
        }, 50ms);
        manager.Start();
        Require(Revision(manager, 0, "TOP") == 1 && Revision(manager, 7, "SIDE") == 1, "Initial models failed");
        Require(!manager.Snapshot(0, "SIDE") && !manager.Snapshot(999, "TOP") && !manager.Snapshot(0, "TOP", "segment"), "Wrong camera/task/index accepted");
        const auto mtime = std::filesystem::last_write_time(csv);
        Text(csv, Rows("16H", "15H", true));
        std::filesystem::last_write_time(csv, mtime);
        std::this_thread::sleep_for(160ms);
        Require(reads.load() == 2, "Unchanged/reordered rows decrypted again");
        Readers readers(manager);
        auto old = manager.Snapshot(0, "TOP");
        std::weak_ptr<dvs_bw8::Classifier> old_weak = old;

        Text(csv, Rows("17H")); // Arriving file: retry the SAME CSV after failure.
        std::filesystem::last_write_time(csv, mtime);
        Wait([&] { return manager.LastStatus().find("Cannot open encrypted") != std::string::npos; });
        Require(Revision(manager, 0, "TOP") == 1, "Missing file dropped active model");
        Model(assets / "reload/v2.dvsenc", directory / Top("17H"));
        Wait([&] { return Revision(manager, 0, "TOP") == 2; });
        Require(old->ModelVersion().back() == '1', "In-flight model changed");
        const int after_update = reads.load();
        // Same filename is deliberately not a change trigger, even if its bytes change.
        Model(assets / "reload/v4.dvsenc", directory / Top("17H"));
        std::this_thread::sleep_for(160ms);
        Require(Revision(manager, 0, "TOP") == 2 && reads.load() == after_update, "Same name reloaded");

        Model(assets / "reload/v3.dvsenc", directory / Top("18H"));
        Model(assets / "reload/v2.dvsenc", directory / Side("16H"));
        Text(csv, Rows("18H", "16H"));
        Wait([&] { return Revision(manager, 7, "SIDE") == 2 && manager.LastStatus().find("Waiting for previous") != std::string::npos; });
        Require(Revision(manager, 0, "TOP") == 2, "Third model created while previous model held");
        old.reset();
        Wait([&] { return Revision(manager, 0, "TOP") == 3; });
        Wait([&] { return old_weak.expired(); });

        auto corrupt = dvs_crypto::ReadFile(assets / "reload/v4.dvsenc"); corrupt.back() ^= 1;
        Text(directory / Top("19H"), std::string(corrupt.begin(), corrupt.end()));
        Text(csv, Rows("19H", "16H"));
        Wait([&] { return manager.LastStatus().find("authentication failed") != std::string::npos; });
        Require(Revision(manager, 0, "TOP") == 3, "Tampered model published");
        Model(assets / "reload/bad-shape-v4.dvsenc", directory / Top("19H"));
        Wait([&] { return manager.LastStatus().find("Tensor shape/type") != std::string::npos; });
        Require(Revision(manager, 0, "TOP") == 3, "Bad model published");
        wrong_key.store(true);
        Model(assets / "reload/v4.dvsenc", directory / Top("19H"));
        Wait([&] { return manager.LastStatus().find("authentication failed") != std::string::npos; });
        Require(Revision(manager, 0, "TOP") == 3, "Wrong key accepted");
        wrong_key.store(false);
        Wait([&] { return Revision(manager, 0, "TOP") == 4; });

        Text(csv, Rows("18H", "16H"));
        Wait([&] { return manager.LastStatus().find("Ignored same/older") != std::string::npos; });
        Require(Revision(manager, 0, "TOP") == 4, "Downgrade accepted");
        Text(csv, Rows("19H", "16H") + "0,TOP,classify," + Top("19H") + "\n");
        Wait([&] { return manager.LastStatus().find("Duplicate CSV index") != std::string::npos; });
        Text(csv, "index,camtype,task,name\n0,TOP,classify," + Top("19H") + "\n");
        Wait([&] { return manager.LastStatus().find("Missing active index") != std::string::npos; });
        auto changed_route = Rows("19H", "16H"); changed_route.replace(changed_route.find("TOP"), 3, "OTHER");
        Text(csv, changed_route);
        Wait([&] { return manager.LastStatus().find("restart required") != std::string::npos; });
        Require(Revision(manager, 0, "TOP") == 4 && Revision(manager, 7, "SIDE") == 2, "Invalid CSV changed routing/models");

        // Add a new index at runtime; it need not be contiguous.
        Text(csv, Rows("19H", "16H") + "90,AUX,classify," + Side("16H") + "\n");
        Wait([&] { return Revision(manager, 90, "AUX") == 2; });
        readers.Stop();
        Require(readers.errors.load() == 0, "Concurrent inference corrupted during reload");
        manager.Stop();

        // Superseded initial request never becomes visible.
        Text(csv, Rows());
        CsvModelManager superseded(csv, directory, [&](int index) {
            if (index == 7) Text(csv, Rows("18H"));
            return dvs_crypto::ReadKey(assets / "example-only.key");
        }, 50ms);
        Reject([&] { superseded.Start(); }, "Superseded startup accepted");
        Require(!superseded.Snapshot(0, "TOP"), "Superseded initial model published");

        // User-owned periodic thread: only InitializeModel / ChangeModel.
        // Direct atomic_load on fixed map entries, with no manager or watcher.
        const auto key_file = temp.path / "manual.key";
        std::filesystem::copy_file(assets / "example-only.key", key_file);
        std::string error;
        CSeVision manual;
        Require(!manual.ChangeModel(error), "Change before initialization accepted");
        Text(csv, Rows("16H", "23H")); // First model valid, second absent.
        Require(!manual.InitializeModel(csv, directory, key_file, error) && manual.thd_classifier.empty(),
                "Manual initialization published partial models");
        Text(csv, Rows());
        Require(manual.InitializeModel(csv, directory, key_file, error), "Manual initialization failed");
        const auto revision = [&] {
            return std::atomic_load(&manual.thd_classifier.at(0))->ModelVersion().back();
        };
        Require(!manual.InitializeModel(csv, directory, key_file, error), "Repeated initialization accepted");
        // Unchanged CSV must work without opening the model/key again.
        std::filesystem::remove(key_file);
        Require(manual.ChangeModel(error) && error.empty(), "Unchanged CSV read key again");
        std::filesystem::copy_file(assets / "example-only.key", key_file);
        auto held = std::atomic_load(&manual.thd_classifier.at(0));
        std::weak_ptr<dvs_bw8::Classifier> manual_old = held;
        Readers manual_readers(manual);
        Model(assets / "reload/v2.dvsenc", directory / Top("17H"));
        Text(csv, Rows("17H"));
        std::this_thread::sleep_for(100ms);
        Require(revision() == '1', "Manual example unexpectedly has an automatic watcher");
        Require(manual.ChangeModel(error) && revision() == '2', "Manual swap failed");
        Require(held->ModelVersion().back() == '1', "Manual in-flight model changed");
        Text(csv, Rows("18H"));
        Require(manual.ChangeModel(error) && revision() == '2', "Manual retirement limit failed");
        held.reset();
        Wait([&] { return manual.ChangeModel(error) && revision() == '3'; });
        Require(manual_old.expired(), "Manual previous model was not reclaimed");
        manual_readers.Stop();
        Require(manual_readers.errors.load() == 0, "Manual shared_ptr swap corrupted inference");

        std::filesystem::remove(directory / Top("19H"));
        Text(csv, Rows("19H"));
        Require(!manual.ChangeModel(error) && revision() == '3', "Manual missing file dropped active model");
        Model(assets / "reload/bad-shape-v4.dvsenc", directory / Top("19H"));
        Require(!manual.ChangeModel(error) && revision() == '3', "Manual invalid shape published");
        Model(assets / "reload/v4.dvsenc", directory / Top("19H"));
        auto bad_key = dvs_crypto::ReadFile(key_file); bad_key[0] ^= 1;
        Text(key_file, std::string(bad_key.begin(), bad_key.end()));
        Require(!manual.ChangeModel(error) && revision() == '3', "Manual wrong key accepted");
        Model(assets / "example-only.key", key_file);
        Require(manual.ChangeModel(error) && revision() == '4', "Manual same-CSV retry failed");
        Text(csv, Rows());
        Require(manual.ChangeModel(error) && revision() == '4', "Manual downgrade accepted");
        Text(csv, Rows("19H") + "90,AUX,classify," + Side("15H") + "\n");
        Require(!manual.ChangeModel(error) && manual.thd_classifier.size() == 2,
                "Manual map structure changed while inference can read it");
        std::cout << "PASS: filename date validation, four-column CSV, multiple cameras, concurrent reload, "
                     "no unchanged reload, independent updates, retirement, missing-file retry, bad key/tamper/shape, "
                     "downgrade/routing rejection, new index, initial failure and superseded CSV; "
                     "two-function manual reload, direct atomic reads, no auto watcher, retry and retirement\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
