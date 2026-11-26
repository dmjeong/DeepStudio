#include "csv_model_watcher.h"
#include "../example_paths.h"
#include <iostream>

namespace {
using namespace std::chrono_literals;
const std::string v1 = "20261126_010000_000001", v2 = "20261126_010000_000002";
const std::string v3 = "20261126_010000_000003", v4 = "20261126_010000_000004";
void Require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class Predicate> void Wait(Predicate check) {
    const auto end = std::chrono::steady_clock::now() + 8s;
    while (!check()) {
        if (std::chrono::steady_clock::now() > end) throw std::runtime_error("Watcher test timed out");
        std::this_thread::sleep_for(10ms);
    }
}
struct Temp {
    std::filesystem::path path;
    Temp() {
        for (int i = 0; i < 100; ++i) {
            auto candidate = std::filesystem::temp_directory_path() /
                ("dvs-csv-reload-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
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
            throw std::runtime_error("Cannot atomically publish test file");
        std::this_thread::sleep_for(10ms);
    }
#else
    std::filesystem::rename(from, to);
#endif
}
void PublishText(const std::filesystem::path& path, const std::string& text) {
    auto temporary = path; temporary += ".pending";
    { std::ofstream stream(temporary, std::ios::binary);
      stream << text; if (!stream) throw std::runtime_error("Cannot write test CSV"); }
    Replace(temporary, path);
}
void PublishModel(const std::filesystem::path& source, const std::filesystem::path& path) {
    auto temporary = path; temporary += ".pending";
    std::filesystem::copy_file(source, temporary, std::filesystem::copy_options::overwrite_existing);
    Replace(temporary, path);
}
std::string Row(const std::string& version) {
    // BOM, quoted comma, UTF-8 path, CRLF; path resolves against CSV, not CWD.
    return "\xef\xbb\xbfslot,task,model_file,model_version\r\nchip,classify,\""
        + std::string(u8"검사, model.dvsenc") + "\"," + version + "\r\n";
}
struct Readers {
    std::atomic<bool> running{true};
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    explicit Readers(dvs_bw8::CsvModelWatcher& watcher) {
        for (int i = 0; i < 4; ++i) threads.emplace_back([this, &watcher] {
            const std::vector<uint8_t> frame(224 * 224, 255);
            while (running.load()) {
                try {
                    auto model = watcher.Snapshot();
                    const auto version = model->ModelVersion();
                    const auto result = model->InferBW8(frame.data(), 224, 224, 224);
                    const std::string expected = "white_v" + version.substr(21, 1);
                    if (result.class_id != 0 || result.class_name != expected) ++errors;
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
        using dvs_bw8::csv_reload_detail::ValidVersion;
        Require(ValidVersion(v1), "Valid version rejected");
        Require(!ValidVersion("20260230_010000_000001"), "Invalid calendar date accepted");
        Require(!ValidVersion("20261126_240000_000001"), "Invalid hour accepted");
        const auto quoted = dvs_bw8::csv_reload_detail::ParseCsv("a,b\n\"x\"\"y\",\"p,q\"\n");
        Require(quoted[1][0] == "x\"y" && quoted[1][1] == "p,q", "CSV quoting failed");
        const auto assets = example::ExecutableDirectory() / "assets";
        Temp folder;
        const auto csv = folder.path / "models.csv";
        const auto live = folder.path / std::filesystem::u8path(u8"검사, model.dvsenc");
        PublishModel(assets / "reload/model.dvsenc", live);
        PublishText(csv, Row(v1));
        std::atomic<int> reads{0};
        dvs_bw8::CsvModelWatcher watcher(csv, "chip", [&] {
            ++reads; return dvs_crypto::ReadKey(assets / "example-only.key");
        }, 50ms);
        watcher.Start();
        Require(watcher.Snapshot()->ModelVersion() == v1, "Initial load failed");
        std::this_thread::sleep_for(160ms);
        Require(reads.load() == 1, "Unchanged model reread/decrypted");
        Readers readers(watcher);
        auto old = watcher.Snapshot();
        std::weak_ptr<dvs_bw8::Classifier> released_old = old;
        const auto original_mtime = std::filesystem::last_write_time(csv);
        PublishModel(assets / "reload/v2.dvsenc", live); // Fixed filename.
        PublishText(csv, Row(v2)); // Publish CSV last.
        std::filesystem::last_write_time(csv, original_mtime);
        Wait([&] { return watcher.Snapshot()->ModelVersion() == v2; });
        Require(old->ModelVersion() == v1, "In-flight model changed");

        PublishModel(assets / "reload/v3.dvsenc", live);
        PublishText(csv, Row(v3));
        Wait([&] { return watcher.LastStatus().find("Waiting for previous") != std::string::npos; });
        Require(watcher.Snapshot()->ModelVersion() == v2 && reads.load() == 2,
                "Unbounded live sessions while old model is held");
        old.reset();
        Wait([&] { return watcher.Snapshot()->ModelVersion() == v3; });
        Require(released_old.expired(), "Retired model was not reclaimed");

        // GCM rejection, then shape rejection, then successful retry with the
        // SAME CSV. A failure must never cache the desired version as applied.
        auto corrupt = dvs_crypto::ReadFile(assets / "reload/v4.dvsenc");
        corrupt.back() ^= 1;
        PublishText(live, std::string(corrupt.begin(), corrupt.end()));
        PublishText(csv, Row(v4));
        Wait([&] { return watcher.LastStatus().find("Keeping current model:") == 0; });
        Require(watcher.Snapshot()->ModelVersion() == v3, "Tampered model published");
        PublishModel(assets / "reload/bad-shape-v4.dvsenc", live);
        Wait([&] { return watcher.LastStatus().find("Tensor shape/type") != std::string::npos; });
        Require(watcher.Snapshot()->ModelVersion() == v3, "Bad I/O model published");
        PublishModel(assets / "reload/v4.dvsenc", live);
        Wait([&] { return watcher.Snapshot()->ModelVersion() == v4; });

        const int after_update = reads.load();
        PublishText(csv, Row(v1));
        Wait([&] { return watcher.LastStatus().find("Ignored older") == 0; });
        Require(watcher.Snapshot()->ModelVersion() == v4 && reads.load() == after_update, "Downgrade occurred");
        PublishText(csv, Row(v4) + "chip,classify,same.dvsenc," + v4 + "\n");
        Wait([&] { return watcher.LastStatus().find("duplicate CSV slot") != std::string::npos; });
        Require(watcher.Snapshot()->ModelVersion() == v4, "Malformed CSV replaced model");
        PublishText(csv, Row("20261126_010000_000005"));
        Wait([&] { return watcher.LastStatus().find("CSV/model version mismatch") != std::string::npos; });
        Require(watcher.Snapshot()->ModelVersion() == v4, "CSV/package version mismatch published");
        readers.Stop();
        Require(readers.errors.load() == 0, "Parallel inference corrupted during reload");
        watcher.Stop();

        // A CSV changed during construction cannot publish the obsolete row.
        PublishModel(assets / "reload/model.dvsenc", live);
        PublishText(csv, Row(v1));
        dvs_bw8::CsvModelWatcher superseded(csv, "chip", [&] {
            PublishText(csv, Row(v2));
            return dvs_crypto::ReadKey(assets / "example-only.key");
        }, 50ms);
        bool rejected = false;
        try { superseded.Start(); } catch (const std::exception&) { rejected = true; }
        Require(rejected && !superseded.Snapshot(), "Superseded CSV published during load");
        std::cout << "PASS: CSV parsing, fixed-path swaps, concurrent inference, bounded retirement, "
                     "retry, tamper/shape/version rejection, unchanged timestamp, superseded CSV\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
