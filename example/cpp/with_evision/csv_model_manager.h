#pragma once
#include "csv_model_watcher.h" // Shared CSV parser; legacy watcher remains available.

namespace dvs_bw8 {
namespace filename_csv_detail {
struct Spec {
    int index = 0;
    std::string camtype, task, name, family, version;
    bool operator==(const Spec& other) const {
        return index == other.index && camtype == other.camtype && task == other.task && name == other.name;
    }
};
using Specs = std::map<int, Spec>;

inline bool Digits(const std::string& text) {
    return !text.empty() && text.find_first_not_of("0123456789") == std::string::npos;
}
inline bool ControlCharacter(const std::string& text) {
    for (unsigned char ch : text) if (ch < 32 || ch == 127) return true;
    return false;
}
inline void ParseFilename(Spec& spec) {
    // Filename only, not a path. Dates are 2000..2099 in one agreed timezone.
    const auto& name = spec.name;
    const std::string extension = ".dvsenc";
    if (name.size() <= extension.size() || name.find_first_of("/\\:*?\"<>|") != std::string::npos
            || ControlCharacter(name) || name.substr(name.size() - extension.size()) != extension)
        throw std::invalid_argument("name must be a .dvsenc filename, not a path");
    const auto stem = name.substr(0, name.size() - extension.size());
    const auto time_start = stem.rfind('_');
    const auto date_start = time_start == std::string::npos || time_start == 0
        ? std::string::npos : stem.rfind('_', time_start - 1);
    if (date_start == std::string::npos || date_start == 0)
        throw std::invalid_argument("Expected filename like sevision_ai_261126_16H.dvsenc or sevision_ai_261126_163025.dvsenc");
    const auto date = stem.substr(date_start + 1, time_start - date_start - 1);
    auto time = stem.substr(time_start + 1);
    if (time.size() == 3 && time.back() == 'H') time = time.substr(0, 2) + "0000";
    if (date.size() != 6 || time.size() != 6 || !Digits(date) || !Digits(time))
        throw std::invalid_argument("Invalid filename date/time: " + name);
    spec.version = "20" + date + "_" + time + "_000000";
    if (!csv_reload_detail::ValidVersion(spec.version))
        throw std::invalid_argument("Invalid filename calendar date/time: " + name);
    spec.family = stem.substr(0, date_start);
}

inline Specs Read(const std::filesystem::path& csv) {
    std::ifstream stream(csv, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Cannot open model CSV: " + csv.u8string());
    const auto length = stream.tellg();
    if (length < 0 || static_cast<uint64_t>(length) > 1024 * 1024)
        throw std::invalid_argument("CSV exceeds 1 MiB");
    std::string text(static_cast<size_t>(length), '\0');
    stream.seekg(0);
    if (!text.empty() && !stream.read(&text[0], static_cast<std::streamsize>(text.size())))
        throw std::runtime_error("Cannot read complete model CSV");
    const auto rows = csv_reload_detail::ParseCsv(text);
    if (rows.size() < 2) throw std::invalid_argument("Model CSV has no rows");
    std::map<std::string, size_t> columns;
    for (size_t i = 0; i < rows[0].size(); ++i)
        if (rows[0][i].empty() || !columns.emplace(rows[0][i], i).second)
            throw std::invalid_argument("Empty or duplicate CSV header");
    if (columns.size() != 4) throw std::invalid_argument("Expected CSV columns: index,camtype,task,name");
    for (const auto* name : {"index", "camtype", "task", "name"})
        if (!columns.count(name)) throw std::invalid_argument(std::string("Missing CSV column: ") + name);
    Specs result;
    for (size_t i = 1; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (row.size() != 4) throw std::invalid_argument("Wrong CSV column count");
        const auto& index = row[columns.at("index")];
        if (!Digits(index) || index.size() > 10 || std::stoull(index) > INT32_MAX)
            throw std::invalid_argument("index must be a nonnegative 32-bit integer");
        Spec spec;
        spec.index = std::stoi(index);
        spec.camtype = row[columns.at("camtype")];
        spec.task = row[columns.at("task")];
        spec.name = row[columns.at("name")];
        if (spec.camtype.empty() || ControlCharacter(spec.camtype))
            throw std::invalid_argument("Invalid camtype");
        if (spec.task != "classify")
            throw std::invalid_argument("This BW8 manager supports task=classify only");
        ParseFilename(spec);
        if (!result.emplace(spec.index, spec).second) throw std::invalid_argument("Duplicate CSV index");
    }
    return result;
}
} // namespace filename_csv_detail

// One background thread for ALL classification slots. Only this thread writes
// the immutable table; inference readers atomically take an owning model pointer.
// Start/Stop: owner thread only. Join inference workers before destruction.
class CsvModelManager {
public:
    using KeyLoader = std::function<dvs_crypto::Key(int index)>;
    CsvModelManager(const std::filesystem::path& csv, const std::filesystem::path& model_directory,
                    KeyLoader keys, std::chrono::milliseconds interval = std::chrono::seconds(1))
        : csv_(std::filesystem::absolute(csv).lexically_normal()),
          directory_(Resolve(csv_, model_directory)), keys_(std::move(keys)), interval_(interval) {
        if (!keys_ || interval_ < std::chrono::milliseconds(50))
            throw std::invalid_argument("Invalid model manager options");
    }
    CsvModelManager(const std::filesystem::path& csv, const std::filesystem::path& model_directory,
                    const std::filesystem::path& key_file,
                    std::chrono::milliseconds interval = std::chrono::seconds(1))
        : CsvModelManager(csv, model_directory,
            [file = Resolve(std::filesystem::absolute(csv), key_file)](int) { return dvs_crypto::ReadKey(file); }, interval) {}
    ~CsvModelManager() { Stop(); }
    CsvModelManager(const CsvModelManager&) = delete;
    CsvModelManager& operator=(const CsvModelManager&) = delete;

    // All initial models must be valid/warmed before any are published.
    void Start() {
        if (started_) throw std::logic_error("Model manager already started");
        const auto desired = filename_csv_detail::Read(csv_);
        auto table = std::make_shared<Table>();
        for (const auto& row : desired) table->emplace(row.first, Entry{row.second, Load(row.second)});
        if (filename_csv_detail::Read(csv_) != desired)
            throw std::runtime_error("CSV changed during initial load; retry Start");
        stopping_.store(false);
        std::shared_ptr<const Table> ready = table;
        std::atomic_store(&current_, ready);
        try { worker_ = std::thread([this] { Watch(); }); }
        catch (...) { std::atomic_store(&current_, std::shared_ptr<const Table>{}); throw; }
        started_ = true;
        Status("Ready: " + std::to_string(table->size()) + " models");
    }
    void Stop() noexcept {
        { const std::lock_guard<std::mutex> lock(wait_mutex_); stopping_.store(true); }
        wake_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
    // No file access, key access, model initialization or explicit Context here.
    // A wrong/missing route returns null rather than inspecting with another camera's model.
    std::shared_ptr<Classifier> Snapshot(int index, const std::string& camtype,
                                         const std::string& task = "classify") const {
        const auto table = std::atomic_load(&current_);
        if (!table) return {};
        const auto found = table->find(index);
        if (found == table->end() || found->second.spec.camtype != camtype || found->second.spec.task != task) return {};
        return found->second.model;
    }
    std::string LastStatus() const {
        const std::lock_guard<std::mutex> lock(status_mutex_); return status_;
    }

private:
    using Spec = filename_csv_detail::Spec;
    struct Entry { Spec spec; std::shared_ptr<Classifier> model; };
    using Table = std::map<int, Entry>;
    static std::filesystem::path Resolve(const std::filesystem::path& csv, const std::filesystem::path& file) {
        return std::filesystem::absolute(file.is_absolute() ? file : csv.parent_path() / file).lexically_normal();
    }
    std::shared_ptr<Classifier> Load(const Spec& spec) {
        auto key = keys_(spec.index);
        struct EraseKey {
            dvs_crypto::Key& key;
            ~EraseKey() { dvs_crypto::Wipe(key.data(), key.size()); }
        } erase{key};
        dvs_crypto::Package package(directory_ / std::filesystem::u8path(spec.name), key);
        dvs_crypto::Wipe(key.data(), key.size());
        // Filename is the deployment version. The authenticated package's export
        // timestamp is independent; renaming never changes its weights/metadata.
        return std::make_shared<Classifier>(package); // One decryption; includes warmup.
    }
    void Status(std::string text) {
        const std::lock_guard<std::mutex> lock(status_mutex_); status_ = std::move(text);
    }
    void Poll() {
        for (auto it = retired_.begin(); it != retired_.end();) {
            if (it->second.use_count() == 1) it = retired_.erase(it);
            else ++it;
        }
        const auto desired = filename_csv_detail::Read(csv_);
        const auto before = std::atomic_load(&current_);
        // Prevent a partial CSV or accidental routing edit from removing an
        // active model. Changes to existing routes/families require a restart.
        for (const auto& row : *before) {
            const auto found = desired.find(row.first);
            if (found == desired.end()) throw std::runtime_error("Missing active index: " + std::to_string(row.first));
            const auto& old = row.second.spec;
            const auto& next = found->second;
            if (old.camtype != next.camtype || old.task != next.task || old.family != next.family)
                throw std::runtime_error("Changed camtype/task/model family; restart required for index " + std::to_string(row.first));
        }
        std::string messages;
        for (const auto& row : desired) {
            if (stopping_.load()) return;
            const auto& spec = row.second;
            try {
                const auto active = std::atomic_load(&current_);
                const auto found = active->find(spec.index);
                if (found != active->end()) {
                    const auto& old = found->second.spec;
                    if (spec == old) continue;
                    if (spec.version <= old.version) {
                        messages += "index " + std::to_string(spec.index) + ": Ignored same/older filename version; ";
                        continue;
                    }
                }
                if (retired_.count(spec.index)) {
                    messages += "index " + std::to_string(spec.index) + ": Waiting for previous model users; ";
                    continue;
                }
                Status("Loading index " + std::to_string(spec.index) + ": " + spec.name);
                auto shadow = Load(spec);
                // Revalidate the complete CSV; never publish a superseded request.
                if (filename_csv_detail::Read(csv_) != desired) {
                    Status("CSV changed during load; keeping current models"); return;
                }
                if (stopping_.load()) return;
                auto table = std::make_shared<Table>(*active);
                (*table)[spec.index] = Entry{spec, std::move(shadow)};
                if (found != active->end()) retired_.emplace(spec.index, found->second.model);
                std::shared_ptr<const Table> ready = table;
                std::atomic_store(&current_, std::move(ready));
            } catch (const std::exception& error) {
                messages += "index " + std::to_string(spec.index) + ": Keeping current model: " + error.what() + "; ";
            }
        }
        Status(messages.empty() ? "Ready: " + std::to_string(std::atomic_load(&current_)->size()) + " models" : messages);
    }
    void Watch() noexcept {
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(wait_mutex_);
                if (wake_.wait_for(lock, interval_, [this] { return stopping_.load(); })) return;
            }
            try { Poll(); }
            catch (const std::exception& error) {
                try { Status(std::string("Keeping current models: ") + error.what()); } catch (...) {}
            }
            catch (...) { try { Status("Keeping current models: unknown error"); } catch (...) {} }
        }
    }
    const std::filesystem::path csv_, directory_;
    const KeyLoader keys_;
    const std::chrono::milliseconds interval_;
    std::shared_ptr<const Table> current_;
    std::map<int, std::shared_ptr<Classifier>> retired_;
    bool started_ = false;
    std::atomic<bool> stopping_{false};
    std::thread worker_;
    std::mutex wait_mutex_;
    std::condition_variable wake_;
    mutable std::mutex status_mutex_;
    std::string status_;
};
} // namespace dvs_bw8
