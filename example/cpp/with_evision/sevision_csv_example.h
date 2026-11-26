#pragma once
#include "csv_model_manager.h" // Reuses the CSV/date parser only; creates NO manager/thread.

// In your existing CSeVision, copy these members and the TWO functions.
// Initialize before starting inspection. Call ChangeModel from your own thread.
// Join update/inspection threads before destroying CSeVision.
class CSeVision {
public:
    // Fixed index entries after initialization. Callers must only atomic_load
    // their shared_ptr values; never erase/insert/assign entries during inspection.
    std::map<int, std::shared_ptr<dvs_bw8::Classifier>> thd_classifier;

    // 1. Call once at startup. Relative paths are resolved against the CSV directory.
    bool InitializeModel(const std::filesystem::path& csv,
                         const std::filesystem::path& modelDirectory,
                         const std::filesystem::path& keyFile,
                         std::string& error) {
        const std::lock_guard<std::mutex> lock(change_mutex_);
        error.clear();
        try {
            if (initialized_) throw std::logic_error("Models already initialized");
            const auto csv_path = std::filesystem::absolute(csv).lexically_normal();
            const auto directory = std::filesystem::absolute(modelDirectory.is_absolute()
                ? modelDirectory : csv_path.parent_path() / modelDirectory).lexically_normal();
            const auto key_path = std::filesystem::absolute(keyFile.is_absolute()
                ? keyFile : csv_path.parent_path() / keyFile).lexically_normal();
            auto desired = dvs_bw8::filename_csv_detail::Read(csv_path);
            std::map<int, std::shared_ptr<dvs_bw8::Classifier>> ready;
            for (const auto& row : desired) {
                auto key = dvs_crypto::ReadKey(key_path);
                EraseKey erase{key};
                dvs_crypto::Package package(directory / std::filesystem::u8path(row.second.name), key);
                dvs_crypto::Wipe(key.data(), key.size());
                ready.emplace(row.first, std::make_shared<dvs_bw8::Classifier>(package)); // Includes warmup.
            }
            if (dvs_bw8::filename_csv_detail::Read(csv_path) != desired)
                throw std::runtime_error("CSV changed during initial load; retry initialization");
            csv_ = csv_path; directory_ = directory; key_ = key_path;
            applied_.swap(desired);
            thd_classifier.swap(ready); // Before any inspection thread starts.
            initialized_ = true;
            return true;
        } catch (const std::exception& ex) { error = ex.what(); return false; }
    }

    // 2. Call periodically from YOUR thread (e.g. once per second).
    // One CSV check, then return. No thread, sleep, loop or model reload if unchanged.
    // True = checked successfully (also true if unchanged/waiting for old users).
    // False = error contains the failure; already active models remain usable.
    bool ChangeModel(std::string& error) {
        const std::lock_guard<std::mutex> lock(change_mutex_);
        error.clear();
        try {
            if (!initialized_) throw std::logic_error("InitializeModel must succeed first");
            for (auto it = retired_.begin(); it != retired_.end();) {
                if (it->second.use_count() == 1) it = retired_.erase(it);
                else ++it;
            }
            const auto desired = dvs_bw8::filename_csv_detail::Read(csv_);
            if (desired.size() != applied_.size())
                throw std::runtime_error("CSV index set changed; restart to add/remove models");
            for (const auto& row : applied_) {
                const auto found = desired.find(row.first);
                if (found == desired.end() || found->second.camtype != row.second.camtype
                        || found->second.task != row.second.task || found->second.family != row.second.family)
                    throw std::runtime_error("CSV index/camtype/task/model family changed; restart required");
            }
            for (const auto& row : desired) {
                const int index = row.first;
                const auto& next = row.second;
                const auto& previous = applied_.at(index);
                if (next.name == previous.name || next.version <= previous.version || retired_.count(index)) continue;
                try {
                    auto key = dvs_crypto::ReadKey(key_);
                    EraseKey erase{key};
                    dvs_crypto::Package package(directory_ / std::filesystem::u8path(next.name), key);
                    dvs_crypto::Wipe(key.data(), key.size());
                    auto shadow = std::make_shared<dvs_bw8::Classifier>(package); // One decryption + warmup.
                    if (dvs_bw8::filename_csv_detail::Read(csv_) != desired)
                        throw std::runtime_error("CSV changed during load; retry next call");
                    // Finish allocations before publishing either state or model.
                    auto applied = applied_;
                    applied.at(index) = next;
                    auto& slot = thd_classifier.at(index);
                    retired_.emplace(index, std::atomic_load(&slot));
                    applied_.swap(applied);
                    std::atomic_store(&slot, std::move(shadow));
                } catch (const std::exception& ex) {
                    error += "index " + std::to_string(index) + ": " + ex.what() + "\n";
                }
            }
            return error.empty();
        } catch (const std::exception& ex) { error = ex.what(); return false; }
    }

private:
    struct EraseKey {
        dvs_crypto::Key& key;
        ~EraseKey() { dvs_crypto::Wipe(key.data(), key.size()); }
    };
    bool initialized_ = false;
    std::filesystem::path csv_, directory_, key_;
    dvs_bw8::filename_csv_detail::Specs applied_;
    std::map<int, std::shared_ptr<dvs_bw8::Classifier>> retired_;
    std::mutex change_mutex_; // Update calls only; inference never takes this lock.
};
