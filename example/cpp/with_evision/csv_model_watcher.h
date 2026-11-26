#pragma once
#include "classifier.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <thread>

namespace dvs_bw8 {
namespace csv_reload_detail {
inline std::string Trim(std::string value) {
    const auto first = value.find_first_not_of(" \t");
    if (first == std::string::npos) return {};
    return value.substr(first, value.find_last_not_of(" \t") - first + 1);
}

// UTF-8 CSV, optional BOM, CRLF, quoted commas and doubled quotes.
inline std::vector<std::vector<std::string>> ParseCsv(std::string text) {
    if (text.compare(0, 3, "\xef\xbb\xbf") == 0) text.erase(0, 3);
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool quoted = false, closed_quote = false;
    const auto end_field = [&] {
        row.push_back(Trim(field)); field.clear(); closed_quote = false;
    };
    const auto end_row = [&] {
        end_field();
        if (row.size() != 1 || !row.front().empty()) rows.push_back(std::move(row));
        row.clear();
    };
    for (size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (ch == '\0') throw std::invalid_argument("NUL in CSV");
        if (quoted) {
            if (ch == '"') {
                if (i + 1 < text.size() && text[i + 1] == '"') { field += '"'; ++i; }
                else { quoted = false; closed_quote = true; }
            } else field += ch;
        } else if (ch == ',') end_field();
        else if (ch == '\r' || ch == '\n') {
            end_row();
            if (ch == '\r' && i + 1 < text.size() && text[i + 1] == '\n') ++i;
        } else if (closed_quote) {
            if (ch != ' ' && ch != '\t') throw std::invalid_argument("Unexpected text after CSV quote");
        } else if (ch == '"') {
            if (!field.empty()) throw std::invalid_argument("Unexpected CSV quote");
            quoted = true;
        } else field += ch;
    }
    if (quoted) throw std::invalid_argument("Unclosed CSV quote");
    if (!row.empty() || !field.empty() || closed_quote) end_row();
    return rows;
}

inline bool ValidVersion(const std::string& version) {
    if (version.size() != 22 || version[8] != '_' || version[15] != '_') return false;
    for (size_t i = 0; i < version.size(); ++i)
        if (i != 8 && i != 15 && (version[i] < '0' || version[i] > '9')) return false;
    const int year = std::stoi(version.substr(0, 4)), month = std::stoi(version.substr(4, 2));
    const int day = std::stoi(version.substr(6, 2));
    if (year < 1 || month < 1 || month > 12 || day < 1) return false;
    const int days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    const bool leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    return day <= days[month - 1] + (month == 2 && leap ? 1 : 0)
        && std::stoi(version.substr(9, 2)) < 24
        && std::stoi(version.substr(11, 2)) < 60
        && std::stoi(version.substr(13, 2)) < 60;
}

struct Spec {
    std::filesystem::path model;
    std::string version;
    bool operator==(const Spec& other) const { return model == other.model && version == other.version; }
};

inline Spec ReadSpec(const std::filesystem::path& csv, const std::string& slot) {
    std::ifstream stream(csv, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Cannot open model CSV: " + csv.u8string());
    const auto length = stream.tellg();
    if (length < 0 || static_cast<uint64_t>(length) > 1024 * 1024)
        throw std::invalid_argument("CSV exceeds 1 MiB");
    std::string text(static_cast<size_t>(length), '\0');
    stream.seekg(0);
    if (!text.empty() && !stream.read(&text[0], static_cast<std::streamsize>(text.size())))
        throw std::runtime_error("Cannot read complete model CSV");
    const auto rows = ParseCsv(text);
    if (rows.empty()) throw std::invalid_argument("Empty model CSV");
    std::map<std::string, size_t> columns;
    for (size_t i = 0; i < rows[0].size(); ++i)
        if (rows[0][i].empty() || !columns.emplace(rows[0][i], i).second)
            throw std::invalid_argument("Empty or duplicate CSV header");
    for (const auto* name : {"slot", "task", "model_file", "model_version"})
        if (!columns.count(name)) throw std::invalid_argument(std::string("Missing CSV column: ") + name);
    std::set<std::string> slots;
    Spec result;
    bool found = false;
    for (size_t i = 1; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (row.size() != rows[0].size()) throw std::invalid_argument("Wrong CSV column count");
        const auto& id = row[columns.at("slot")];
        if (id.empty() || !slots.insert(id).second) throw std::invalid_argument("Empty or duplicate CSV slot");
        if (id != slot) continue;
        if (row[columns.at("task")] != "classify")
            throw std::invalid_argument("This classifier watcher supports task=classify only");
        result.version = row[columns.at("model_version")];
        if (!ValidVersion(result.version)) throw std::invalid_argument("Expected model_version YYYYMMDD_HHMMSS_microseconds");
        const auto& file = row[columns.at("model_file")];
        if (file.empty() || file.find_first_of("\r\n") != std::string::npos)
            throw std::invalid_argument("Invalid model_file");
        result.model = std::filesystem::u8path(file);
        if (result.model.is_relative()) result.model = csv.parent_path() / result.model;
        result.model = result.model.lexically_normal();
        found = true;
    }
    if (!found) throw std::invalid_argument("CSV slot not found: " + slot);
    return result;
}
} // namespace csv_reload_detail

// One watcher/writer per classification slot. Start/Stop belong to the owner
// thread; Snapshot/LastStatus may be called concurrently by inference workers.
class CsvModelWatcher {
public:
    using KeyLoader = std::function<dvs_crypto::Key()>;
    CsvModelWatcher(std::filesystem::path csv, std::string slot, KeyLoader keys,
                    std::chrono::milliseconds interval = std::chrono::seconds(1))
        : csv_(std::filesystem::absolute(csv).lexically_normal()), slot_(std::move(slot)),
          keys_(std::move(keys)), interval_(interval) {
        if (slot_.empty() || !keys_ || interval_ < std::chrono::milliseconds(50))
            throw std::invalid_argument("Invalid model watcher options");
    }
    // A relative key path, like model_file, is relative to the CSV directory.
    CsvModelWatcher(const std::filesystem::path& csv, std::string slot,
                    const std::filesystem::path& key_file,
                    std::chrono::milliseconds interval = std::chrono::seconds(1))
        : CsvModelWatcher(csv, std::move(slot),
            [file = std::filesystem::absolute(key_file.is_absolute()
                ? key_file : std::filesystem::absolute(csv).parent_path() / key_file)] {
                return dvs_crypto::ReadKey(file);
            }, interval) {}
    ~CsvModelWatcher() { Stop(); }
    CsvModelWatcher(const CsvModelWatcher&) = delete;
    CsvModelWatcher& operator=(const CsvModelWatcher&) = delete;

    // Initial load is synchronous: return only after a valid model is warmed.
    // An initial failure throws; background failures retain the current model.
    void Start() {
        if (worker_.joinable()) throw std::logic_error("Watcher already started");
        stopping_.store(false);
        Reload();
        if (!Snapshot()) throw std::runtime_error("Initial model not ready");
        worker_ = std::thread([this] { Watch(); });
    }
    void Stop() noexcept {
        {
            const std::lock_guard<std::mutex> lock(wait_mutex_);
            stopping_.store(true);
        }
        wake_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
    std::shared_ptr<Classifier> Snapshot() const { return std::atomic_load(&current_); }
    std::string LastStatus() const {
        const std::lock_guard<std::mutex> lock(status_mutex_);
        return status_;
    }

private:
    void Status(std::string text) {
        const std::lock_guard<std::mutex> lock(status_mutex_);
        status_ = std::move(text);
    }
    void Reload() {
        // Keep ownership on this thread until the last inference releases it.
        if (retired_ && retired_.use_count() == 1) retired_.reset();
        const auto desired = csv_reload_detail::ReadSpec(csv_, slot_);
        auto active = Snapshot();
        if (active && desired.version <= active->ModelVersion()) {
            Status(desired.version == active->ModelVersion() ? "Active: " + desired.version
                : "Ignored older version: " + desired.version);
            return;
        }
        // Bound session growth: wait before creating a third model. The newest
        // CSV is reread on the next poll, so intermediate updates can be skipped.
        if (retired_) { Status("Waiting for previous model users to finish"); return; }
        Status("Loading: " + desired.version);
        auto key = keys_();
        struct EraseKey {
            dvs_crypto::Key& value;
            ~EraseKey() { dvs_crypto::Wipe(value.data(), value.size()); }
        } erase{key};
        dvs_crypto::Package package(desired.model, key);
        dvs_crypto::Wipe(key.data(), key.size());
        if (package.ModelVersion() != desired.version)
            throw std::invalid_argument("CSV/model version mismatch");
        auto shadow = std::make_shared<Classifier>(package); // Includes warmup.
        // Do not publish a model requested by a superseded/partially written CSV.
        if (!(csv_reload_detail::ReadSpec(csv_, slot_) == desired)) {
            Status("CSV changed during load; retrying latest version"); return;
        }
        if (stopping_.load()) return;
        retired_ = std::atomic_exchange(&current_, std::move(shadow));
        Status("Active: " + desired.version);
    }
    void Watch() noexcept {
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(wait_mutex_);
                if (wake_.wait_for(lock, interval_, [this] { return stopping_.load(); })) return;
            }
            try { Reload(); }
            catch (const std::exception& error) {
                try { Status(std::string("Keeping current model: ") + error.what()); } catch (...) {}
            }
            catch (...) { try { Status("Keeping current model: unknown reload error"); } catch (...) {} }
            // Failed updates are retried even when the CSV timestamp/content
            // stays the same (e.g. the model file finishes arriving later).
        }
    }
    const std::filesystem::path csv_;
    const std::string slot_;
    const KeyLoader keys_;
    const std::chrono::milliseconds interval_;
    std::shared_ptr<Classifier> current_, retired_;
    std::atomic<bool> stopping_{false};
    std::thread worker_;
    std::mutex wait_mutex_;
    std::condition_variable wake_;
    mutable std::mutex status_mutex_;
    std::string status_;
};
} // namespace dvs_bw8
