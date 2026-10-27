#pragma once
#ifndef DVS_MODEL_CRYPTO_H_INCLUDED
#define DVS_MODEL_CRYPTO_H_INCLUDED
// DVSENC01 memory-only loader. Requires C++17, ONNX Runtime and nlohmann/json.
// Windows: bcrypt.lib (OS component). Other platforms: OpenSSL libcrypto.
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#else
#include <openssl/evp.h>
#include <openssl/crypto.h>
#endif

namespace dvs_crypto {
using Key = std::array<uint8_t, 32>;
inline void Wipe(void* data, size_t size) noexcept {
#ifdef _WIN32
    if (size) SecureZeroMemory(data, size);
#else
    if (size) OPENSSL_cleanse(data, size);
#endif
}
struct Secret {
    std::vector<uint8_t> bytes;
    ~Secret() { Wipe(bytes.data(), bytes.size()); }
    Secret() = default;
    Secret(const Secret&) = delete;
    Secret& operator=(const Secret&) = delete;
};
inline std::vector<uint8_t> ReadFile(const std::filesystem::path& path, size_t limit = 2147483647) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open encrypted model/key file");
    const auto end = file.tellg();
    if (end < 0 || static_cast<uint64_t>(end) > limit)
        throw std::invalid_argument("Encrypted model/key exceeds size limit");
    std::vector<uint8_t> data(static_cast<size_t>(end));
    file.seekg(0);
    if (!data.empty() && !file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size())))
        throw std::runtime_error("Cannot read encrypted model/key file");
    return data;
}
inline Key ReadKey(const std::filesystem::path& path) {
    Secret buffer;
    buffer.bytes = ReadFile(path, 32);
    if (buffer.bytes.size() != 32) throw std::invalid_argument("Model key must contain exactly 32 bytes");
    Key key{};
    std::memcpy(key.data(), buffer.bytes.data(), key.size());
    return key; // Caller owns the key and should wipe it after initialization.
}
inline std::string SafeName(const std::string& name) {
    if (name.empty() || name.front() == '/' || name.find_first_of("\\:") != std::string::npos ||
        name.find('\0') != std::string::npos)
        throw std::invalid_argument("Invalid encrypted graph name");
    size_t start = 0;
    do {
        auto end = name.find('/', start);
        auto part = name.substr(start, end == std::string::npos ? end : end-start);
        if (part.empty() || part == "." || part == "..") throw std::invalid_argument("Invalid encrypted graph path");
        if (end == std::string::npos) break;
        start = end + 1;
    } while (true);
    return name;
}

inline void Decrypt(const std::vector<uint8_t>& file, const Key& key, Secret& plain) {
    static const uint8_t magic[] = {'D','V','S','E','N','C','0','1'};
    if (file.size() < 40 || file.size() > 2147483647 || std::memcmp(file.data(), magic, 8))
        throw std::invalid_argument("Unsupported or truncated encrypted model");
    const size_t size = file.size() - 36;
    plain.bytes.resize(size);
#ifdef _WIN32
    struct Algorithm { BCRYPT_ALG_HANDLE h = nullptr; ~Algorithm(){ if(h) BCryptCloseAlgorithmProvider(h, 0); } } algorithm;
    struct KeyHandle { BCRYPT_KEY_HANDLE h = nullptr; ~KeyHandle(){ if(h) BCryptDestroyKey(h); } } handle;
    auto check = [](NTSTATUS s) { if (s < 0) throw std::runtime_error("Model decryption/authentication failed"); };
    check(BCryptOpenAlgorithmProvider(&algorithm.h, BCRYPT_AES_ALGORITHM, nullptr, 0));
    check(BCryptSetProperty(algorithm.h, BCRYPT_CHAINING_MODE,
          reinterpret_cast<PUCHAR>(const_cast<wchar_t*>(BCRYPT_CHAIN_MODE_GCM)), sizeof(BCRYPT_CHAIN_MODE_GCM), 0));
    check(BCryptGenerateSymmetricKey(algorithm.h, &handle.h, nullptr, 0,
          const_cast<PUCHAR>(key.data()), static_cast<ULONG>(key.size()), 0));
    BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO auth;
    BCRYPT_INIT_AUTH_MODE_INFO(auth);
    auth.pbNonce = const_cast<PUCHAR>(file.data() + 8); auth.cbNonce = 12;
    auth.pbAuthData = const_cast<PUCHAR>(magic); auth.cbAuthData = 8;
    auth.pbTag = const_cast<PUCHAR>(file.data() + file.size() - 16); auth.cbTag = 16;
    ULONG written = 0;
    check(BCryptDecrypt(handle.h, const_cast<PUCHAR>(file.data() + 20), static_cast<ULONG>(size),
          &auth, nullptr, 0, plain.bytes.data(), static_cast<ULONG>(size), &written, 0));
    if (written != size) throw std::runtime_error("Invalid decrypted model size");
#else
    std::unique_ptr<EVP_CIPHER_CTX, decltype(&EVP_CIPHER_CTX_free)> ctx(EVP_CIPHER_CTX_new(), EVP_CIPHER_CTX_free);
    auto check = [](int status) { if (status != 1) throw std::runtime_error("Model decryption/authentication failed"); };
    if (!ctx) throw std::bad_alloc();
    check(EVP_DecryptInit_ex(ctx.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr));
    check(EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_IVLEN, 12, nullptr));
    check(EVP_DecryptInit_ex(ctx.get(), nullptr, nullptr, key.data(), file.data()+8));
    int written = 0, tail = 0;
    check(EVP_DecryptUpdate(ctx.get(), nullptr, &written, magic, 8));
    check(EVP_DecryptUpdate(ctx.get(), plain.bytes.data(), &written, file.data()+20, static_cast<int>(size)));
    check(EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_TAG, 16,
                            const_cast<uint8_t*>(file.data()+file.size()-16)));
    check(EVP_DecryptFinal_ex(ctx.get(), plain.bytes.data()+written, &tail));
    if (static_cast<size_t>(written + tail) != size) throw std::runtime_error("Invalid decrypted model size");
#endif
}

// Owns authenticated bytes until sessions have copied their ONNX data. No disk
// extraction; no use_ort_model_bytes_directly option. Never retain keys here.
class Package {
public:
    Package(const std::filesystem::path& path, const Key& key) : plain_(std::make_shared<Secret>()) {
        auto file = ReadFile(path);
        Decrypt(file, key, *plain_);
        const auto& b = plain_->bytes;
        if (b.size() < 4) throw std::invalid_argument("Truncated package payload");
        const uint32_t size = uint32_t(b[0]) | uint32_t(b[1])<<8 | uint32_t(b[2])<<16 | uint32_t(b[3])<<24;
        if (!size || size > 16*1024*1024 || size > b.size()-4) throw std::invalid_argument("Invalid package JSON size");
        // Reject ambiguous metadata just like the Python package decoder. A
        // normal JSON DOM silently replaces an earlier duplicate object key.
        std::vector<std::set<std::string>> object_keys;
        const auto unique_keys = [&object_keys](int, nlohmann::json::parse_event_t event, nlohmann::json& value) {
            using Event = nlohmann::json::parse_event_t;
            if (event == Event::object_start) object_keys.emplace_back();
            else if (event == Event::key && !object_keys.back().insert(value.get<std::string>()).second)
                throw std::invalid_argument("Duplicate encrypted package JSON field");
            else if (event == Event::object_end) object_keys.pop_back();
            return true;
        };
        const auto doc = nlohmann::json::parse(b.begin()+4, b.begin()+4+size, unique_keys);
        if (!doc.is_object() || doc.value("format", std::string()) != "dvs-model-v1" || !doc.at("config").is_object())
            throw std::invalid_argument("Invalid encrypted model contract");
        config_ = doc.at("config");
        const auto& models = doc.at("models");
        if (!models.is_array() || models.empty() || models.size() > 64) throw std::invalid_argument("Invalid graph count");
        size_t offset = 4 + size;
        for (const auto& row : models) {
            const auto name = SafeName(row.at("name").get<std::string>());
            const auto& length = row.at("size");
            if (!length.is_number_unsigned() || length.get<uint64_t>() == 0 || length.get<uint64_t>() > b.size()-offset)
                throw std::invalid_argument("Invalid graph size");
            const size_t n = length.get<size_t>();
            if (!models_.emplace(name, std::make_pair(offset, n)).second) throw std::invalid_argument("Duplicate graph name");
            offset += n;
        }
        if (offset != b.size()) throw std::invalid_argument("Trailing package data");
        std::map<std::string, bool> required;
        required[SafeName(config_.at("model_path").get<std::string>())] = true;
        if (config_.contains("contracts")) {
            const auto& contracts = config_.at("contracts");
            if (!contracts.is_object()) throw std::invalid_argument("Invalid encrypted graph contract");
            if (contracts.contains("graphs")) {
                const auto& graphs = contracts.at("graphs");
                if (!graphs.is_object()) throw std::invalid_argument("Invalid encrypted graph contract");
                for (const auto& graph : graphs) {
                    if (!graph.is_object()) throw std::invalid_argument("Invalid encrypted graph entry");
                    required[SafeName(graph.at("file").get<std::string>())] = true;
                }
            }
        }
        if (required.size() != models_.size()) throw std::invalid_argument("Graph/config mismatch");
        for (const auto& item : required) if (!models_.count(item.first)) throw std::invalid_argument("Missing encrypted graph");
    }
    const nlohmann::json& Config() const { return config_; }
    std::unique_ptr<Ort::Session> Session(const Ort::Env& env, const std::string& name,
                                         const Ort::SessionOptions& options) const {
        const auto span = models_.at(SafeName(name));
        return std::make_unique<Ort::Session>(env, plain_->bytes.data()+span.first, span.second, options);
    }
private:
    std::shared_ptr<Secret> plain_;
    nlohmann::json config_;
    std::map<std::string, std::pair<size_t, size_t>> models_;
};
} // namespace dvs_crypto
#endif // DVS_MODEL_CRYPTO_H_INCLUDED
