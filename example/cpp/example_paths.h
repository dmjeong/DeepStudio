#pragma once
#include <filesystem>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace example {
// Relative sample paths always mean "beside the executable", even from an IDE
// or a different current working directory. No argv[0] is needed.
inline std::filesystem::path ExecutableDirectory() {
#ifdef _WIN32
    std::vector<wchar_t> buffer(512);
    for (;;) {
        const auto size = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (!size) throw std::runtime_error("Cannot locate executable");
        if (size < buffer.size()) return std::filesystem::path(buffer.data()).parent_path();
        buffer.resize(buffer.size() * 2);
    }
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size);
    if (_NSGetExecutablePath(buffer.data(), &size) != 0)
        throw std::runtime_error("Cannot locate executable");
    return std::filesystem::canonical(buffer.data()).parent_path();
#else
    return std::filesystem::read_symlink("/proc/self/exe").parent_path();
#endif
}
} // namespace example
