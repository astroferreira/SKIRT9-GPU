/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       Copyright 2026 Astronomical Observatory, Ghent University ////
///////////////////////////////////////////////////////////////// */

#ifndef GPUKERNELCACHE_HPP
#define GPUKERNELCACHE_HPP

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

////////////////////////////////////////////////////////////////////

/** Provides a small persistent cache for NVRTC output. Cache failures are deliberately reported as misses so they
    can never affect simulation results. The caller must include every input that can affect generated PTX in the
    key, including the complete CUDA source, the NVRTC version, and the exact compiler options. */
namespace GpuKernelCache
{
    namespace detail
    {
        constexpr std::uint64_t maxKeySize = 64ULL * 1024ULL * 1024ULL;
        constexpr std::uint64_t maxPtxSize = 256ULL * 1024ULL * 1024ULL;
        constexpr std::uint64_t headerSize = 32ULL;
        constexpr std::uint64_t fnvOffset = 14695981039346656037ULL;
        constexpr std::uint64_t fnvPrime = 1099511628211ULL;

        inline std::uint64_t fnv1a(const void* bytes, size_t count, std::uint64_t hash = fnvOffset)
        {
            const unsigned char* current = static_cast<const unsigned char*>(bytes);
            for (size_t index = 0; index != count; ++index)
            {
                hash ^= current[index];
                hash *= fnvPrime;
            }
            return hash;
        }

        inline bool enabled()
        {
            const char* value = std::getenv("SKIRTGPU_KERNEL_CACHE");
            return !value || std::strcmp(value, "0") != 0;
        }

        inline std::string directory()
        {
            const char* overrideDirectory = std::getenv("SKIRTGPU_KERNEL_CACHE_DIR");
            if (overrideDirectory) return overrideDirectory;

            const char* xdgDirectory = std::getenv("XDG_CACHE_HOME");
            if (xdgDirectory && *xdgDirectory) return std::string(xdgDirectory) + "/skirtgpu/nvrtc";

            const char* homeDirectory = std::getenv("HOME");
            if (homeDirectory && *homeDirectory) return std::string(homeDirectory) + "/.cache/skirtgpu/nvrtc";
            return std::string();
        }

        inline std::string hashName(const std::string& key)
        {
            std::uint64_t hash = fnv1a(key.data(), key.size());
            static const char digits[] = "0123456789abcdef";
            char name[16];
            for (int index = 15; index >= 0; --index)
            {
                name[index] = digits[hash & 0xfU];
                hash >>= 4;
            }
            return std::string(name, sizeof(name)) + ".cache";
        }

        inline std::string filePath(const std::string& key)
        {
            std::string cacheDirectory = directory();
            if (cacheDirectory.empty()) return std::string();
            if (cacheDirectory.back() != '/') cacheDirectory += '/';
            return cacheDirectory + hashName(key);
        }

        inline bool makeDirectory(const std::string& path)
        {
            if (path.empty()) return false;
            std::string current;
            size_t offset = 0;
            if (path.front() == '/')
            {
                current = "/";
                offset = 1;
            }

            while (offset <= path.size())
            {
                size_t separator = path.find('/', offset);
                size_t end = separator == std::string::npos ? path.size() : separator;
                if (end != offset)
                {
                    if (!current.empty() && current.back() != '/') current += '/';
                    current.append(path, offset, end - offset);
                    if (::mkdir(current.c_str(), 0700) != 0 && errno != EEXIST) return false;
                    struct stat status;
                    if (::stat(current.c_str(), &status) != 0 || !S_ISDIR(status.st_mode)) return false;
                }
                if (separator == std::string::npos) break;
                offset = separator + 1;
            }
            return !current.empty();
        }

        inline void encode64(unsigned char* destination, std::uint64_t value)
        {
            for (int index = 0; index != 8; ++index)
            {
                destination[index] = static_cast<unsigned char>(value & 0xffU);
                value >>= 8;
            }
        }

        inline std::uint64_t decode64(const unsigned char* source)
        {
            std::uint64_t value = 0;
            for (int index = 7; index >= 0; --index) value = (value << 8) | source[index];
            return value;
        }

        inline bool readAll(int descriptor, void* destination, size_t count)
        {
            unsigned char* current = static_cast<unsigned char*>(destination);
            while (count)
            {
                ssize_t amount = ::read(descriptor, current, count);
                if (amount < 0)
                {
                    if (errno == EINTR) continue;
                    return false;
                }
                if (!amount) return false;
                current += amount;
                count -= static_cast<size_t>(amount);
            }
            return true;
        }

        inline bool writeAll(int descriptor, const void* source, size_t count)
        {
            const unsigned char* current = static_cast<const unsigned char*>(source);
            while (count)
            {
                ssize_t amount = ::write(descriptor, current, count);
                if (amount < 0)
                {
                    if (errno == EINTR) continue;
                    return false;
                }
                if (!amount) return false;
                current += amount;
                count -= static_cast<size_t>(amount);
            }
            return true;
        }

        inline int openReadOnly(const std::string& path)
        {
            int flags = O_RDONLY;
#ifdef O_CLOEXEC
            flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
            flags |= O_NOFOLLOW;
#endif
            return ::open(path.c_str(), flags);
        }

        inline int openExclusive(const std::string& path)
        {
            int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_CLOEXEC
            flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
            flags |= O_NOFOLLOW;
#endif
            return ::open(path.c_str(), flags, 0600);
        }

        class FileDescriptor
        {
        public:
            explicit FileDescriptor(int descriptor = -1) : _descriptor(descriptor) {}
            FileDescriptor(FileDescriptor&& other) noexcept : _descriptor(other.release()) {}
            FileDescriptor& operator=(FileDescriptor&& other) noexcept
            {
                if (this != &other)
                {
                    if (_descriptor >= 0) ::close(_descriptor);
                    _descriptor = other.release();
                }
                return *this;
            }
            ~FileDescriptor()
            {
                if (_descriptor >= 0) ::close(_descriptor);
            }
            int get() const { return _descriptor; }
            int release()
            {
                int descriptor = _descriptor;
                _descriptor = -1;
                return descriptor;
            }

        private:
            FileDescriptor(const FileDescriptor&) = delete;
            FileDescriptor& operator=(const FileDescriptor&) = delete;
            int _descriptor;
        };
    }

    /** Loads the PTX for the specified key. Returns false, leaving ptx empty, for a miss or any cache error. */
    inline bool load(const std::string& key, std::string& ptx)
    {
        ptx.clear();
        try
        {
            if (!detail::enabled() || key.size() > detail::maxKeySize) return false;
            std::string path = detail::filePath(key);
            if (path.empty()) return false;
            detail::FileDescriptor file(detail::openReadOnly(path));
            if (file.get() < 0) return false;

            struct stat status;
            if (::fstat(file.get(), &status) != 0 || !S_ISREG(status.st_mode) || status.st_size < 0
                || static_cast<std::uint64_t>(status.st_size) < detail::headerSize
                || static_cast<std::uint64_t>(status.st_size)
                       > detail::headerSize + detail::maxKeySize + detail::maxPtxSize)
                return false;

            unsigned char header[detail::headerSize];
            if (!detail::readAll(file.get(), header, sizeof(header))
                || std::memcmp(header, "SKPTXC01", 8) != 0)
                return false;
            std::uint64_t keySize = detail::decode64(header + 8);
            std::uint64_t ptxSize = detail::decode64(header + 16);
            std::uint64_t expectedChecksum = detail::decode64(header + 24);
            if (keySize != key.size() || keySize > detail::maxKeySize || ptxSize > detail::maxPtxSize
                || static_cast<std::uint64_t>(status.st_size) != detail::headerSize + keySize + ptxSize)
                return false;

            std::uint64_t checksum = detail::fnvOffset;
            unsigned char buffer[16384];
            size_t compared = 0;
            while (compared != key.size())
            {
                size_t amount = key.size() - compared;
                if (amount > sizeof(buffer)) amount = sizeof(buffer);
                if (!detail::readAll(file.get(), buffer, amount)
                    || std::memcmp(buffer, key.data() + compared, amount) != 0)
                    return false;
                checksum = detail::fnv1a(buffer, amount, checksum);
                compared += amount;
            }

            std::string candidate;
            candidate.resize(static_cast<size_t>(ptxSize));
            if (ptxSize && !detail::readAll(file.get(), &candidate[0], static_cast<size_t>(ptxSize))) return false;
            checksum = detail::fnv1a(candidate.data(), candidate.size(), checksum);
            if (checksum != expectedChecksum) return false;
            ptx.swap(candidate);
            return true;
        }
        catch (...)
        {
            ptx.clear();
            return false;
        }
    }

    /** Atomically stores PTX for the specified key. Any cache error is ignored. */
    inline void store(const std::string& key, const std::string& ptx)
    {
        std::string temporaryPath;
        bool temporaryCreated = false;
        try
        {
            if (!detail::enabled() || key.size() > detail::maxKeySize || ptx.size() > detail::maxPtxSize) return;
            std::string cacheDirectory = detail::directory();
            if (!detail::makeDirectory(cacheDirectory)) return;
            std::string finalPath = detail::filePath(key);
            if (finalPath.empty()) return;

            static std::atomic<unsigned long long> sequence{0};
            detail::FileDescriptor file;
            for (int attempt = 0; attempt != 64; ++attempt)
            {
                unsigned long long suffix = sequence.fetch_add(1, std::memory_order_relaxed);
                temporaryPath = finalPath + ".tmp." + std::to_string(static_cast<unsigned long long>(::getpid()))
                                + "." + std::to_string(suffix);
                int descriptor = detail::openExclusive(temporaryPath);
                if (descriptor >= 0)
                {
                    file = detail::FileDescriptor(descriptor);
                    temporaryCreated = true;
                    break;
                }
                if (errno != EEXIST) return;
            }
            if (file.get() < 0) return;

            unsigned char header[detail::headerSize];
            std::memcpy(header, "SKPTXC01", 8);
            detail::encode64(header + 8, key.size());
            detail::encode64(header + 16, ptx.size());
            std::uint64_t checksum = detail::fnv1a(key.data(), key.size());
            checksum = detail::fnv1a(ptx.data(), ptx.size(), checksum);
            detail::encode64(header + 24, checksum);

            bool written = detail::writeAll(file.get(), header, sizeof(header))
                           && detail::writeAll(file.get(), key.data(), key.size())
                           && detail::writeAll(file.get(), ptx.data(), ptx.size()) && ::fsync(file.get()) == 0;
            int descriptor = file.release();
            if (::close(descriptor) != 0) written = false;
            if (!written || ::rename(temporaryPath.c_str(), finalPath.c_str()) != 0)
            {
                ::unlink(temporaryPath.c_str());
                temporaryCreated = false;
                return;
            }
            temporaryCreated = false;
        }
        catch (...)
        {
            if (temporaryCreated) ::unlink(temporaryPath.c_str());
        }
    }

    /** Removes the entry for the specified key, ignoring missing files and other cache errors. */
    inline void remove(const std::string& key)
    {
        try
        {
            if (!detail::enabled() || key.size() > detail::maxKeySize) return;
            std::string path = detail::filePath(key);
            if (!path.empty()) ::unlink(path.c_str());
        }
        catch (...)
        {
        }
    }
}

////////////////////////////////////////////////////////////////////

#endif
