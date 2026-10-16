#include "../SKIRT/core/GpuKernelCache.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace
{
    int failures = 0;

    void expect(bool condition, const char* description)
    {
        if (!condition)
        {
            std::cerr << "FAIL: " << description << '\n';
            ++failures;
        }
    }

    bool overwriteByte(const std::string& path, off_t offset)
    {
        int descriptor = ::open(path.c_str(), O_RDWR);
        if (descriptor < 0) return false;
        unsigned char byte = 0;
        bool ok = ::pread(descriptor, &byte, 1, offset) == 1;
        byte ^= 0x40U;
        ok = ok && ::pwrite(descriptor, &byte, 1, offset) == 1;
        ok = ::close(descriptor) == 0 && ok;
        return ok;
    }

    void removeTemporaryDirectory(const std::string& path)
    {
        DIR* directory = ::opendir(path.c_str());
        if (directory)
        {
            while (dirent* entry = ::readdir(directory))
            {
                if (std::strcmp(entry->d_name, ".") && std::strcmp(entry->d_name, ".."))
                    ::unlink((path + "/" + entry->d_name).c_str());
            }
            ::closedir(directory);
        }
        ::rmdir(path.c_str());
    }
}

int main()
{
    char directoryTemplate[] = "/tmp/skirt-gpu-kernel-cache-XXXXXX";
    char* temporaryDirectory = ::mkdtemp(directoryTemplate);
    if (!temporaryDirectory)
    {
        std::cerr << "mkdtemp failed: " << std::strerror(errno) << '\n';
        return 2;
    }
    std::string directory(temporaryDirectory);

    ::unsetenv("SKIRTGPU_KERNEL_CACHE_DIR");
    std::string xdgDirectory = directory + "/xdg";
    ::setenv("XDG_CACHE_HOME", xdgDirectory.c_str(), 1);
    expect(GpuKernelCache::detail::directory() == xdgDirectory + "/skirtgpu/nvrtc",
           "XDG cache directory is used by default");
    ::unsetenv("XDG_CACHE_HOME");
    const char* configuredHome = std::getenv("HOME");
    expect(configuredHome && *configuredHome
               && GpuKernelCache::detail::directory() == std::string(configuredHome) + "/.cache/skirtgpu/nvrtc",
           "existing HOME cache directory is used when XDG_CACHE_HOME is absent");

    ::setenv("SKIRTGPU_KERNEL_CACHE_DIR", directory.c_str(), 1);
    ::unsetenv("SKIRTGPU_KERNEL_CACHE");

    const char keyBytes[] = "source\0--gpu-architecture=compute_89\0nvrtc-12.8";
    const std::string key(keyBytes, sizeof(keyBytes) - 1);
    const char ptxBytes[] = {'.', 'v', 'e', 'r', 's', 'i', 'o', 'n', ' ', '8', '.', '7', '\n', '\0', 'x', '\0'};
    const std::string ptx(ptxBytes, sizeof(ptxBytes));
    std::string loaded = "must be cleared";

    expect(!GpuKernelCache::load(key, loaded) && loaded.empty(), "initial lookup misses and clears output");
    GpuKernelCache::store(key, ptx);
    expect(GpuKernelCache::load(key, loaded) && loaded == ptx, "round trip preserves every PTX byte including NULs");
    expect(!GpuKernelCache::load(key + "changed", loaded), "changed key misses");

    std::string path = GpuKernelCache::detail::filePath(key);
    expect(overwriteByte(path, 32), "stored key byte can be corrupted for test");
    expect(!GpuKernelCache::load(key, loaded) && loaded.empty(), "stored key mismatch is rejected");

    GpuKernelCache::store(key, ptx);
    expect(overwriteByte(path, 32 + static_cast<off_t>(key.size())), "PTX byte can be corrupted for test");
    expect(!GpuKernelCache::load(key, loaded) && loaded.empty(), "checksum rejects same-length PTX corruption");

    GpuKernelCache::store(key, ptx);
    expect(overwriteByte(path, 23), "stored PTX length can be corrupted for test");
    expect(!GpuKernelCache::load(key, loaded) && loaded.empty(), "oversized declared PTX is rejected before allocation");

    GpuKernelCache::store(key, ptx);
    struct stat status;
    expect(::stat(path.c_str(), &status) == 0 && status.st_size > 0, "stored record has a size");
    expect(::truncate(path.c_str(), status.st_size - 1) == 0, "record can be truncated for test");
    expect(!GpuKernelCache::load(key, loaded), "truncated record is rejected");

    GpuKernelCache::store(key, ptx);
    ::setenv("SKIRTGPU_KERNEL_CACHE", "0", 1);
    expect(!GpuKernelCache::load(key, loaded), "disabled cache does not read an existing entry");
    const std::string disabledKey = key + "disabled";
    GpuKernelCache::store(disabledKey, ptx);
    expect(::access(GpuKernelCache::detail::filePath(disabledKey).c_str(), F_OK) != 0,
           "disabled cache does not write entries");
    ::unsetenv("SKIRTGPU_KERNEL_CACHE");

    const std::string unusablePath = directory + "/ordinary-file";
    int unusableDescriptor = ::open(unusablePath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    expect(unusableDescriptor >= 0, "unusable cache path fixture is created");
    if (unusableDescriptor >= 0) ::close(unusableDescriptor);
    ::setenv("SKIRTGPU_KERNEL_CACHE_DIR", unusablePath.c_str(), 1);
    GpuKernelCache::store("unwritable", ptx);
    expect(!GpuKernelCache::load("unwritable", loaded), "non-directory cache path fails as a miss");
    ::setenv("SKIRTGPU_KERNEL_CACHE_DIR", directory.c_str(), 1);

    const std::string concurrentKey = "same-key-concurrent-writers";
    const std::string firstPayload(200000, 'A');
    const std::string secondPayload(200000, 'B');
    pid_t children[4];
    for (int index = 0; index != 4; ++index)
    {
        children[index] = ::fork();
        if (children[index] == 0)
        {
            const std::string& payload = index & 1 ? firstPayload : secondPayload;
            for (int repetition = 0; repetition != 8; ++repetition)
                GpuKernelCache::store(concurrentKey, payload);
            ::_exit(0);
        }
        expect(children[index] > 0, "concurrent writer process starts");
    }
    for (pid_t child : children)
    {
        if (child <= 0) continue;
        int childStatus = -1;
        expect(::waitpid(child, &childStatus, 0) == child && WIFEXITED(childStatus) && WEXITSTATUS(childStatus) == 0,
               "concurrent writer process succeeds");
    }
    expect(GpuKernelCache::load(concurrentKey, loaded) && (loaded == firstPayload || loaded == secondPayload),
           "concurrent stores leave one complete valid record");

    GpuKernelCache::remove(concurrentKey);
    expect(!GpuKernelCache::load(concurrentKey, loaded), "remove deletes only the selected entry");
    expect(GpuKernelCache::load(key, loaded) && loaded == ptx, "remove preserves another entry");

    ::unlink(unusablePath.c_str());
    removeTemporaryDirectory(directory);
    if (failures)
    {
        std::cerr << failures << " cache test(s) failed\n";
        return 1;
    }
    std::cout << "GpuKernelCache tests passed\n";
    return 0;
}
