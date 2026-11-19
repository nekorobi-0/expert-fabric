#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <string>
#include <linux/magic.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>
#include <hipfile.h>

namespace {
constexpr size_t block_size = 4096;

struct ProbeResources {
    int fd = -1;
    bool driver_open = false;
    hipFileHandle_t file = nullptr;
    void *gpu = nullptr;
    bool registered = false;

    ~ProbeResources() {
        if (registered) {
            static_cast<void>(hipFileBufDeregister(gpu));
        }
        if (gpu) {
            static_cast<void>(hipFree(gpu));
        }
        if (file) {
            hipFileHandleDeregister(file);
        }
        if (driver_open) {
            static_cast<void>(hipFileDriverClose());
        }
        if (fd >= 0) {
            close(fd);
        }
    }
};

bool is_rdma_mount(int fd) {
    std::ifstream fdinfo("/proc/self/fdinfo/" + std::to_string(fd));
    std::string mount_id;
    for (std::string line; std::getline(fdinfo, line);) {
        if (line.starts_with("mnt_id:")) {
            std::istringstream(line.substr(7)) >> mount_id;
            break;
        }
    }
    if (mount_id.empty()) {
        return false;
    }
    std::ifstream mounts("/proc/self/mountinfo");
    for (std::string line; std::getline(mounts, line);) {
        if (!line.starts_with(mount_id + " ")) {
            continue;
        }
        const size_t fields = line.find(" - ");
        if (fields == std::string::npos) {
            return false;
        }
        std::istringstream mount(line.substr(fields + 3));
        std::string filesystem, source, options;
        if (!(mount >> filesystem >> source >> options)) {
            return false;
        }
        return ("," + options + ",").find(",proto=rdma,") != std::string::npos ||
               ("," + options + ",").find(",proto=rdma6,") != std::string::npos;
    }
    return false;
}

bool check(hipFileError_t status, const char *operation) {
    if (status.err == hipFileSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s failed (hipFile=%d, HIP=%d)\n",
                 operation, static_cast<int>(status.err), static_cast<int>(status.hip_drv_err));
    return false;
}
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s FILE_ON_NFS_RDMA_MOUNT\n", argv[0]);
        return 2;
    }
    ProbeResources resources;
    resources.fd = open(argv[1], O_RDONLY | O_DIRECT | O_CLOEXEC | O_NONBLOCK);
    if (resources.fd < 0) {
        std::perror("open with O_DIRECT");
        return 1;
    }
    struct statfs filesystem {};
    if (fstatfs(resources.fd, &filesystem) < 0) {
        std::perror("fstatfs");
        return 1;
    }
    if (filesystem.f_type != NFS_SUPER_MAGIC || !is_rdma_mount(resources.fd)) {
        std::fprintf(stderr, "The input file must be on an NFS mount using RDMA.\n");
        return 1;
    }
    struct stat file_info {};
    if (fstat(resources.fd, &file_info) < 0) {
        std::perror("fstat");
        return 1;
    }
    if (!S_ISREG(file_info.st_mode) || file_info.st_size < static_cast<off_t>(block_size)) {
        std::fprintf(stderr, "Provide a regular file containing at least %zu bytes.\n", block_size);
        return 1;
    }
    // Required by hipFile for the Linux NFSoRDMA fast path.
    if (setenv("HIPFILE_UNSUPPORTED_FILE_SYSTEMS", "true", 1) < 0 ||
        setenv("HIPFILE_ALLOW_COMPAT_MODE", "false", 1) < 0) {
        std::perror("setenv for hipFile");
        return 1;
    }
    if (!check(hipFileDriverOpen(), "hipFileDriverOpen")) {
        return 1;
    }
    resources.driver_open = true;
    hipFileDescr_t descriptor {};
    descriptor.type = hipFileHandleTypeOpaqueFD;
    descriptor.handle.fd = resources.fd;
    if (!check(hipFileHandleRegister(&resources.file, &descriptor), "hipFileHandleRegister")) {
        return 1;
    }
    const hipError_t gpu_status = hipMalloc(&resources.gpu, block_size);
    if (gpu_status != hipSuccess) {
        std::fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(gpu_status));
        return 1;
    }
    if (!check(hipFileBufRegister(resources.gpu, block_size, 0), "hipFileBufRegister")) {
        return 1;
    }
    resources.registered = true;
    const ssize_t bytes_read = hipFileRead(resources.file, resources.gpu, block_size, 0, 0);
    if (bytes_read == -1) {
        std::perror("hipFileRead");
        return 1;
    }
    if (bytes_read != static_cast<ssize_t>(block_size)) {
        std::fprintf(stderr, "hipFileRead returned %zd, expected %zu bytes\n", bytes_read, block_size);
        return 1;
    }
    std::printf("Read %zd bytes into VRAM with hipFile compatibility mode disabled.\n", bytes_read);
}
