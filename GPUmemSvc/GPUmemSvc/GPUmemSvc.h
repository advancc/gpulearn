#ifndef GPUMEMSVC_H
#define GPUMEMSVC_H

#include "SniperKernel/SvcBase.h"
#include <cuda_runtime_api.h>
#include <string>
#include <stdexcept>

class GPUmemSvc : public SvcBase
{
  public:
    GPUmemSvc(const std::string &name);

    bool initialize();
    bool finalize();
    void managed_allocate(size_t num_bytes, initial_visibility_t initial_visibility);
    void free(void *ptr);
    void prefetch(void *managed_ptr,size_t num_bytes,int dstDevice,cudaStream_t stream_id)
    constexpr inline bool is_success(status_t status) { return status == (status_t)status::success; }
    constexpr inline bool is_failure(status_t status) { return status != (status_t)status::success; }

  private:
    void *m_handle;
    int m_ivar;
    void *m_ptr;

    enum class initial_visibility_t
    {
        to_all_devices,
        to_supporters_of_concurrent_managed_access,
    };
};

#endif
