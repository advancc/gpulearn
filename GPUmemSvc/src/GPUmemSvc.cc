#include "GPUmemSvc/GPUmemSvc.h"
#include "SniperKernel/SvcFactory.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>

DECLARE_SERVICE(GPUmemSvc);

GPUmemSvc::GPUmemSvc(const std::string &name)
    : SvcBase(name),
      m_ivar(0)
{
}

bool GPUmemSvc::initialize()
{
    m_ptr = nullptr;
    return true;
}

bool GPUmemSvc::finalize()
{
    free(m_ptr);
    return true;
}

void GPUmemSvc::managed_allocate(size_t num_bytes,
                                 initial_visibility_t initial_visibility = initial_visibility_t::to_all_devices)
{
    void *allocated = nullptr;
    auto flags = (initial_visibility == initial_visibility_t::to_all_devices) ? cudaMemAttachGlobal : cudaMemAttachHost;
    auto status = cudaMallocManaged(&allocated, num_bytes, flags);
    if (is_success(status) && allocated == nullptr)
    {
        status = (status_t)status::unknown;
    }
    return allocated;
}

void GPUmemSvc::free(void *ptr)
{
    auto result = cudaFree(ptr);
}

// Prefetches memory to the specified destination device.
void GPUmemSvc::prefetch(
    void *managed_ptr,
    size_t num_bytes,
    int dstDevice,
    cudaStream_t stream_id)
{
    auto result = cudaMemPrefetchAsync(managed_ptr, num_bytes, dstDevice, stream_id);
}

//Attach memory to a stream asynchronously.
// Enqueues an operation in stream to specify stream association of length bytes of memory starting from devPtr. 
void GPUmemSvc::attach(
    void *managed_ptr,
    cudaStream_t stream_id)
{
    auto result = cudaStreamAttchMemAsync(stream_id,managed_ptr);
}
