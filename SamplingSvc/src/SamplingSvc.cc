#include "SamplingSvc/SamplingSvc.h"
#include "SniperKernel/SvcFactory.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>

DECLARE_SERVICE(SamplingSvc);

SamplingSvc::SamplingSvc(const std::string& name)
    : SvcBase(name),
      m_ivar(0)
{
}

bool SamplingSvc::initialize()
{
    m_handle = dlopen("../lib/simu_so.so", RTLD_LAZY);
    if(!m_handle)
	{
		LogDebug<<dlerror()<<std::endl;
		return -1;
	}
    typedef float (*cdf_sampling_t)(double *h_pmt, double *h_hit, double *h_result,int total_num,int nBytes,int max_n,int max_time);
    m_cdf_sampling_p = (cdf_sampling_t) dlsym(m_handle, "CDF_Sampling_wrapper");
    // m_cdf_sampling_p = dlsym(m_handle, "CDF_Sampling_wrapper");
    if(!m_cdf_sampling_p)
	{
		LogDebug<<dlerror()<<std::endl;
		dlclose(m_handle);
		return -1;
	}
    LogDebug << "initialized successfully" << std::endl;
    return true;
}

bool SamplingSvc::finalize()
{
    LogDebug << "finalized successfully" << std::endl;
    return true;
}

void SamplingSvc::svc_method(double *pmt, double *hitting, double *h_result,int total_num,int nBytes,int max_n,int max_time)
{
    float time_use = m_cdf_sampling_p(pmt, hitting, h_result, total_num, nBytes, max_n, max_time);
    LogDebug << time_use << std::endl;
}
