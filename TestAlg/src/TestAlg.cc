#include "TestAlg.h"
#include "SamplingSvc/SamplingSvc.h"
#include "SniperKernel/AlgFactory.h"

DECLARE_ALGORITHM(TestAlg);

TestAlg::TestAlg(const std::string& name)
    : AlgBase(name)
{
    declProp("Sampling", m_svcname);
}

bool TestAlg::initialize()
{
    //load service 
    SniperPtr<SamplingSvc> svc(getScope(), m_svcname);
    if ( svc.invalid() ) {
        LogError << "can not find the service" << std::endl;
        return false;
    }

    m_svc = svc.data();

    LogDebug << "initialized successfully" << std::endl;
    return true;
}

bool TestAlg::execute()
{
    int total_num = 10000;
    int max_n = 10;
    int max_time = 10;
    size_t nBytes = total_num * max_n * sizeof(double);
    
    double *h_pmt = (double*)malloc(nBytes);
    if (h_pmt == NULL)
    {
        printf("CPU内存分配失败\n");
        exit(0);
    }
    for (int i = 0;i < total_num;i++)
    {
        for (int j = 0;j < max_n;j++)
        {
            h_pmt[i*max_n +j] = 0.1;
        }
    }
    double * h_hittime = (double*)malloc(nBytes);
    for (int i = 0;i < total_num;i++)
    {
        for (int j = 0;j < max_time;j++)
        {
            h_hittime[i*max_time +j] = 0.1;
        }
    }
    double *h_res = (double*)malloc(nBytes);
    m_svc->svc_method(h_pmt,h_hittime,h_res,total_num,nBytes,max_n,max_time);
    

    return true;
}

bool TestAlg::finalize()
{
    LogDebug << "finalized successfully" << std::endl;
    return true;
}
