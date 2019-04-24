#include "MFSAlg.h"
#include "SniperKernel/Incident.h"
#include "SniperKernel/AlgFactory.h"

DECLARE_ALGORITHM(MFSAlg);

MFSAlg::MFSAlg(const std::string& name)
    : AlgBase(name),
      m_count(0)
{
}

bool MFSAlg::initialize()
{
    LogDebug << "initialized successfully" << std::endl;
    return true;
}

bool MFSAlg::execute()
{
    ++m_count;
    LogDebug << "loop " << m_count << std::endl;
    CudaRun();
    return true;
}

bool MFSAlg::finalize()
{
    LogDebug << "finalized successfully" << std::endl;
    return true;
}
