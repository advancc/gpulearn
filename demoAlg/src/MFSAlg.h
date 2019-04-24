#ifndef MFSALG_H
#define MFSALG_H

#include "SniperKernel/AlgBase.h"
#include <vector>
#include "test.cuh"

class MFSAlg : public AlgBase
{
    public :
        MFSAlg(const std::string& name);

        bool initialize();
        bool execute();
        bool finalize();

    private :

        int m_count;
};

#endif