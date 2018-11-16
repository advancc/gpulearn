#ifndef TESTALG_H
#define TESTALG_H

#include "SniperKernel/AlgBase.h"

class SamplingSvc;

class TestAlg : public AlgBase
{
    public :
        TestAlg(const std::string& name);

        bool initialize();
        bool execute();
        bool finalize();

    private :

        SamplingSvc* m_svc;
        std::string m_svcname;
};

#endif
