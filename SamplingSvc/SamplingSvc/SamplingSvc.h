#ifndef SAMPLINGSVC_H
#define SAMPLINGSVC_H

#include "SniperKernel/SvcBase.h"


class SamplingSvc : public SvcBase
{
    public :
        SamplingSvc(const std::string& name);
        // SamplingSvc(const std::basic_string<char, std::char_traits<char>, std::allocator<char> >&);

        bool initialize();
        bool finalize();

        void svc_method(double *h_pmt, double *h_hit, double *h_result,int total_num,int nBytes,int max_n,int max_time);

    private :
        void *m_handle;
        int m_ivar;
        float (*m_cdf_sampling_p)(double*, double*, double*,int ,int ,int ,int);
        
};

#endif
