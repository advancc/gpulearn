#include <iostream>
#include "stdio.h"
#include "MuonSimu.cuh"
#include "../np/NP.hh"

using namespace std;

class MuonSimu
{
  public:
    int m_pmt_num;
    NP<double> *m_data_step;
    NP<double> *m_data_pmt_x;
    NP<double> *m_data_pmt_y;
    NP<double> *m_data_pmt_z;
    NP<double> *m_data_hit;
    NP<double> *m_data_npe;
    vector<int> m_seed;
    MuonSimu(string path, string filename);
    MuonSimu(string path, string filename, int pmtnum);
    vector<int> CreateSeed(vector<int> &seed, int seed_num);
    void TestDataLoad();
    void run();
};

MuonSimu::MuonSimu(string path, string filename)
{
    m_pmt_num = 17746;
    m_data_step = NP<double>::Load((path + filename).c_str());
    m_data_pmt_x = NP<double>::Load((path + "pmt_x.npy").c_str());
    m_data_pmt_y = NP<double>::Load((path + "pmt_y.npy").c_str());
    m_data_pmt_z = NP<double>::Load((path + "pmt_z.npy").c_str());
    m_data_hit = NP<double>::Load((path + "hittime_cdf.npy").c_str());
    m_data_npe = NP<double>::Load((path + "npe_cdf.npy").c_str());
    CreateSeed(m_seed, m_pmt_num);
}
MuonSimu::MuonSimu(string path, string filename, int pmtnum)
{
    m_pmt_num = pmtnum;
    m_data_step = NP<double>::Load((path + filename).c_str());
    m_data_pmt_x = NP<double>::Load((path + "pmt_x.npy").c_str());
    m_data_pmt_y = NP<double>::Load((path + "pmt_y.npy").c_str());
    m_data_pmt_z = NP<double>::Load((path + "pmt_z.npy").c_str());
    m_data_hit = NP<double>::Load((path + "hittime_cdf.npy").c_str());
    m_data_npe = NP<double>::Load((path + "npe_cdf.npy").c_str());
    m_seed = CreateSeed(m_seed, m_pmt_num);
}
vector<int> MuonSimu::CreateSeed(vector<int> &seed, int seed_num)
{
    for (int i = 1; i <= seed_num; i++)
    {
        seed.push_back(i);
    }
    return seed;
}
void MuonSimu::TestDataLoad()
{
    double *data_step = this->m_data_step->values();
    for (int i = 0; i < 10; i++)
    {
        cout << data_step[i] << endl;
    }
    cout << "data_step number:" << this->m_data_step->num_values() << endl;
    cout << "desc()" << this->m_data_step->desc() << endl;
    vector<int> a = this->m_data_step->shape;
    for (int i = 0; i < a.size(); i++)
    {
        cout << a[i] << endl;
    }
    vector<double> v_data_pmt_x = this->m_data_pmt_x->data;
    vector<double> v_data_pmt_y = this->m_data_pmt_y->data;
    vector<double> v_data_pmt_z = this->m_data_pmt_z->data;
    vector<double> v_data_npe = this->m_data_npe->data;
    vector<double> v_data_hit = this->m_data_hit->data;
    cout << "data_pmt_x size:" << v_data_pmt_x.size() << endl;
    cout << "data_npe size:" << v_data_npe.size() << endl;
    cout << "data_hit size:" << v_data_hit.size() << endl;
    return;
}
void MuonSimu::run()
{
    vector<double> v_data_step = this->m_data_step->data;
    double *v_data_step_1 = new double[this->m_data_step->shape[1]];
    double *v_data_step_2 = new double[this->m_data_step->shape[1]];
    double *v_data_step_3 = new double[this->m_data_step->shape[1]];
    double *v_data_step_4 = new double[this->m_data_step->shape[1]];
    double *v_data_step_5 = new double[this->m_data_step->shape[1]];
    double *v_data_step_6 = new double[this->m_data_step->shape[1]];
    double *v_data_step_7 = new double[this->m_data_step->shape[1]];
    int *v_data_step_8 = new int[this->m_data_step->shape[1]];
    for (int i = 0; i < this->m_data_step->shape[1]; i++)
    {
        v_data_step_1[i] = v_data_step[i];
        v_data_step_2[i] = v_data_step[i + 1 * this->m_data_step->shape[1]];
        v_data_step_3[i] = v_data_step[i + 2 * this->m_data_step->shape[1]];
        v_data_step_4[i] = v_data_step[i + 3 * this->m_data_step->shape[1]];
        v_data_step_5[i] = v_data_step[i + 4 * this->m_data_step->shape[1]];
        v_data_step_6[i] = v_data_step[i + 5 * this->m_data_step->shape[1]];
        v_data_step_7[i] = v_data_step[i + 6 * this->m_data_step->shape[1]];
        v_data_step_8[i] = (int)v_data_step[i + 7 * this->m_data_step->shape[1]];
    }
    double *v_data_pmt_x = new double[this->m_data_pmt_x->data.size()];
    double *v_data_pmt_y = new double[this->m_data_pmt_y->data.size()];
    double *v_data_pmt_z = new double[this->m_data_pmt_z->data.size()];
    double *v_data_hit = new double[this->m_data_hit->data.size()];
    double *v_data_npe = new double[this->m_data_npe->data.size()];
    int *v_seed = new int[this->m_seed.size()];
    memcpy(v_data_pmt_x, &(this->m_data_pmt_x->data[0]), this->m_data_pmt_x->data.size() * sizeof(double));
    memcpy(v_data_pmt_y, &(this->m_data_pmt_y->data[0]), this->m_data_pmt_y->data.size() * sizeof(double));
    memcpy(v_data_pmt_x, &(this->m_data_pmt_z->data[0]), this->m_data_pmt_z->data.size() * sizeof(double));
    memcpy(v_data_hit, &(this->m_data_hit->data[0]), this->m_data_hit->data.size() * sizeof(double));
    memcpy(v_data_npe, &(this->m_data_npe->data[0]), this->m_data_npe->data.size() * sizeof(double));
    memcpy(v_seed, &(this->m_seed[0]), this->m_seed.size() * sizeof(int));

    int size[] = {
        this->m_data_step->shape[1] * sizeof(double),
        this->m_data_pmt_x->data.size() * sizeof(double),
        this->m_data_hit->data.size() * sizeof(double),
        this->m_data_npe->data.size() * sizeof(double),
        m_seed.size() * sizeof(int)};
    double *result = new double[this->m_pmt_num * 2000];
    int *evt_npe_result = new int[1000];
    float total_time = GPU_Sampling_wrapper(v_data_step_1, v_data_step_2, v_data_step_3, v_data_step_4,
                                            v_data_step_5, v_data_step_6, v_data_step_7, v_data_step_8,
                                            1000, v_data_pmt_x, v_data_pmt_y, v_data_pmt_z, v_data_hit,
                                            v_data_npe, v_seed, size, result, evt_npe_result);
 
    int sum = 0;
    for (int i = 0; i < 1000; i++)
    {
        // cout << evt_npe_result[i]<<endl;
        sum += evt_npe_result[i];
    }
    cout << "mean totalPE:" << sum / 1000 << endl;
}

int main()
{
    cout << "---------" << endl;
    cout << "begin run" << endl;
    MuonSimu *simu = new MuonSimu("../../data/", "20190109_data.npy");
    simu->TestDataLoad();
    simu->run();
    return 0;
}
