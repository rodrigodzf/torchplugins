#include "c74_min.h"
#include "twoPole.h"

#ifndef VERSION
#define VERSION "0.0.0"
#endif

using namespace c74::min;

class Filterbank : public object<Filterbank>, public sample_operator<1, 1>
{

public:
    MIN_DESCRIPTION	{ "Filterbank." };
    MIN_TAGS		{ "audio" };
    MIN_AUTHOR		{ "Rodrigo Diaz" };
    MIN_RELATED		{ "biquad~" };

    inlet<> input           { this, "(signal) input", "signal" };
    outlet<> output         { this, "(signal) output", "signal" };
    outlet<> output_list    { this, "(list) dump output" };
    
    message<> dump { this, "dump", "Dump the filterbank coefficients to the Max window.",
        MIN_FUNCTION {
            int n_parallel = m_filters.size();
            int n_biquads = m_filters[0].size();
            std::vector<std::vector<double>> coefficients;
            for (int i = 0; i < n_parallel; i++)
            {
                for (int j = 0; j < n_biquads; j++)
                {
                    std::vector<double> coeffs = m_filters[i][j].get_coefficients();
                    coefficients.push_back(coeffs);

                }
            }

            std::vector<atom> coeffs = to_atoms(coefficients);
            output_list.send(coeffs);

            return {};
        }
    };

    message<> list { this, "list", "Input to the convolution function.",
        MIN_FUNCTION 
        {
            lock lock {m_mutex};
            auto coefficients = from_atoms<std::vector<double>>(args);

            // initialize coefficients
            int n_parallel = m_filters.size();
            int n_biquads = m_filters[0].size();
            int stride = n_biquads * 3;

            // cout << "n_parallel: " << n_parallel << endl;
            // cout << "n_biquads: " << n_biquads << endl;
            // cout << "stride: " << stride << endl;

            for (int i = 0; i < n_parallel; i++)
            {
                for (int j = 0; j < n_biquads; j++)
                {
                    m_filters[i][j].set_coefficients(
                        coefficients[i * n_biquads * stride + j * stride + 0],
                        coefficients[i * n_biquads * stride + j * stride + 1],
                        coefficients[i * n_biquads * stride + j * stride + 2],
                        coefficients[i * n_biquads * stride + j * stride + 4],
                        coefficients[i * n_biquads * stride + j * stride + 5]
                    );
                }
            }
            return {};
        }
    };


    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION
        {
            cout << "filterbank~ - " << VERSION << " - 2022 - Rodrigo Diaz" << endl;
            return {};
        }
    };


public:

    Filterbank(const atoms &args = {});
    ~Filterbank();
    sample operator()(sample x);
    std::vector<std::vector<TwoPole<double>>> m_filters;

private:
    std::mutex m_mutex;
};

Filterbank::Filterbank(const atoms &args)
{   
    if (args.empty()) {
        error("Please provide a number of parallel filters.");
        return;
    }

    int n_parallel = int(args[0]);
    int n_biquads = int(args[1]);
    
    cout << "n_parallel: " << n_parallel << endl;
    cout << "n_biquads: " << n_biquads << endl;

    m_filters.resize(n_parallel);
    for (int i = 0; i < n_parallel; i++) {
        m_filters[i].resize(n_biquads);
    }

    // initialize coefficients
    for (int i = 0; i < n_parallel; i++) {
        for (int j = 0; j < n_biquads; j++) {
            m_filters[i][j].set_coefficients(0.0, 0.0, 0.0, 0.0, 0.0);
        }
    }

}

Filterbank::~Filterbank(){}

sample Filterbank::operator()(sample x)
{
    // we need to process in parallel for each filter
    sample out = 0.0;
    for (int i = 0; i < m_filters.size(); i++)
    {
        // for each filter
        double y = x;
        for (int j = 0; j < m_filters[i].size(); j++)
        {
            y = m_filters[i][j].process(y);
        }

        // add to output
        out += y;
    }

    return out;
}

MIN_EXTERNAL(Filterbank);
