#include "c74_min.h"

#include "PytorchFrontend.h"
#include "ShapeFFT.h"
#include "log.h"

#ifndef VERSION
#define VERSION "0.2.0"
#endif

using namespace c74::min;

class FC : public object<FC> {
public:
    MIN_DESCRIPTION	{"Loads and runs a NN."};
    MIN_TAGS		{"torch"};
    MIN_AUTHOR		{"Rodrigo Diaz"};
    MIN_RELATED		{"print"};

    FC(const atoms &args = {});
    ~FC();

	inlet<>		points	{ this, "(list) flattened points", "list" };
	inlet<>		scaling	{ this, "(list) scaling", "list" };
	outlet<>	output	{ this, "(list) features" };
    
    message<> load { this, "load", "Load a model from a file",
        MIN_FUNCTION {
            if (args.size() == 1)
            {
                auto modelPath = std::string(args[0]);
                loadModel(modelPath);
            }
            return {};
        }
    };
    
    attribute<int> out_features
    { 
        this,
        "out_features",
        32 * 1 * 6,
        description {"Number of output features."},
    };

    message<> set { this, "set", "Set a parameter of the model.",
        MIN_FUNCTION {
            if (args.size() == 2)
            {
                auto name = std::string(args[0]);
                auto value = args[1];

                lock lock {m_mutex};
                // Model::getInstance().callMethod(name, value);
                lock.unlock();
            }
            return {};
        }
    };
    message<> list { this, "list", "Input to the network.",
        MIN_FUNCTION {
            lock lock {m_mutex};

            if (!m_loaded)
            {
                cout << "Model not loaded." << endl;
                return {};
            }

            if (m_output.size() != out_features)
            {
                m_output.resize(out_features);
            }

            // left inlet is for the geometry
            if (inlet == 0)
            {
                std::vector<float> points = from_atoms<std::vector<float>>(args);

                for (int i = 0; i < 64; i++)
                {
                    points_x[i] = points[i];
                    points_y[i] = points[i + 64];
                }

                fft->fft_magnitude(points_x, points_y);

                std::vector<float> features(
                    std::begin(fft->mag),
                    std::end(fft->mag)
                );

                nn->process(features, g_coords);
                nn->scale(g_scaling);
            }
            else // right inlet is for the scaling
            {
                g_scaling = from_atoms<std::vector<float>>(args);
                g_scaling[2] = g_scaling[2] * 1e+9F; // Young's modulus from GPa to Pa
                g_scaling[4] = g_scaling[4] * 1e-8F; // beta damping scaling
                g_scaling[5] = 1.0F / g_scaling[5]; // sampling rate
                nn->scale(g_scaling);
            }

            m_output = nn->coefficients;

            lock.unlock();
            output.send(to_atoms(m_output));

            return {};
        }
    };

    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION
        {
            cout << "fc - " << VERSION << " - 2023 - Rodrigo Diaz" << endl;
            return {};
        }
    };
private:
    void loadModel(const std::string &path);

private:
    bool m_loaded {false};
    std::mutex m_mutex;
    std::unique_ptr<PytorchFrontend> nn;
    std::unique_ptr<ShapeFFT> fft;
    FixedPointsArray points_x;
    FixedPointsArray points_y;

    std::vector<float> m_output;
    std::vector<float> g_coords {0.70238614, 0.4862546};
    std::vector<float> g_scaling {
		1.0F,			   // scale factor
		15000.0F,		   // rho
		8000000000.0F,	   // E
		1.0F,			   // alpha
		0.0000003F,		   // beta
		1.0F / 32000.0F	   // sampling period
	};
};

FC::FC(const atoms &args)
{
    nn = std::make_unique<PytorchFrontend>();
    fft = std::make_unique<ShapeFFT>();

    if (!args.empty())
    {
        auto modelPath = std::string(args[0]);
        loadModel(modelPath);
    }
}

void FC::loadModel(const std::string &path)
{
    if (!nn->load(path)) {
        cerr << "Model not loaded." << endl;
    }
    else
    {
        m_loaded = true;
        cout << "model with path: " << path << " loaded" << endl;
    }
}

FC::~FC() {}

MIN_EXTERNAL(FC);
