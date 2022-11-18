#include "c74_min_unittest.h"   // required unit test header
#include "filterbank_tilde.cpp"


SCENARIO("object produces correct output") {
    ext_main(nullptr);    // every unit test must call ext_main() once to configure the class

    GIVEN("Instantiating a filterbank~ object")
    {
        atoms args = {4, 2};
        Filterbank filterbank = Filterbank(args);

        REQUIRE(filterbank.m_filters.size() == 4);
        REQUIRE(filterbank.m_filters[0].size() == 2);

        WHEN("Receiving a list of coefficients")
        {
            std::vector<float> coefficients = {
                1.0, 0.1, 0.2, 1.0, 0.4, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
            };
            atoms coeffs = to_atoms(coefficients);

            filterbank.list(coeffs);

            std::cout << "filterbank.m_filters[0][0].get_coefficients() " << filterbank.m_filters[0][0].to_string() << std::endl;
            

            REQUIRE(filterbank.m_filters[0][0].get_coefficients()[0] == 1.0f);
            REQUIRE(filterbank.m_filters[0][0].get_coefficients()[1] == 0.1f);
            REQUIRE(filterbank.m_filters[0][0].get_coefficients()[2] == 0.2f);
            REQUIRE(filterbank.m_filters[0][0].get_coefficients()[3] == 0.4f);
            REQUIRE(filterbank.m_filters[0][0].get_coefficients()[4] == 0.5f);
                
        }

    }
}
