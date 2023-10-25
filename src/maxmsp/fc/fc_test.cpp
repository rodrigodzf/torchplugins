#include "c74_min_unittest.h"   // required unit test header
#include "fc.cpp"


SCENARIO("object produces correct output") {
    ext_main(nullptr);    // every unit test must call ext_main() once to configure the class

    GIVEN("Instantiating a fc object")
    {
        atoms args = {4, 2};
        FC fc = FC();

        std::vector<float> points = {
            0.6670909,  0.6317929,  0.5964949,  0.5611969,  0.5258989,  0.4906009,
            0.4553029,  0.4200049,  0.38470688, 0.3494089,  0.31411088, 0.2788129,
            0.24351488, 0.20821688, 0.17291887, 0.13762087, 0.12366271, 0.11611941,
            0.10857611, 0.10103282, 0.12304196, 0.1511757,  0.17930943, 0.20744316,
            0.2355769,  0.26371062, 0.29184437, 0.3199781,  0.34811184, 0.37645793,
            0.40781644, 0.43917498, 0.4705335,  0.5026827,  0.54357654, 0.58447045,
            0.61232543, 0.6392445,  0.6661636,  0.6930827,  0.72000176, 0.7448338,
            0.7694617,  0.79408956, 0.8187174,  0.83837587, 0.85631174, 0.8740561,
            0.89127254, 0.8997585,  0.8966665,  0.8935746,  0.8904826,  0.8873906,
            0.8842986,  0.88120663, 0.87811464, 0.87302023, 0.8470613,  0.8211023,
            0.7911741,  0.7498131,  0.708452,   0.6670909,
            0.9445403,  0.92261666, 0.900693,   0.87876934, 0.8568457,  0.834922,
            0.81299835, 0.79107463, 0.769151,   0.7472273,  0.72530365, 0.70338,
            0.6814563,  0.65953267, 0.637609,   0.6156853,  0.57920057, 0.53833866,
            0.49747676, 0.45661485, 0.42427042, 0.39369118, 0.36311194, 0.33253273,
            0.3019535,  0.27137426, 0.24079503, 0.2102158,  0.17963658, 0.14927578,
            0.12201335, 0.09475093, 0.06748851, 0.04309765, 0.05046566, 0.05783366,
            0.08786093, 0.11951467, 0.1511684,  0.18282214, 0.21447587, 0.24778159,
            0.28124896, 0.31471634, 0.34818372, 0.38463232, 0.42211434, 0.45968664,
            0.49747592, 0.5378278,  0.57926494, 0.6207021,  0.6621392,  0.7035763,
            0.7450135,  0.7864506,  0.8278877,  0.8685375,  0.9009833,  0.9334291,
            0.95648706, 0.9525048,  0.94852257, 0.9445403
        };

        atoms pz = to_atoms(points);

        fc.list(pz); // wont produce any out
        // auto& out = *c74::max::object_getoutput(fc, 0);
        // REQUIRE(out.size() == 0);

        // Load a model
        atom path("extras/pretrained/optimized_curious-salad-167.pt");
        fc.load(path);

        // now proceed to testing various sequences of events
        WHEN("list is passed") {
            //! We need to check that the input size equals the input size of the model
            std::cout << "Input size: " << points.size() << std::endl;

            fc.list(pz); // will produce output
            THEN("will produce output") {
                // TODO: This is not working
                // auto& output = *c74::max::object_getoutput(fc, 0);
                // std::cout << output[0][0] << std::endl;
                // REQUIRE((output.size() == 1));
                // REQUIRE((output[0].size() == 1));
                // REQUIRE((output[0][0] == symbol("hello world")));
            }
        }
    }
}
