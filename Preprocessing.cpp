#include <string>
#include <iostream>
#include <fdeep/fdeep.hpp>

using namespace std;

class Preprocessing
{
private:
    string data_path = "";
public:
    Preprocessing() = default;
    Preprocessing(const string& path) : data_path(path) {}
    ~Preprocessing() = default;

    void train_model()
    {
        fdeep::model model = fdeep::load_model(data_path + "model.json");
    }
};