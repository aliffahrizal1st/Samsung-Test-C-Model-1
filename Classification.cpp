#include <fdeep/fdeep.hpp>
#include <string>
#include <vector>
#include <sstream>
#include <numeric>
#include "EmbeddingLoader.cpp" 

class Classification {
public:
    Classification(const std::string& model_path,
                   const std::string& embedding_path)
        : model(fdeep::load_model(model_path)), loader(embedding_path) {
        embedding_dim = loader.dimension();
    }

    std::string predict(const std::vector<std::string>& words, int id = 1) {
        std::vector<float> avg(embedding_dim, 0.0f);
        int count = 0;
        for (auto& w : words) {
            auto vec = loader.get_vector(w);
            for (int i = 0; i < embedding_dim; i++) {
                avg[i] += vec[i];
            }
            count++;
        }
        if (count > 0) {
            for (int i = 0; i < embedding_dim; i++) {
                avg[i] /= count;
            }
        }

        const auto input_tensor = fdeep::tensor(
            fdeep::tensor_shape(static_cast<std::size_t>(embedding_dim)),
            avg
        );

        const auto result = model.predict({input_tensor});
        auto vec = result.front().as_vector_float();
        
        int label_id = std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
        std::string label;
        if (label_id == 0) label = "ham";
        else if (label_id == 1) label = "spam";
        else label = "other";

        std::ostringstream oss;
        oss << id << " " << label;
        return oss.str();
    }

private:
    fdeep::model model;
    EmbeddingLoader loader;
    int embedding_dim;
};
