#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>

class EmbeddingLoader {
public:
    EmbeddingLoader(const std::string& path) {
        load_embeddings(path);
    }

    std::vector<float> get_vector(const std::string& word) const {
        auto it = embeddings.find(word);
        if (it != embeddings.end()) {
            return it->second;
        }
        // fallback: nol semua
        return std::vector<float>(dim, 0.0f);
    }

    int dimension() const { return dim; }

private:
    std::unordered_map<std::string, std::vector<float>> embeddings;
    int dim = 0;

    void load_embeddings(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Cannot open embedding file: " << path << std::endl;
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string word;
            iss >> word;
            std::vector<float> vec;
            float val;
            while (iss >> val) {
                vec.push_back(val);
            }
            if (dim == 0) dim = static_cast<int>(vec.size());
            embeddings[word] = vec;
        }
        std::cerr << "Loaded " << embeddings.size() << " embeddings, dim=" << dim << std::endl;
    }
};
