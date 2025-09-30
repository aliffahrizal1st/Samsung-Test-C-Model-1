#include "Utilities.cpp"
#include "Classification.cpp"
#include <iostream>
#include "config.h"

Classification clf(MODEL_JSON, EMBEDDING_VEC);

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: TextClassification <input_file.txt> <output_file.txt>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    Utilities util;
    std::string input_text = util.read_file(input_file);

    auto words = util.split_words(input_text);
    std::string prediction = clf.predict(words, 1);

    util.write_file(output_file, prediction);

    return 0;
}
