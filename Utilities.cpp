#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <map>

class Utilities {
public:
    std::string read_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return "";
        std::ostringstream ss;
        ss << file.rdbuf();
        return ss.str();
    }

    void write_file(const std::string& filename, const std::string& content) {
        std::ofstream file(filename);
        file << content;
    }

    std::vector<std::string> split_words(const std::string& text) {
        std::istringstream iss(text);
        std::vector<std::string> words;
        std::string word;
        while (iss >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            word.erase(std::remove_if(word.begin(), word.end(),
                                      [](char c){ return ispunct(c); }),
                       word.end());
            words.push_back(word);
        }
        return words;
    }
};
