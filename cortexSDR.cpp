#include <iostream>
#include <bitset>
#include <unordered_map>

class SparseDistributedRepresentation {
public:
    SparseDistributedRepresentation(std::initializer_list<std::string> vocabulary)
        : vocabulary_(vocabulary), vectorSize_(vocabulary.size()) {
        initializeWordIndices();
    }

    void encodeText(const std::string& text) {
        std::bitset<MAX_VECTOR_SIZE> encodedVector;
        for (const auto& word : splitText(text)) {
            auto it = wordIndices_.find(word);
            if (it != wordIndices_.end()) {
                encodedVector.set(it->second);
            }
        }
        encodedVector_ = encodedVector;
    }

    void printEncodedVector() const {
        std::cout << encodedVector_ << std::endl;
    }

private:
    static const size_t MAX_VECTOR_SIZE = 2000; 
    std::bitset<MAX_VECTOR_SIZE> encodedVector_;
    std::unordered_map<std::string, size_t> wordIndices_;
    std::vector<std::string> vocabulary_;
    size_t vectorSize_;

    void initializeWordIndices() {
        size_t index = 0;
        for (const auto& word : vocabulary_) {
            wordIndices_[word] = index++;
        }
    }

    std::vector<std::string> splitText(const std::string& text) const {
        // Split the text into words (simple example, you might want to handle punctuation, etc.)
        std::vector<std::string> words;
        size_t start = 0;
        size_t end = text.find(' ');
        while (end != std::string::npos) {
            words.push_back(text.substr(start, end - start));
            start = end + 1;
            end = text.find(' ', start);
        }
        words.push_back(text.substr(start));
        return words;
    }
};

int main() {
    SparseDistributedRepresentation sdr({"hello", "my", "name", "is", "henry", "Wolfe", "Gange"});

    // Encode a text
    sdr.encodeText("hello my name is henry Wolfe Gange");

    // Print the encoded vector
    sdr.printEncodedVector();

    return 0;
}

