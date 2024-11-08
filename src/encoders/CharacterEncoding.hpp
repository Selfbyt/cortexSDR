class CharacterEncoding {
public:
    CharacterEncoding();
    std::vector<size_t> encodeText(const std::string& text);
    std::string decodeIndices(const std::vector<size_t>& indices);

private:
    static constexpr size_t CHARS_PER_POSITION = 4;
    size_t getCharacterPosition(char c) const;
}; 