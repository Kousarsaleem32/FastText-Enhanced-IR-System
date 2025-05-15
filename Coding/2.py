import fasttext

# File paths
bin_file_path = '/home/kausar/Assignment1/cc.en.300.bin'  # Path to the .bin file
vec_file_path = '/home/kausar/Assignment1/cc.en.300.vec'  # Desired path for the .vec file

# Load the pre-trained FastText model
print("Loading .bin model...")
model = fasttext.load_model(bin_file_path)

# Convert and save as .vec
print(f"Converting {bin_file_path} to {vec_file_path}...")
with open(vec_file_path, 'w') as vec_file:
    # Write the header: number of words and vector dimensions
    words = model.get_words()
    dimension = model.get_dimension()
    vec_file.write(f"{len(words)} {dimension}\n")
    
    # Write each word and its vector
    for word in words:
        vector = model.get_word_vector(word)
        vec_file.write(f"{word} {' '.join(map(str, vector))}\n")

print(f"Conversion complete! Saved to {vec_file_path}")