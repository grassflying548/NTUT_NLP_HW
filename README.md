# Word Embedding Model with PyTorch

This repository provides a simple Python script that demonstrates how to train a word embedding model using the Skip-Gram technique. The code uses PyTorch to build a simple model that converts words into continuous vector representations. While this example is designed for educational purposes, the concepts can serve as a foundation for understanding more advanced language models.

---

## Overview

The script performs the following tasks:

- **Data Preparation:** Defines a small corpus of sentences and builds a vocabulary.
- **Training Data Generation:** Creates training pairs (center word and context word) using a sliding window.
- **Model Definition:** Implements a Skip-Gram model using an embedding layer and a linear layer.
- **Training:** Trains the model using cross-entropy loss to predict context words.
- **Inspection:** Prints the learned embeddings for each word in the vocabulary.

> **Note:** This implementation is intended for educational purposes and uses a very small dataset. In real-world applications (especially for large language models), you would typically work with much larger datasets, advanced batching techniques, and more sophisticated architectures (e.g., transformers).

---

## Files

- **`embedding_train.py`**: The main Python script containing the code.

---

## Requirements

- **Python 3.x**
- **PyTorch**
- **NumPy**

You can install the required packages using `pip`:

```bash
pip install torch numpy
```

---

## How to Run

1. **Clone the repository or download the script.**

2. **Open a terminal in the directory containing the script.**

3. **Run the script:**

   ```bash
   python embedding_train.py
   ```

During execution, the script will:

- Build a vocabulary from the provided sentences.
- Generate training pairs using a sliding window.
- Train the Skip-Gram model for a specified number of epochs.
- Print the training loss at regular intervals.
- Display the learned embeddings for each word.

---

## Code Explanation

1. **Data Preparation:**
   - A list of example sentences is defined.
   - A vocabulary is constructed by mapping each unique word to an index.

2. **Generating Training Data:**
   - The function `generate_training_data` creates (center word, context word) pairs using a sliding window technique over each sentence.

3. **Model Definition:**
   - The `SkipGramModel` class defines a simple neural network with:
     - An **embedding layer**: Maps each word (by index) to an embedding vector.
     - A **linear layer**: Predicts the probability distribution over the vocabulary for the given center word.
   - Cross-entropy loss is used to compare the model's prediction with the true context word.

4. **Training:**
   - The training loop iterates over each training pair.
   - The model makes predictions, computes the loss, and updates the weights using the Adam optimizer.

5. **Inspecting the Embeddings:**
   - After training, the script prints out the learned embedding vectors for each word in the vocabulary.

---

## Customization

- **Corpus:** Modify the `sentences` list to experiment with a different corpus.
- **Window Size:** Adjust the `window_size` parameter in the `generate_training_data` function to change the context range.
- **Embedding Dimension:** Change the `embedding_dim` variable to control the size of the embedding vectors.
- **Training Parameters:** Tweak `learning_rate` and `num_epochs` to experiment with different training configurations.

---

## Further Reading

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Understanding Word2Vec and the Skip-Gram Model](https://www.tensorflow.org/tutorials/text/word2vec)
- [Large Language Models (LLMs)](https://openai.com/research/language-models)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This educational example demonstrates how word embeddings can be learned using a simple Skip-Gram model in PyTorch. For further exploration into large language models, consider studying transformer-based architectures and more advanced training techniques.

Happy coding!