# Transformer_Text
Implementing a Transformer model in PyTorch,  training the model with Tiny Shakespeare dataset, and generating text.
The generated text mimcs Shakespeare's writing style.

The model is capable of generating text in Shakespearean writing style given a short prompt such as "ROMEO:" or "JULIET:".
.PY FILES:
model.py: defines the Transformer model (embeddings, positional encoding, multi-head self-attention, feed-forward blocks, normalization, residuals, final projection).

train.py: prepares the dataset, trains the model with AdamW and cross-entropy loss, and saves the trained checkpoint.

generate.py: loads the trained model and vocabulary, and generates text samples from a given prompt.

INSTRUCTIONS
Run model.py
Train the model by running train.py.
This will:
Save the best model weights to best_transformer.pt
Save vocabulary files (stoi.pt, itos.pt)
Log training and validation loss per epoch to training_log.csv

Generate samples from prompts "ROMEO:" and "JULIET:" by running generate.py
This will save outputs to sample_ROMEO.txt and sample_JULIET.txt.

SUMMARY OF RESULTS
The model was trained for 5 epochs on a T4 GPU. Training loss steadily decreased from 1.8 to 1.03,
while validation loss reached a minimum of 1.57 before slightly increasing (due to overfitting).
The generated samples show that the model mimics Shakespearean writing style.





