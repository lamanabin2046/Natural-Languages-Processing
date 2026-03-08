English → Nepali Machine Translation (Dash App)

This project implements an English to Nepali Neural Machine Translation (NMT) system using a custom Transformer (Encoder–Decoder) architecture built in PyTorch and deployed as an interactive Dash web application.
The system also provides attention visualization for interpretability.

1. Project Overview

Source Language: English

Target Language: Nepali

Model: Custom Seq2Seq Transformer (Additive / General Attention)

Frontend: Dash (Plotly)

Backend: PyTorch

Tokenization:

English: spaCy tokenizer

Nepali: WordPiece tokenizer

Features:

Live translation via web UI

Greedy decoding

Attention heatmap visualization

GPU support (if available)

2. Folder Structure
A3/
├── code/
│   ├── app.py              # Dash web application
│   ├── code.ipynb          # Training & experimentation notebook
│
├── model/
│   ├── additive_state_dict.pt
│   ├── general_state_dict.pt
│   ├── vocab.pt            # Shared vocabulary
│
├── src/
│   ├── model_def.py        # Transformer architecture & build_model()
│   ├── data_utils.py       # Tokenization & text_transform utilities
│   ├── infer.py            # Greedy decoding & attention extraction
│
├── README.md

3. Model Description

Architecture: Encoder–Decoder Transformer

Hidden Dimension: 256

Encoder Layers: 3

Decoder Layers: 3

Attention Heads: 8

Feedforward Dimension: 512

Attention Types Supported:

Additive Attention

General Attention

Special Tokens:

<unk> = 0

<pad> = 1

<sos> = 2

<eos> = 3

The trained weights are stored in the model/ directory and loaded at runtime.

4. Requirements

Install the required dependencies:

pip install torch torchtext dash plotly pandas spacy nepalitokenizers
python -m spacy download en_core_web_sm


(Optional GPU support requires CUDA-enabled PyTorch.)

5. Running the Application

From the project root directory:

cd code
python app.py


Then open your browser at:

http://127.0.0.1:8050

6. How the Dash App Works

User enters English text in the input box

Text is tokenized and converted into vocabulary indices

The Transformer model performs greedy decoding

Output Nepali tokens are converted back to text

Attention weights are visualized as a heatmap (head 0)

7. Attention Visualization

Displays alignment between source English tokens and generated Nepali tokens

Useful for:

Debugging translation quality

Model interpretability

Academic demonstration

8. Switching Attention Types

To use a different trained model:

In app.py, change:

state_dict = torch.load("model/additive_state_dict.pt")


to:

state_dict = torch.load("model/general_state_dict.pt")


Ensure the configuration matches the checkpoint.

9. Notes & Limitations

Greedy decoding is used (beam search can be added later)

Output length is capped for UI stability

Long or complex sentences may degrade translation quality

Vocabulary is fixed based on training data

10. Future Improvements

Beam search decoding

BLEU score evaluation

Sentence-level batching

FastAPI backend with Dash frontend

Docker-based deployment

Fine-tuning with larger parallel corpora

11. Academic Use

This project is suitable for:

MSc / MTech coursework

Research demonstrations

NLP / Machine Translation experiments

Attention analysis studies

Author

Nabin Gangtan Lama
MSc in Data Science & Artificial Intelligence
Asian Institute of Technology (AIT), Thailand

Demo:
## DEMO
![Machine Translation](demo.gif)
