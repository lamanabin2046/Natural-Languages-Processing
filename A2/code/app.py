# ==============================
# Dash + Bootstrap LSTM LM App
# ==============================

import pickle
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# ------------------------------------------------
# 1. Device
# ------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ------------------------------------------------
# 2. Load vocab
# ------------------------------------------------
with open("../model/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

stoi = vocab["stoi"]
itos = vocab["itos"]

PAD = stoi.get("<pad>", 0)
UNK = stoi.get("<unk>", 1)
vocab_size = len(itos)

# ------------------------------------------------
# 3. Model
# ------------------------------------------------
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, state=None):
        emb = self.embedding(x)
        out, state = self.lstm(emb, state)
        logits = self.fc(out)
        return logits, state

# ------------------------------------------------
# 4. Load checkpoint
# ------------------------------------------------
checkpoint = torch.load("../model/lstm_language_model.pt", map_location=device)

model = LSTMLanguageModel(
    vocab_size=vocab_size,
    emb_dim=256,
    hidden_dim=1024,
    num_layers=3,
    dropout=0.3,
    pad_idx=PAD,
).to(device)

model.load_state_dict(checkpoint["state_dict"])
model.eval()

print("✅ Model loaded")

# ------------------------------------------------
# 5. Tokenizer
# ------------------------------------------------
def word_tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", text)

def encode_tokens(tokens):
    return [stoi.get(t, UNK) for t in tokens]

def decode_tokens(ids):
    return [itos[i] for i in ids]

PUNCT_NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", "%", ")", "]", "}"}
PUNCT_NO_SPACE_AFTER = {"(", "[", "{"}

def detokenize(tokens):
    out = []
    for t in tokens:
        if not out:
            out.append(t)
        elif t in PUNCT_NO_SPACE_BEFORE:
            out[-1] += t
        elif out[-1] in PUNCT_NO_SPACE_AFTER:
            out[-1] += t
        else:
            out.append(" " + t)
    return "".join(out)

# ------------------------------------------------
# 6. Text post-processing
# ------------------------------------------------
def capitalize_sentences(text: str) -> str:
    """
    Capitalize the first letter of each sentence.
    """
    text = text.strip()
    if not text:
        return text

    # Capitalize very first character
    text = text[0].upper() + text[1:]

    # Capitalize after . ? !
    return re.sub(
        r'([.!?]\s+)([a-z])',
        lambda m: m.group(1) + m.group(2).upper(),
        text
    )

# ------------------------------------------------
# 7. Generation
# ------------------------------------------------
def sample_top_k(logits, k):
    values, indices = torch.topk(logits, k)
    probs = F.softmax(values, dim=-1)
    choice = torch.multinomial(probs, 1)
    return indices.gather(-1, choice)

@torch.no_grad()
def generate_text(seed_text, max_new_tokens, temperature, top_k):
    seed_ids = encode_tokens(word_tokenize(seed_text)) or [UNK]
    x = torch.tensor([seed_ids], dtype=torch.long).to(device)
    _, state = model(x)

    last_id = x[:, -1:]
    generated = seed_ids[:]

    for _ in range(max_new_tokens):
        logits, state = model(last_id, state)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if 1 <= top_k < logits.size(-1):
            next_id = sample_top_k(logits, top_k)
        else:
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)

        generated.append(next_id.item())
        last_id = next_id

    text = detokenize(decode_tokens(generated))
    return capitalize_sentences(text)

# ------------------------------------------------
# 8. Dash App (Styled Pro UI)
# ------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
)

app.layout = dbc.Container(
    fluid=True,
    style={
        "minHeight": "100vh",
        "backgroundColor": "#050511",
        "paddingTop": "60px",
    },
    children=[
        dbc.Row(
            justify="center",
            children=[
                dbc.Col(
                    width=7,
                    children=[

                        html.H2(
                            "LSTM Language Model",
                            className="text-center mb-4",
                            style={"color": "#eaeaea"},
                        ),

                        # Prompt card
                        dbc.Card(
                            dbc.CardBody([
                                dbc.Label(
                                    "Prompt",
                                    style={
                                        "fontSize": "25px",
                                        "color": "#f5f5f5",
                                        "marginBottom": "20px",
                                    },
                                ),
                                dcc.Textarea(
                                    id="seed-text",
                                    value="alice was",
                                    style={
                                        "width": "100%",
                                        "height": "90px",
                                        "backgroundColor": "#1b1f2a",
                                        "color": "#f5f5f5",
                                        "border": "1px solid #3a3f4b",
                                        "padding": "10px",
                                        "fontFamily": "Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
                                        "fontSize": "20px",
                                        "borderRadius": "10px",
                                    },
                                ),
                            ]),
                            className="mb-3",
                            style={"backgroundColor": "#141824"},
                        ),

                        # Controls card
                        dbc.Card(
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Temperature", style={"fontSize": "25px"}),
                                        dbc.Input(
                                            id="temperature",
                                            type="number",
                                            value=0.7,
                                            step=0.1,
                                            style={
                                                "backgroundColor": "#1b1f2a",
                                                "color": "#f5f5f5",
                                                "border": "1px solid #3a3f4b",
                                            },
                                        ),
                                    ]),
                                    dbc.Col([
                                        dbc.Label("Top-k", style={"fontSize": "25px"}),
                                        dbc.Input(
                                            id="top-k",
                                            type="number",
                                            value=20,
                                            step=1,
                                            style={
                                                "backgroundColor": "#1b1f2a",
                                                "color": "#f5f5f5",
                                                "border": "1px solid #3a3f4b",
                                            },
                                        ),
                                    ]),
                                    dbc.Col([
                                        dbc.Label("Max tokens", style={"fontSize": "25px"}),
                                        dbc.Input(
                                            id="max-tokens",
                                            type="number",
                                            value=100,
                                            step=10,
                                            style={
                                                "backgroundColor": "#1b1f2a",
                                                "color": "#f5f5f5",
                                                "border": "1px solid #3a3f4b",
                                            },
                                        ),
                                    ]),
                                    dbc.Col(
                                        dbc.Button(
                                            "Generate",
                                            id="generate-btn",
                                            color="primary",
                                            className="mt-4 w-100",
                                        ),
                                    ),
                                ]),
                            ]),
                            className="mb-3",
                            style={"backgroundColor": "#141824"},
                        ),

                        # Output card
                        dbc.Card(
                            dbc.CardBody([
                                dbc.Label(
                                    "Generated Text",
                                    style={
                                        "fontSize": "25px",
                                        "color": "#f5f5f5",
                                        "marginBottom": "20px",
                                    },
                                ),
                                html.Div(
                                    id="output-text",
                                    className="generated-text",
                                    style={
                                        "whiteSpace": "pre-wrap",
                                        "fontFamily": "Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
                                        "fontSize": "20px",
                                        "lineHeight": "1.8",
                                        "color": "#D5DDDF",
                                        "minHeight": "300px",
                                        "textAlign": "justify",
                                    },
                                ),
                            ]),
                            style={"backgroundColor": "#10131a"},
                        ),
                    ],
                )
            ],
        )
    ],
)

# ------------------------------------------------
# 9. Callback
# ------------------------------------------------
@app.callback(
    Output("output-text", "children"),
    Input("generate-btn", "n_clicks"),
    State("seed-text", "value"),
    State("temperature", "value"),
    State("top-k", "value"),
    State("max-tokens", "value"),
)
def run_generation(n, seed, temperature, top_k, max_tokens):
    if not n:
        return ""

    return generate_text(
        seed_text=seed,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
    )

# ------------------------------------------------
# 10. Run
# ------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
