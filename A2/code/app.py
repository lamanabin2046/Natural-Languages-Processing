import os, sys
import torch
import torchtext

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(_file_))))

from dash import Dash, dcc, html, Input, Output, State
from torchtext.data.utils import get_tokenizer
from nepalitokenizers import WordPiece

from src.model_def import build_model
from src.data_utils import make_text_transform
from src.infer import greedy_decode


# -------------------- Model + Tokenizers --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SRC_LANG = "en"
TARG_LANG = "ne"

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

vocab_transform = torch.load("model/vocab.pt", map_location="cpu")

token_transform = {}
token_transform[SRC_LANG] = get_tokenizer("spacy", language="en_core_web_sm")
token_transform[TARG_LANG] = WordPiece()

text_transform = make_text_transform(
    token_transform=token_transform,
    vocab_transform=vocab_transform,
    SRC_LANG=SRC_LANG,
    TARG_LANG=TARG_LANG,
    SOS_IDX=SOS_IDX,
    EOS_IDX=EOS_IDX,
)

config = {
    "HID_DIM": 256,
    "ENC_LAYERS": 3,
    "DEC_LAYERS": 3,
    "ENC_HEADS": 8,
    "DEC_HEADS": 8,
    "ENC_PF_DIM": 512,
    "DEC_PF_DIM": 512,
    "ENC_DROPOUT": 0.1,
    "DEC_DROPOUT": 0.1,
    "ATTEN_TYPE": "additive",
    "MAX_LEN": 5000,  # must match checkpoint pos_embedding sizes
    "SRC_PAD_IDX": PAD_IDX,
    "TRG_PAD_IDX": PAD_IDX,
}

INPUT_DIM = len(vocab_transform[SRC_LANG])
OUTPUT_DIM = len(vocab_transform[TARG_LANG])

model = build_model(config, INPUT_DIM, OUTPUT_DIM, DEVICE)
state_dict = torch.load("model/additive_state_dict.pt", map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

ne_itos = vocab_transform[TARG_LANG].get_itos()


def decode_ids_to_tokens(pred_ids, eos_idx=EOS_IDX, max_out_tokens=25):
    """
    Robust decoding:
    - stop at EOS id
    - cap output tokens for UI
    - stop if model loops repeating tokens
    """
    tokens = []
    ids = pred_ids.tolist()

    for t in ids:
        if t == eos_idx:
            break
        if t in (SOS_IDX, PAD_IDX):
            continue

        tok = ne_itos[t]
        tokens.append(tok)

        if len(tokens) >= max_out_tokens:
            break

        # repetition guard
        if len(tokens) >= 12 and len(set(tokens[-8:])) <= 2:
            break

    return tokens


# -------------------- Dash UI (New Layout) --------------------
app = Dash(_name_)
app.title = "EN → NE Translator"

CARD_STYLE = {
    "background": "white",
    "borderRadius": "16px",
    "padding": "18px",
    "boxShadow": "0 10px 30px rgba(0,0,0,0.08)",
    "border": "1px solid rgba(0,0,0,0.06)",
}

app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "background": "linear-gradient(135deg, #f6f8ff, #f8fbff)",
        "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial",
        "padding": "24px",
    },
    children=[
        html.Div(
            style={**CARD_STYLE, "width": "min(920px, 96vw)"},
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "gap": "12px",
                        "marginBottom": "12px",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.H2(
                                    "English → Nepali Translator",
                                    style={"margin": "0", "fontSize": "26px"},
                                ),
                                html.Div(
                                    "Additive Attention Transformer (demo UI)",
                                    style={"opacity": "0.65", "marginTop": "4px"},
                                ),
                            ]
                        ),
                        html.Div(
                            style={
                                "padding": "8px 12px",
                                "borderRadius": "999px",
                                "background": "#f2f5ff",
                                "border": "1px solid rgba(0,0,0,0.06)",
                                "fontSize": "13px",
                            },
                            children="Tip: keep input short (1–2 sentences)",
                        ),
                    ],
                ),

                # Middle Input Box
                html.Div(
                    style={"display": "flex", "justifyContent": "center"},
                    children=[
                        html.Div(
                            style={"width": "min(720px, 92vw)"},
                            children=[
                                html.Label(
                                    "Enter English text",
                                    style={"display": "block", "marginBottom": "8px", "fontWeight": "600"},
                                ),
                                dcc.Textarea(
                                    id="src_text",
                                    value="English to Nepali Translation Model",
                                    placeholder="Type your English sentence here...",
                                    style={
                                        "width": "100%",
                                        "height": "120px",
                                        "borderRadius": "12px",
                                        "padding": "12px",
                                        "border": "1px solid rgba(0,0,0,0.18)",
                                        "outline": "none",
                                        "fontSize": "15px",
                                        "resize": "none",
                                    },
                                ),
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "gap": "10px",
                                        "alignItems": "center",
                                        "marginTop": "12px",
                                    },
                                    children=[
                                        html.Button(
                                            "Translate",
                                            id="btn_translate",
                                            n_clicks=0,
                                            style={
                                                "border": "none",
                                                "borderRadius": "12px",
                                                "padding": "10px 16px",
                                                "cursor": "pointer",
                                                "fontWeight": "700",
                                                "background": "#2b59ff",
                                                "color": "white",
                                            },
                                        ),
                                        html.Button(
                                            "Clear",
                                            id="btn_clear",
                                            n_clicks=0,
                                            style={
                                                "borderRadius": "12px",
                                                "padding": "10px 16px",
                                                "cursor": "pointer",
                                                "fontWeight": "700",
                                                "background": "white",
                                                "border": "1px solid rgba(0,0,0,0.18)",
                                            },
                                        ),
                                        html.Div(
                                            id="status",
                                            style={"opacity": "0.7", "fontSize": "13px"},
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),

                # Output Card
                html.Div(
                    style={**CARD_STYLE, "marginTop": "18px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "baseline"},
                            children=[
                                html.H3("Nepali Output", style={"margin": "0"}),
                                html.Div(
                                    "Output is capped to avoid looping",
                                    style={"opacity": "0.55", "fontSize": "12px"},
                                ),
                            ],
                        ),
                        html.Div(
                            id="out_text",
                            style={
                                "marginTop": "10px",
                                "whiteSpace": "pre-wrap",
                                "fontSize": "18px",
                                "lineHeight": "1.55",
                                "padding": "12px",
                                "borderRadius": "12px",
                                "background": "#fafafa",
                                "border": "1px solid rgba(0,0,0,0.06)",
                                "minHeight": "64px",
                            },
                        ),
                    ],
                ),

                html.Div(
                    style={"marginTop": "14px", "opacity": "0.6", "fontSize": "12px"},
                    children=" ",
                ),
            ],
        )
    ],
)


# -------------------- Callbacks --------------------
@app.callback(
    Output("src_text", "value"),
    Input("btn_clear", "n_clicks"),
    prevent_initial_call=True,
)
def clear_text(_):
    return ""


@app.callback(
    Output("out_text", "children"),
    Output("status", "children"),
    Input("btn_translate", "n_clicks"),
    State("src_text", "value"),
)
def translate(n_clicks, src_text):
    if not n_clicks:
        return "", ""

    if not src_text or not src_text.strip():
        return "⚠️ Please enter some English text first.", "Waiting for input"

    # Transform input
    src_ids = text_transform[SRC_LANG](src_text.lower())
    src_tensor = src_ids.unsqueeze(0).to(DEVICE)

    # Decode (keep steps reasonable)
    pred_ids, _ = greedy_decode(
        model,
        src_tensor,
        SOS_IDX,
        EOS_IDX,
        max_len=80,
    )

    tokens = decode_ids_to_tokens(pred_ids, eos_idx=EOS_IDX, max_out_tokens=25)
    out = " ".join(tokens).strip()

    if not out:
        out = "(No output generated — try a shorter/simple sentence.)"

    return out, f"Translated {len(tokens)} tokens"


if _name_ == "_main_":
    app.run(debug=True)