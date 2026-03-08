from dash import Dash, html, dcc, Input, Output, State
import torch
import os
from transformers import BertTokenizer
from Bert import BERT, calculate_similarity

# ======================
# Device (keep same)
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Load model (keep same)
# ======================
model_path = '../model/sen_bert.pth'
params, state = torch.load(model_path)
model_bert = BERT(**params, device=device).to(device)
model_bert.load_state_dict(state)
model_bert.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

app = Dash(__name__)

# ======================
# BEAUTIFUL LAYOUT
# ======================
app.layout = html.Div(

    style={
        "minHeight": "100vh",
        "background": "linear-gradient(135deg, #eef2f3, #ffffff)",
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "padding": "40px"
    },

    children=[

        # ---------- TITLE ----------
        html.H1(
            "🤝 A4: Do You Agree?",
            style={
                "textAlign": "center",
                "fontWeight": "600",
                "marginBottom": "10px"
            }
        ),

        html.P(
            "Sentence Similarity using Custom BERT",
            style={
                "textAlign": "center",
                "color": "#555",
                "marginBottom": "40px"
            }
        ),

        # ---------- CARD ----------
        html.Div(
            style={
                "maxWidth": "520px",
                "margin": "0 auto",
                "backgroundColor": "white",
                "padding": "30px",
                "borderRadius": "12px",
                "boxShadow": "0 10px 25px rgba(0,0,0,0.08)"
            },

            children=[

                dcc.Input(
                    id="query-one",
                    placeholder="Enter first sentence",
                    type="text",
                    style={
                        "width": "100%",
                        "padding": "12px",
                        "borderRadius": "8px",
                        "border": "1px solid #ccc",
                        "marginBottom": "15px",
                        "fontSize": "15px"
                    }
                ),

                dcc.Input(
                    id="query-two",
                    placeholder="Enter second sentence",
                    type="text",
                    style={
                        "width": "100%",
                        "padding": "12px",
                        "borderRadius": "8px",
                        "border": "1px solid #ccc",
                        "marginBottom": "25px",
                        "fontSize": "15px"
                    }
                ),

                html.Button(
                    "Generate Similarity",
                    id="search-button",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "padding": "12px",
                        "borderRadius": "8px",
                        "border": "none",
                        "backgroundColor": "#0d6efd",
                        "color": "white",
                        "fontSize": "16px",
                        "fontWeight": "500",
                        "cursor": "pointer"
                    }
                )
            ]
        ),

        # ---------- RESULT ----------
        html.Div(
            id="search-results",
            style={
                "marginTop": "30px",
                "display": "flex",
                "justifyContent": "center"
            }
        )
    ]
)

# ======================
# CALLBACK (UNCHANGED LOGIC)
# ======================
@app.callback(
    Output("search-results", "children"),
    Input("search-button", "n_clicks"),
    State("query-one", "value"),
    State("query-two", "value")
)
def search(n_clicks, query_one, query_two):

    if n_clicks == 0:
        return html.Div(
            "Enter two sentences and click Generate",
            style={"color": "#777"}
        )

    if not query_one or not query_two:
        return html.Div(
            "Please fill both input fields.",
            style={"color": "red"}
        )

    score = calculate_similarity(
        model_bert,
        tokenizer,
        params["max_len"],
        query_one,
        query_two,
        device
    )

    if score >= 0.75:
        label = "Entailment"
        color = "#198754"
    elif score < 0.4:
        label = "Contradiction"
        color = "#dc3545"
    else:
        label = "Neutral"
        color = "#fd7e14"

    return html.Div(
        style={
            "maxWidth": "520px",
            "backgroundColor": "white",
            "padding": "20px",
            "borderRadius": "10px",
            "boxShadow": "0 8px 20px rgba(0,0,0,0.08)",
            "textAlign": "center"
        },
        children=[
            html.H4(label, style={"color": color}),
            html.P(f"Similarity Score: {score:.3f}")
        ]
    )

# ======================
# RUN
# ======================
if __name__ == "__main__":
    app.run(debug=True)
