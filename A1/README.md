Step 2: Navigate to Assignment1
cd Assignment1

Step 3: Run Flask App
python app.py

Step 4: Open in Browser
http://127.0.0.1:5000

📊 Experimental Results
Model Comparison
Model	Window Size	Training Loss	Training Time (sec)	Syntactic Accuracy	Semantic Accuracy
Skip-gram	2	8.0564	433.89	3.33%	0.00%
Skip-gram (NEG)	2	14.3561	474.58	0.00%	0.00%
GloVe (Scratch)	2	9.5779	98.30	0.00%	0.00%
GloVe (Gensim)	–	–	–	55.45%	93.87%
Spearman Correlation (Word Similarity)
Model	Spearman Correlation
Skip-gram	0.1058
Skip-gram (NEG)	0.0241
GloVe (Scratch)	0.0510
GloVe (Gensim)	0.53
🔍 Observations

All scratch models performed poorly on syntactic and semantic tasks.

This is expected due to:

Small corpus size

Low embedding dimension

Limited context window

GloVe (Gensim) significantly outperformed scratch models.

Spearman correlation values for scratch models were close to zero, indicating poor alignment with human judgment.

🧠 Why Scratch Models Perform Poorly

Word embedding models require large corpora to learn meaningful relationships.

Without sufficient data, embeddings become noisy and unreliable.

Pretrained models benefit from:

Billions of tokens

Better hyperparameter tuning

Optimized training pipelines

📝 Conclusion

This assignment demonstrates:

Core understanding of word embedding algorithms

Practical challenges in training embeddings from scratch

Evaluation using standard NLP benchmarks

Application of embeddings in a real-world search engine

Clear performance gap between scratch and pretrained models

🚀 Future Improvements

Increase corpus size

Increase embedding dimension

Tune learning rate and epochs

Use FastText for subword embeddings

Extend search to sentence-level similarity

Add Assignment 2 in a separate folder

👤 Author

Name: Your Name
Course: Natural Language Processing
Semester: Second Semester
University: Your University Name

📌 Notes for GitHub

Git initialized at Assignment/ level

Assignment1 is pushed as a subfolder

Assignment2 will be added later without affecting Assignment1


Note: `model_gensim.pkl` is excluded from the repository due to GitHub file size limits.