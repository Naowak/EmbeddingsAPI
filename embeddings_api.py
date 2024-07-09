from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch

app = Flask(__name__)

# Load the model and tokenizer
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
    return embeddings

@app.route('/embed', methods=['POST'])
def embed_texts():
    data = request.json
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    try:
        embeddings = get_embeddings(texts)
        return jsonify(embeddings), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
        app.run(debug=True)

        
