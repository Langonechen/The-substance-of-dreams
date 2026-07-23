# The-substance-of-dreams
This repository contains a complete, end-to-end Machine Learning pipeline that analyzes, retrieves, and continues dream reports based on emotional context. It utilizes a custom Retrieval-Augmented Generation (RAG) architecture, bridging a state-of-the-art dense retrieval system with a custom-built, PyTorch-based Transformer language model.

🌟 Key FeaturesCustom Transformer Architecture: 
- Custom Transformer Architecture: Implements a GPT-style Bigram Language Model from scratch in PyTorch, complete with Multi-Head Self-Attention, FeedForward layers, and Layer Normalization.
- Dense Document Retrieval: Uses BAAI/bge-small-en-v1.5 embeddings and FAISS (Facebook AI Similarity Search) to retrieve semantically similar dream context efficiently.
- Semantic Chunking: Leverages langchain-text-splitters for intelligent text chunking based on characters and natural breakpoints.
- RAG-Powered Generation: Constructs dynamic prompts using retrieved context to guide the custom LLM in generating contextually relevant dream continuations.
- Batch Evaluation: Automatically evaluates incomplete dream reports and outputs a generated continuation into a structured CSV file.

📂 Data Requirements:
To run this pipeline, you will need the following datasets in your root directory: 
- dreams_emotions1.csv --> The primary training corpus containing full dream texts and dominant macro-emotion labels.
- Truncated_dreams_reports.csv -- > The testing dataset containing incomplete dream reports and query IDs for evaluation.

Note: The script will automatically extract the dream text from the primary dataset and generate a local dream_text.txt file for corpus processing.  

🛠️ Installation & Setup
Ensure you have Python 3.8+ installed. You can install the required dependencies using pip:
```
pip install torch pandas numpy matplotlib scikit-learn faiss-cpu sentence-transformers transformers langchain-text-splitters tqdm
```
Hardware Recommendation:
While the script can run on a CPU, a CUDA-enabled GPU (e.g., NVIDIA Tesla T4) is highly recommended for faster training and embedding generation. The script automatically detects and utilizes CUDA if available.  

🚀 Usage
1. Prepare your data
Ensure that dreams_emotions1.csv and truncated_dreams_reports.csv are placed in the same directory as the main Python script. The primary dataset must contain a column named dream_text.
2. Run the pipelineExecute the main script to trigger the embedding, training, and generation phases.

```
python main.py
```
3. Pipeline Execution PhasesWhen you run the script, it will execute in three distinct phases:
   1. Retrieval Indexing: Embeds the dream corpus and builds the FAISS index.
   2. Model Training: Trains the custom Transformer model for a specified number of iterations (default is 5000),         logging training and validation loss every 500 steps.
   3. Inference & Generation: Iterates through the truncated test set, retrieves similar dreams via RAG, and              generates 50-token continuations for each incomplete dream.

4. Upon completion, the pipeline will generate a predictions.csv file containing the original Emotion Label (id) and the model's generated continuation (answer).  



