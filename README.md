# Adaptive RAG for Failure-Aware Retrieval and Conflict Resolution

## 🚀 Overview
This project implements an **adaptive Retrieval-Augmented Generation (RAG) system** that dynamically adjusts retrieval and reasoning strategies based on query failure types such as:
- Bridge (multi-hop)
- Misinformation
- Noise

Instead of using a fixed pipeline, the system selects different strategies to improve robustness in multi-source question answering.

---

## 🧠 Key Features
- Failure-aware query classification  
- Hybrid + multi-hop retrieval  
- Trust-weighted aggregation  
- Conflict-aware answer generation  
- Evaluation pipeline with EM/F1 metrics  

---

## 📊 Results
- **42% Exact Match (EM)**  
- **52% F1 Score**  
- **88% misinformation suppression**  
- Strong performance on noisy/conflicting datasets  

---

## ⚠️ Known Limitations
- Multi-hop retrieval remains a bottleneck (0% recall on HotpotQA)  
- Evaluated on a small dataset (100 examples)  

---

## 🛠️ Setup

### 1. Create virtual environment
python -m venv rag_env

### 2. Activate environment
rag_env\Scripts\activate

Mac/Linux:
source rag_env/bin/activate

### 3. install dependencies
pip install -r requirements.txt

### 4. Set API key
set OPENAI_API_KEY=your_api_key

or 
export OPENAI_API_KEY=your_api_key

### 5. Run pipeline
python main.py


## 📥 Data Setup

The project uses HotpotQA and RAMDocs datasets.

To download and prepare the data automatically, run:
python data/data_load.py