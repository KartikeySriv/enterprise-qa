from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Lazy loading of retrieval module to avoid startup issues
def get_retrieval_functions():
    """Lazy load retrieval functions to avoid import issues at startup."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from retrieval.search import (
        load_embeddings,
        load_json,
        load_sections,
        load_embedding_model,
        search as semantic_search,
    )
    return {
        'load_embeddings': load_embeddings,
        'load_json': load_json,
        'load_sections': load_sections,
        'load_embedding_model': load_embedding_model,
        'semantic_search': semantic_search,
    }

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data file paths
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.json"
QA_EVALUATION_PATH = PROJECT_ROOT / "evaluation" / "qa_evaluation.json"
RETRIEVAL_EVALUATION_PATH = PROJECT_ROOT / "evaluation" / "retrieval_evaluation.json"
TRAINER_STATE_PATH = PROJECT_ROOT / "models" / "tinyroberta-aurelia-qa" / "trainer_state.json"

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Enterprise QA Research Dashboard API",
    description="API for serving QA fine-tuning and semantic retrieval experiment results",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# PYDANTIC MODELS
# ============================================================

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON file with error handling."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_text_file(path: Path) -> str:
    """Load a text file with error handling."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return f.read()

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/api/stats")
async def get_stats():
    """Return overall experiment statistics from data files."""
    try:
        # Load corpus data
        corpus_data = load_json_file(CORPUS_PATH)
        
        # Load QA evaluation data
        qa_eval = load_json_file(QA_EVALUATION_PATH)
        
        # Load retrieval evaluation data
        retrieval_eval = load_json_file(RETRIEVAL_EVALUATION_PATH)
        
        # Load training state for model info
        trainer_state = load_json_file(TRAINER_STATE_PATH)
        
        # Extract QA model parameters from training state
        total_params = 81529346  # From the train.py output in README
        
        # Calculate improvements
        baseline_em = qa_eval["baseline"]["exact_match"]
        fine_tuned_em = qa_eval["fine_tuned"]["exact_match"]
        baseline_f1 = qa_eval["baseline"]["token_f1"]
        fine_tuned_f1 = qa_eval["fine_tuned"]["token_f1"]
        
        em_improvement = (fine_tuned_em - baseline_em) * 100
        f1_improvement = (fine_tuned_f1 - baseline_f1) * 100
        
        return {
            "data": {
                "source_documents": corpus_data["document_count"],
                "structured_records": corpus_data["record_count"],
                "qa_examples": 150,  # From dataset_report.json
                "train": 105,
                "validation": 23,
                "test": 22
            },
            "qa_model": {
                "base_model": "deepset/tinyroberta-squad2",
                "fine_tuned_model": "tinyroberta-aurelia-qa",
                "total_parameters": total_params,
                "trainable_parameters": total_params,
                "epochs": trainer_state["num_train_epochs"],
                "learning_rate": "2e-5",
                "batch_size": trainer_state["train_batch_size"],
                "weight_decay": 0.01,
                "device": "cpu"
            },
            "qa_results": {
                "baseline": {
                    "exact_match": baseline_em,
                    "token_f1": baseline_f1
                },
                "fine_tuned": {
                    "exact_match": fine_tuned_em,
                    "token_f1": fine_tuned_f1
                },
                "improvement": {
                    "em_pp": round(em_improvement, 2),
                    "f1_pp": round(f1_improvement, 2)
                }
            },
            "retrieval": {
                "embedding_model": retrieval_eval["model"],
                "dimension": retrieval_eval["embedding_dimension"],
                "records_embedded": retrieval_eval["document_records"],
                "recall_at_1": retrieval_eval["recall_at_1"],
                "recall_at_3": retrieval_eval["recall_at_3"],
                "recall_at_5": retrieval_eval["recall_at_5"]
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/qa/comparison")
async def get_qa_comparison():
    """Return QA evaluation data with baseline vs fine-tuned predictions."""
    try:
        qa_eval = load_json_file(QA_EVALUATION_PATH)
        return qa_eval
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/retrieval/metrics")
async def get_retrieval_metrics():
    """Return retrieval evaluation metrics."""
    try:
        retrieval_eval = load_json_file(RETRIEVAL_EVALUATION_PATH)
        return retrieval_eval
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/training/history")
async def get_training_history():
    """Return training loss by epoch from trainer state."""
    try:
        trainer_state = load_json_file(TRAINER_STATE_PATH)
        
        # Extract training and validation loss from log_history
        log_history = trainer_state.get("log_history", [])
        
        epochs = []
        for entry in log_history:
            if "epoch" in entry and "loss" in entry:
                epoch_data = {
                    "epoch": int(entry["epoch"]),
                    "train_loss": entry["loss"]
                }
                # Find corresponding eval loss
                eval_entry = next(
                    (e for e in log_history 
                     if e.get("epoch") == entry["epoch"] and "eval_loss" in e),
                    None
                )
                if eval_entry:
                    epoch_data["eval_loss"] = eval_entry["eval_loss"]
                epochs.append(epoch_data)
        
        return {"epochs": epochs}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/retrieval/search")
async def search_retrieval(request: SearchRequest):
    """Semantic search using BGE embeddings, reusing retrieval/search.py functions."""
    try:
        # Lazy load retrieval functions
        retrieval_funcs = get_retrieval_functions()
        
        # Load required data using existing functions
        embeddings_path = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
        metadata_path = PROJECT_ROOT / "data" / "processed" / "embedding_metadata.json"
        sections_path = PROJECT_ROOT / "data" / "processed" / "sections.jsonl"
        
        embeddings = retrieval_funcs['load_embeddings'](embeddings_path)
        metadata = retrieval_funcs['load_json'](metadata_path)
        sections = retrieval_funcs['load_sections'](sections_path)
        model = retrieval_funcs['load_embedding_model']()
        
        # Perform search using existing function
        results = retrieval_funcs['semantic_search'](
            query=request.query,
            model=model,
            document_embeddings=embeddings,
            metadata=metadata,
            sections=sections,
            top_k=request.top_k
        )
        
        # Filter results by similarity >= 0.5
        filtered_results = [
            result for result in results 
            if result["score"] >= 0.5
        ]
        
        # Format response
        formatted_results = []
        for result in filtered_results:
            formatted_results.append({
                "rank": result["rank"],
                "similarity": result["score"],
                "record_id": result["record_id"],
                "document": result["document_title"],
                "section": result["section"],
                "subsection": result["subsection"],
                "source": result["source_file"],
                "text": result["text"]
            })
        
        return {
            "query": request.query,
            "results": formatted_results,
            "total_candidates": len(results),
            "filtered_results": len(formatted_results)
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)