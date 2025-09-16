
import os
import json
import time
import difflib
import random
from typing import Dict, Optional

from flask import Flask, request, jsonify

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_MODEL_NAME = os.environ.get("BASE_MODEL", "gpt2-medium")
FRIDAY_MODEL_DIR = "NARA-1r-beta1"   # Your Model name
FINE_TUNED_DIR = "ft_model"        
KNOWLEDGE_FILE = "knowledge.json"
TRAIN_STATUS = "training_status.json"
MODEL_USE_LOG = "model_use.log"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_write_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def append_log(path, line):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


class ChatSystem:
    def __init__(self, base_model_name=BASE_MODEL_NAME, friday_dir=FRIDAY_MODEL_DIR):
        self.base_model_name = base_model_name
        self.friday_dir = friday_dir
        self.device = DEVICE

        self.tokenizer = None
        self.model = None
        self.model_source = None  

  
        self.knowledge: Dict[str,str] = {}
        self._load_knowledge()

 
        self._ensure_tokenizer()

    def _load_knowledge(self):
        if os.path.exists(KNOWLEDGE_FILE):
            try:
                with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.knowledge = {k.strip(): v.strip() for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
                    else:
                        self.knowledge = {}
            except Exception:
                self.knowledge = {}
        else:
            self.knowledge = {}

    def _save_knowledge(self):
        safe_write_json(KNOWLEDGE_FILE, self.knowledge)

    def _ensure_tokenizer(self):
        if self.tokenizer is not None:
            return
       
        try:
            if os.path.isdir(self.friday_dir):
                self.tokenizer = AutoTokenizer.from_pretrained(self.friday_dir, use_fast=True)
                self.model_source = "friday"
               
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                return
        except Exception:
            self.tokenizer = None
       
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self, prefer_friday: bool = True):
        """
        Load model into self.model.
        If prefer_friday and friday_dir exists, load that saved model.
        Otherwise load base model from HF hub (BASE_MODEL_NAME).
        """
        if self.model is not None:
            return self.model

        self._ensure_tokenizer()

     
        if prefer_friday and os.path.isdir(self.friday_dir):
            try:
                print("[MODEL] Loading saved model from", self.friday_dir)
                model = AutoModelForCausalLM.from_pretrained(self.friday_dir)
                model.to(self.device)
                model.eval()
                self.model = model
                self.model_source = "friday"
                return self.model
            except Exception as e:
                print("Warning: failed to load friday_model:", e)

   
        print("[MODEL] Loading base model from HF:", self.base_model_name)
        model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        model.to(self.device)
        model.eval()
        self.model = model
        self.model_source = "base"
        return self.model

    def reload_model(self):
       
        self.model = None
        self.model_source = None
        return self._load_model(prefer_friday=True)


    def generate_answer(self, query: str, max_new_tokens: int = 64, temperature: float = 0.0, top_k: int = 50, prefer_friday: bool = True):
        q = str(query or "").strip()
        if not q:
            return "Empty query"

      
        try:
            model = self._load_model(prefer_friday=prefer_friday)
            tokenizer = self.tokenizer
            prompt = f"Question: {q}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
            pad_id = tokenizer.eos_token_id

            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            if temperature and float(temperature) > 0.0:
                gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_k": int(top_k)})
            else:
                gen_kwargs.update({"do_sample": False})

            outputs = model.generate(**gen_kwargs)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)

       
            ans = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
            ans = self._clean_generated(ans)
            if ans:
                self._log_use(True, q, ans)
                return ans
        except Exception as e:
         
            print("Generation failed:", e)

    
        for k, v in self.knowledge.items():
            if k.strip().lower() == q.lower() and v.strip():
                self._log_use(False, q, v)
                return v

  
        keys = list(self.knowledge.keys())
        close = difflib.get_close_matches(q.lower(), [k.lower() for k in keys], n=1, cutoff=0.6)
        if close:
            for k in keys:
                if k.strip().lower() == close[0]:
                    v = self.knowledge[k]
                    self._log_use(False, q, v)
                    return v

        fallback = f"I don't know about '{q}'. Teach me with POST /teach or add to {KNOWLEDGE_FILE}"
        self._log_use(False, q, fallback)
        return fallback

    def _clean_generated(self, text: str):
      
        toks = text.strip().split()
        if not toks:
            return ""
    
        out = []
        prev = None
        count = 0
        for t in toks:
            if t == prev:
                count += 1
            else:
                prev = t
                count = 1
            if count <= 3:
                out.append(t)
        ans = " ".join(out).strip()
       
        if len(ans.split()) <= 1:
            return ans
        if len(set(ans.split())) <= 2 and len(ans.split()) >= 3:
            return ""
        return ans

    def _log_use(self, used_model: bool, q: str, a: str):
        s = (a[:200] + "...") if len(a) > 200 else a
        append_log(MODEL_USE_LOG, f"{time.asctime()}\tMODEL_USED={used_model}\tQ={q!r}\tA={s!r}")

 
    def teach(self, question: str, answer: str):
        q = str(question).strip()
        a = str(answer).strip()
        if not q or not a:
            return {"ok": False, "error": "question and answer must be non-empty"}
        self.knowledge[q] = a
        self._save_knowledge()
        return {"ok": True, "msg": f"Learned '{q}' -> '{a}'"}


app = Flask(__name__)
chat = ChatSystem()

@app.route("/status", methods=["GET"])
def status():
    st = {}
 
    if os.path.exists(TRAIN_STATUS):
        try:
            with open(TRAIN_STATUS, "r", encoding="utf-8") as f:
                st = json.load(f)
        except Exception:
            st = {}

    st["friday_model_exists"] = os.path.isdir(FRIDAY_MODEL_DIR)
    st["model_loaded_from"] = chat.model_source
    st["knowledge_count"] = len(chat.knowledge)
    st["device"] = str(DEVICE)
    return jsonify(st)

@app.route("/query", methods=["POST"])
def query_endpoint():
    data = request.get_json() or {}
    q = data.get("query") or data.get("q") or ""
    if not q:
        return jsonify({"ok": False, "error": "No query provided"}), 400
    temp = float(data.get("temperature", 0.0))
    top_k = int(data.get("top_k", 50))
    prefer_friday = bool(data.get("prefer_friday", True))
    max_new = int(data.get("max_new_tokens",  100))
    try:
        ans = chat.generate_answer(q, max_new_tokens=max_new, temperature=temp, top_k=top_k, prefer_friday=prefer_friday)
        return jsonify({"ok": True, "query": q, "answer": ans})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/teach", methods=["POST"])
def teach_endpoint():
    payload = request.get_json() or {}
    q = payload.get("question") or payload.get("q")
    a = payload.get("answer") or payload.get("a")
    if not q or not a:
        return jsonify({"ok": False, "error": "Provide question and answer"}), 400
    res = chat.teach(q, a)
    return jsonify(res)

@app.route("/reload_model", methods=["POST"])
def reload_model():
    try:
        chat.tokenizer = None
        chat.model = None
        chat.model_source = None
        m = chat._load_model(prefer_friday=True)
        ok = m is not None
        return jsonify({"ok": ok, "loaded_from": chat.model_source})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"ok": True, "device": str(DEVICE)})

if __name__ == "__main__":
    print("Starting server. Device:", DEVICE)
    print("FRIDAY_MODEL_DIR:", FRIDAY_MODEL_DIR, "| Base model:", BASE_MODEL_NAME)
    print("Endpoints: POST /query  POST /teach  POST /reload_model  GET /status  GET /ping")
    app.run(host="0.0.0.0", port=5000)
