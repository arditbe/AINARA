
"""
Advanced AINARA AI Chatbot

Commands:
  - teach -> <question> -> <answer>
      Teaches a new Q&A pair (stored in knowledge.json). You may also use:
        teach <question> -> <answer>
  - personality -> <description>
      Set personality description.
  - csv -> <filepath>
      Load Q&A pairs from a CSV file.
  - learn -> <filepath>
      Learn from a source file (HTML, CSS, JS, TS, Python, Java, etc.).
  - train -> <filepath>
      Load Q&A pairs from a text file (if no "->" is found, the file is treated as a document).
  - modeltrain
      Train the deep-learning model using all learned data (from memory, documents, and conversation).
  - analyze image -> <filepath>
      Analyze an image file using MobileNetV2.
  - show sources
      List the names of all loaded source files.
  - (Any other text is treated as a query.)

Special:
  If you ask for code generation (e.g., "generate me hello world code on python"),
  Friday will try to generate code using a dedicated branch.
  
Note:
  This framework assumes you have a pre-trained seq2seq model saved as "friday_model.h5"
  and a tokenizer saved as "tokenizer.pickle". The image analysis module uses MobileNetV2.
  
Extend and modify as needed.
"""

import os
import json
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import random
import cv2
import ast
import re
import pytesseract
import difflib 

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess, decode_predictions as mobilenet_decode
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


current_topic_key = None  
memory = {}              
conversation_context = [] 



def load_memory(filename="knowledge.json"):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
            mem = {entry["question"].strip().lower(): entry["answer"].strip() 
                   for entry in data.get("learned", [])}
            mem['__personality__'] = data.get('__personality__', "")
            mem["__documents__"] = data.get("__documents__", [])
            mem["__quotes__"] = data.get("__quotes__", [])
            if "__code_files__" in data:
                mem["__code_files__"] = data["__code_files__"]
            return mem
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_memory(memory, filename="knowledge.json"):
    learned = [{"question": q, "answer": a} for q, a in memory.items() 
               if q not in ['__personality__', '__code_files__', '__documents__', '__quotes__']]
    data = {
        "learned": learned,
        "__personality__": memory.get('__personality__', ""),
        "__documents__": memory.get("__documents__", []),
        "__quotes__": memory.get("__quotes__", [])
    }
    if "__code_files__" in memory:
        data["__code_files__"] = memory["__code_files__"]
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

memory = load_memory()


def load_tokenizer(tokenizer_path="tokenizer.pickle"):
    global tokenizer
    try:
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
    except (FileNotFoundError, pickle.PickleError):
        tokenizer = Tokenizer()

def load_ai_model(model_path="friday_model.h5"):
    global model
    try:
        model = load_model(model_path)
    except (FileNotFoundError, IOError):
        model = None

load_tokenizer()
load_ai_model()


def decode_sequence(sequence):
    if isinstance(sequence, (np.int64, int)):
        sequence = [sequence]
    words = []
    for idx in sequence:
        if idx == 0:
            continue
        word = tokenizer.index_word.get(idx, '')
        if word == '<end>':
            break
        words.append(word)
    return ' '.join(words)

def generate_response_deep(input_text, max_len=None):
    try:
        if max_len is None and model is not None:
            max_len = model.input_shape[1]
        elif max_len is None:
            max_len = 20
        sequence = tokenizer.texts_to_sequences([input_text])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post')
        predictions = model.predict(padded)
        predicted_indices = np.argmax(predictions, axis=-1)
        if predicted_indices.ndim == 2:
            predicted_indices = predicted_indices[0]
        response_text = decode_sequence(predicted_indices)
        return response_text.strip()
    except Exception as e:
        print("Error in generating response:", e)
        return "I'm still learning. I will get better soon."


def generate_code(prompt, max_len=100):
    return generate_response_deep(prompt, max_len=max_len)

def handle_code_generation(query):
    lower_query = query.lower()
    if "python" in lower_query:
        code_files = memory.get("__code_files__", {})
        for fname, info in code_files.items():
            if info.get("language", "").lower() == "python":
                prompt = info.get("content", "")
                full_prompt = prompt + "\n" + query
                return generate_code(full_prompt, max_len=150)
        return generate_code(query, max_len=150)
    elif "html" in lower_query:
        code_files = memory.get("__code_files__", {})
        for fname, info in code_files.items():
            if info.get("language", "").lower() == "html":
                prompt = info.get("content", "")
                full_prompt = prompt + "\n" + query
                return generate_code(full_prompt, max_len=150)
        return generate_code(query, max_len=150)
    return generate_code(query, max_len=150)


def handle_function_generation(query):
    """
    Searches for the requested language and (optionally) a specific file.
    It then gathers candidate function names from the learned source files that match
    the language and constructs a prompt for code generation.
    """
    query_lower = query.lower()
    languages = ["python", "javascript", "java", "c++", "c", "ruby", "php", "html", "css", "typescript"]
    lang = None
    for l in languages:
        if l in query_lower:
            lang = l
            break
    if lang is None:
        lang = "python" 
    
    file_match = re.search(r'from\s+(\S+\.\w+)', query_lower)
    target_file = file_match.group(1) if file_match else None

    candidate_funcs = []
    if target_file:
        code_file = memory.get("__code_files__", {}).get(target_file)
        if code_file and code_file.get("analysis") and code_file["analysis"].get("functions"):
            candidate_funcs = list(code_file["analysis"]["functions"].keys())
    else:
        for fname, info in memory.get("__code_files__", {}).items():
            if info.get("language", "").lower() == lang and info.get("analysis") and info["analysis"].get("functions"):
                candidate_funcs.extend(list(info["analysis"]["functions"].keys()))
    if not candidate_funcs:
        prompt = f"Generate a new {lang} function that is useful and complete."
        generated_function = generate_code(prompt, max_len=150)
        return generated_function
    func_name = random.choice(candidate_funcs)
    docstring = ""
    if target_file and target_file in memory.get("__code_files__", {}):
        docstring = memory["__code_files__"][target_file]["analysis"]["functions"].get(func_name, "")
    prompt = f"Generate a new {lang} function named {func_name}()."
    if docstring:
        prompt += f" The function should do the following: {docstring}"
    else:
        prompt += " The function should be useful and complete."
    generated_function = generate_code(prompt, max_len=150)
    return generated_function


def generate_quote(query):
    """
    Checks the loaded quotes (from memory["__quotes__"]) for ones matching the requested category.
    Categories are determined by keywords in the query.
    Returns a random matching quote if available.
    """
    quotes = memory.get("__quotes__", [])
    if not quotes:
        return "I don't have any quotes yet, but I can cheer you up!"
    
    query_lower = query.lower()
    candidates = []
    
    category = None
    if "motivational" in query_lower or "motivate" in query_lower:
        category = "motivational"
    elif "sad" in query_lower:
        category = "sad"
    elif "happy" in query_lower:
        category = "happy"
    elif "inspire" in query_lower:
        category = "motivational"
    
    if category:
        for q in quotes:
            if q.get("category", "").lower() == category:
                candidates.append(q)
        if not candidates:
            for q in quotes:
                if "tags" in q and any(category in tag.lower() for tag in q["tags"]):
                    candidates.append(q)
    else:
        candidates = quotes

    if candidates:
        chosen = random.choice(candidates)
        return chosen.get("text", "Here's a quote!")
    else:
        return "I don't have a specific quote for that, but I can cheer you up: Keep your head up!"


image_model = MobileNetV2(weights="imagenet")

def analyze_image(image_path):
    if not os.path.exists(image_path):
        return "Image file not found."
    img = cv2.imread(image_path)
    if img is None:
        return "Failed to load image."
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(img_rgb, axis=0)
    x = mobilenet_preprocess(x)
    preds = image_model.predict(x)
    predictions = mobilenet_decode(preds, top=3)[0]
    objects_result = "\n".join([f"{label}: {round(prob * 100, 2)}%" for (_, label, prob) in predictions])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    custom_config = r'--oem 3 --psm 6'
    text_result = pytesseract.image_to_string(gray, config=custom_config).strip()
    final_result = "Objects detected:\n" + objects_result
    if text_result:
        final_result += "\n\nText detected:\n" + text_result
    else:
        final_result += "\n\nNo readable text detected."
    return final_result


def find_best_match(input_text, memory):
    questions = [q for q in memory.keys() if q not in ['__personality__', '__code_files__', '__documents__', '__quotes__']]
    if "not implemented" in input_text.lower():
        for q in questions:
            if "not implemented" in memory[q].lower():
                return q
    for q in questions:
        if q in input_text.lower():
            return q
    matches = difflib.get_close_matches(input_text.lower(), questions, n=1, cutoff=0.2)
    if matches:
        return matches[0]
    if not questions:
        return None
    input_text_lower = input_text.lower()
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    vectors = vectorizer.fit_transform(questions + [input_text_lower])
    similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    if not similarities.any():
        return None
    best_match_index = similarities.argmax()
    if similarities[best_match_index] < 0.25:
        return None
    return questions[best_match_index]


def find_document_completion(query):
    import re
    documents = memory.get("__documents__", [])
    best_candidate = None
    best_score = 0
    query_lower = query.lower()
    for doc in documents:
        sentences = re.split(r'(?<=[.!?])\s+', doc)
        for sent in sentences:
            sent_lower = sent.lower()
            if sent_lower.startswith(query_lower):
                score = 1.0
            elif query_lower in sent_lower:
                score = difflib.SequenceMatcher(None, query_lower, sent_lower).ratio()
            else:
                score = 0
            if score > best_score:
                best_score = score
                best_candidate = sent
    return best_candidate if best_candidate is not None and best_score >= 0.2 else None


def analyze_python_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            source_code = file.read()
        tree = ast.parse(source_code)
        summary = {"functions": {}, "classes": {}, "variables": []}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node) or "No description provided."
                summary["functions"][node.name] = docstring
            elif isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node) or "No description provided."
                summary["classes"][node.name] = docstring
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        summary["variables"].append(target.id)
        return summary
    except Exception as e:
        return {"error": str(e)}

def get_python_info(filename, request):
    code_files = memory.get("__code_files__", {})
    if filename not in code_files:
        return f"I haven't learned {filename} yet."
    file_data = code_files[filename]
    if file_data.get("language", "").lower() != "python" or not file_data.get("analysis"):
        return f"No detailed analysis available for {filename}."
    analysis = file_data["analysis"]
    if request == "summary":
        functions = list(analysis["functions"].keys())
        classes = list(analysis["classes"].keys())
        variables = analysis["variables"]
        return (f"Summary of {filename}:\n"
                f"Functions: {functions}\n"
                f"Classes: {classes}\n"
                f"Variables: {variables}")
    if request in analysis["functions"]:
        return f"Function '{request}': {analysis['functions'][request]}"
    if request in analysis["classes"]:
        return f"Class '{request}': {analysis['classes'][request]}"
    return f"Could not find '{request}' in {filename}."


def teach_ai(question, answer):
    question_lower = question.strip().lower()
    memory[question_lower] = answer.strip()
    save_memory(memory)
    print("Knowledge updated (teach).")

def teach_personality(personality_str):
    memory['__personality__'] = personality_str.strip()
    save_memory(memory)
    print("Personality updated.")

def learn_from_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        new_data = {str(row[0]).strip().lower(): str(row[1]).strip() 
                    for _, row in df.iterrows()}
        memory.update(new_data)
        save_memory(memory)
        print(f"Loaded CSV file: {filepath}")
    except Exception as e:
        print(f"Failed to load CSV file: {e}")

def learn_from_file(filepath):
    ext = filepath.split('.')[-1].lower()
    new_data = {}
    try:
        if ext == "csv":
            learn_from_csv(filepath)
            return
        elif ext == "json":
            with open(filepath, "r") as file:
                data = json.load(file)
                new_data = {entry["question"].strip().lower(): entry["answer"].strip() 
                            for entry in data.get("learned", [])}
        else:
            with open(filepath, "r", encoding='utf-8') as file:
                lines = file.readlines()
            if any("->" in line for line in lines):
                for line in lines:
                    if "->" in line:
                        q, a = map(str.strip, line.split("->", 1))
                        new_data[q.lower()] = a
            else:
                content = "".join(lines)
                if "__documents__" not in memory:
                    memory["__documents__"] = []
                memory["__documents__"].append(content)
                print(f"Loaded document from file: {filepath}")
                save_memory(memory)
                return
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return
    memory.update(new_data)
    save_memory(memory)
    print(f"Loaded file: {filepath}")

def learn_from_source_file(filepath):
    ext = filepath.split('.')[-1].lower()
    supported_exts = {
        "html": "HTML", "css": "CSS", "js": "JavaScript", "ts": "TypeScript",
        "py": "Python", "java": "Java", "txt": "Text", "cpp": "C++", "c": "C"
    }
    if ext not in supported_exts:
        print(f"File extension '{ext}' is not recognized for source learning.")
        return
    try:
        with open(filepath, "r", encoding='utf-8') as file:
            content = file.read()
        analysis = None
        if ext == "py":
            analysis = analyze_python_file(filepath)
        if "__code_files__" not in memory:
            memory["__code_files__"] = {}
        memory["__code_files__"][os.path.basename(filepath)] = {
            "language": supported_exts[ext],
            "content": content,
            "analysis": analysis
        }
        save_memory(memory)
        print(f"Learned source file: {filepath}")
    except Exception as e:
        print(f"Failed to read source file {filepath}: {e}")

# ---------------------------
# Aggregated Training Data and Model Training
# ---------------------------
def aggregate_training_data():
    samples = []
    for q, a in memory.items():
        if q not in ['__personality__', '__code_files__', '__documents__', '__quotes__']:
            samples.append("Q: " + q + " A: " + a)
    if memory.get('__personality__'):
        samples.append("Personality: " + memory['__personality__'])
    if "__code_files__" in memory:
        for fname, info in memory["__code_files__"].items():
            samples.append(f"File: {fname} ({info.get('language','')}) Content: " + info.get("content", ""))
    if "__documents__" in memory:
        for doc in memory["__documents__"]:
            samples.append("Document: " + doc)

    if "__quotes__" in memory:
        for quote in memory["__quotes__"]:
            samples.append("Quote: " + quote.get("text", ""))
    for q, a in conversation_context:
        samples.append("Conversation: Q: " + q + " A: " + a)
    return samples

def model_train_from_memory():
    training_samples = aggregate_training_data()
    if not training_samples:
        print("No training data available from memory.")
        return
    global tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(training_samples)
    sequences = tokenizer.texts_to_sequences(training_samples)
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 50
    inputs = Input(shape=(max_len,))
    x = Embedding(vocab_size, embedding_dim, input_length=max_len)(inputs)
    x = LSTM(128, return_sequences=True)(x)
    outputs = tf.keras.layers.TimeDistributed(Dense(vocab_size, activation='softmax'))(x)
    new_model = tf.keras.Model(inputs, outputs)
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    X_train = padded_sequences
    y_train = np.expand_dims(padded_sequences, -1)
    print(f"Starting advanced model training on {len(training_samples)} aggregated samples...")
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    checkpoint_cb = ModelCheckpoint("friday_model.h5", save_best_only=True, verbose=1)
    earlystop_cb = EarlyStopping(patience=3, verbose=1, restore_best_weights=True)
    reduce_lr_cb = ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)
    new_model.fit(X_train, y_train, epochs=10, batch_size=32,
                  callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb])
    with open("tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Advanced model training completed and best model saved as 'friday_model.h5'. Tokenizer updated in 'tokenizer.pickle'.")
    global model
    model = new_model


def handle_source_query(input_text):
    lower_input = input_text.lower()
    for fname, info in memory.get("__code_files__", {}).items():
        if fname.lower() in lower_input:
            if ("summarize" in lower_input or "what do you know" in lower_input or 
                ("function" in lower_input and "in" in lower_input)):
                return get_python_info(fname, "summary")
            match = re.search(r'show me (?:the code for|code of)\s+"?([a-zA-Z0-9_]+)"?', lower_input)
            if match:
                req = match.group(1)
                return get_python_info(fname, req)
    return None


def update_context(question, answer):
    if len(conversation_context) > 5:
        conversation_context.pop(0)
    conversation_context.append((question, answer))

def infer_new_knowledge(question):
    for q, a in memory.items():
        words = question.lower().split()
        if "capital" in q and any(word in q for word in words):
            return f"{a} is the capital of {q.split()[-1]}."
        if "dangerous" in question.lower() and "chernobyl" in q:
            return a
    return None

def chain_reasoning(question):
    for q, a in memory.items():
        if q in ['__personality__', '__code_files__', '__documents__', '__quotes__']:
            continue
        for word in question.lower().split():
            if word in q.lower():
                for q2, a2 in memory.items():
                    if q2 in ['__personality__', '__code_files__', '__documents__', '__quotes__']:
                        continue
                    if a and (a in q2) and a2 and (a2 != a):
                        return f"{a} and {a2} are related."
    return None

def is_followup_question(input_text):
    followup_keywords = ["more", "still", "further", "additional", "also", "nice", "provide", "tell me more", "what about"]
    lower_input = input_text.lower()
    if any(word in lower_input for word in followup_keywords):
        return True
    if len(lower_input.split()) < 4:
        return True
    return False


def get_response(input_text):
    global current_topic_key

 
    if "quote" in input_text.lower() or "motivate" in input_text.lower():
        quote_response = generate_quote(input_text)
        update_context(input_text, quote_response)
        return quote_response

  
    if "generate me" in input_text.lower() and "function" in input_text.lower():
        response = handle_function_generation(input_text)
        update_context(input_text, response)
        return response


    if "generate me" in input_text.lower() and "code" in input_text.lower():
        code_response = handle_code_generation(input_text)
        if code_response:
            update_context(input_text, code_response)
            return code_response

 
    source_response = handle_source_query(input_text)
    if source_response is not None:
        return source_response

    inferred = infer_new_knowledge(input_text)
    if inferred:
        update_context(input_text, inferred)
        return inferred

    best_match = find_best_match(input_text, memory)
    if best_match:
        response = memory[best_match]
        if len(response.split()) > 15:
            current_topic_key = best_match
        update_context(input_text, response)
        return response

    doc_completion = find_document_completion(input_text)
    if doc_completion:
        update_context(input_text, doc_completion)
        return doc_completion

    if "generate me" in input_text.lower() and "code" not in input_text.lower():
        stripped = input_text.lower().replace("generate me", "").strip()
        doc_completion = find_document_completion(stripped)
        if doc_completion:
            update_context(input_text, doc_completion)
            return doc_completion

    reasoned = chain_reasoning(input_text)
    if reasoned:
        update_context(input_text, reasoned)
        return reasoned

    if is_followup_question(input_text) and current_topic_key is not None:
        combined_query = memory[current_topic_key] + " " + input_text
        if model is not None:
            gen_response = generate_response_deep(combined_query)
            if gen_response:
                update_context(input_text, gen_response)
                return gen_response

    if model is not None:
        gen_response = generate_response_deep(input_text)
        if gen_response:
            update_context(input_text, gen_response)
            return gen_response

    fallback = "I don't know yet, but I want to learn!"
    update_context(input_text, fallback)
    return fallback


def generate_quote(query):
    """
    Searches the loaded quotes (from memory["__quotes__"]) for a quote matching the requested category.
    If a category such as 'motivational', 'sad', or 'happy' is mentioned, it selects a matching quote.
    Otherwise, it picks a random quote or provides a fallback.
    """
    quotes = memory.get("__quotes__", [])
    if not quotes:
        return "I don't have any quotes yet, but I can cheer you up!"
    
    query_lower = query.lower()
    candidates = []
    category = None
    if "motivational" in query_lower or "motivate" in query_lower or "inspire" in query_lower:
        category = "motivational"
    elif "sad" in query_lower:
        category = "sad"
    elif "happy" in query_lower:
        category = "happy"
    
    if category:
        for q in quotes:
            if q.get("category", "").lower() == category:
                candidates.append(q)
        if not candidates:
            for q in quotes:
                if "tags" in q and any(category in tag.lower() for tag in q["tags"]):
                    candidates.append(q)
    else:
        candidates = quotes

    if candidates:
        chosen = random.choice(candidates)
        return chosen.get("text", "Here's a quote!")
    else:
        return "I don't have a specific quote for that, but I can cheer you up: Keep your head up!"


def print_help():
    help_text = """
Available Commands:
  teach -> <question> -> <answer>
      - Teach a new Q&A pair.
  personality -> <description>
      - Set personality description.
  csv -> <filepath>
      - Load Q&A pairs from a CSV file.
  learn -> <filepath>
      - Learn from a source file (HTML, CSS, JS, TS, Python, Java, etc.).
  train -> <filepath>
      - Load Q&A pairs from a text file (if the file has no '->', it is treated as a document).
  modeltrain
      - Train the deep-learning model using all learned data (from memory, documents, and conversation).
  analyze image -> <filepath>
      - Analyze an image file and return its top predictions.
  show sources
      - List the names of all loaded source files.
  exit or quit
      - Exit the application.
  (Any other text is treated as a query.)
"""
    print(help_text)

def main():
    print("Friday is online. Type your command (or type 'help' for available commands).")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            lower_input = user_input.lower()
            if lower_input in ["exit", "quit"]:
                break
            if lower_input == "help":
                print_help()
                continue
            if "->" in user_input:
                parts = user_input.split("->")
                if parts[0].strip().lower().startswith("teach"):
                    tokens = parts[0].strip().split()
                    if tokens[0].lower() != "teach":
                        print("Command not recognized or wrong number of arguments.")
                        continue
                    if len(parts) == 2:
                        question = " ".join(tokens[1:]).strip()
                        answer = parts[1].strip()
                        if question == "":
                            print("No question provided in teach command.")
                            continue
                        teach_ai(question, answer)
                        continue
                    elif len(parts) == 3:
                        teach_ai(parts[1].strip(), parts[2].strip())
                        continue
                    else:
                        print("Command not recognized or wrong number of arguments.")
                        continue
                elif parts[0].strip().lower() == "personality" and len(parts) == 2:
                    teach_personality(parts[1].strip())
                    continue
                elif parts[0].strip().lower() == "csv" and len(parts) == 2:
                    learn_from_csv(parts[1].strip())
                    continue
                elif parts[0].strip().lower() == "learn" and len(parts) == 2:
                    learn_from_source_file(parts[1].strip())
                    continue
                elif parts[0].strip().lower() == "train" and len(parts) == 2:
                    learn_from_file(parts[1].strip())
                    continue
                elif parts[0].strip().lower() == "modeltrain":
                    model_train_from_memory()
                    continue
                elif parts[0].strip().lower() == "analyze image" and len(parts) == 2:
                    result = analyze_image(parts[1].strip())
                    print("Friday (Image Analysis):\n" + result)
                    continue
                else:
                    print("Command not recognized or wrong number of arguments.")
                    continue
            elif lower_input == "modeltrain":
                model_train_from_memory()
                continue
            elif lower_input == "show sources":
                if memory.get("__code_files__"):
                    print("Loaded source files:")
                    for fname, info in memory["__code_files__"].items():
                        print(f"  - {fname} ({info['language']})")
                else:
                    print("No source files loaded yet.")
                continue
            response = get_response(user_input)
            print("Friday:", response)
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
