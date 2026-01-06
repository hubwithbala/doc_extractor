import os
import json
import requests
import pdfplumber
import logging
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pydantic import BaseModel, ValidationError, Field, BeforeValidator, AliasChoices
from typing import List, Optional, Literal, Annotated

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

OLLAMA_MODEL = "ministral-3:3b" 
OLLAMA_API_URL = "http://localhost:11434/api/generate"



def clean_content(v):
    if isinstance(v, list): return " ".join(str(i) for i in v)
    if v is None: return ""
    return str(v)

class DocNode(BaseModel):
    heading: str = Field(validation_alias=AliasChoices('heading', 'section', 'title'))
    content: Annotated[str, BeforeValidator(clean_content)] = ""
    is_duplicate: bool = False
    children: List['DocNode'] = []

DocNode.model_rebuild()

class DocumentAnalysis(BaseModel):
    quality_rating: Literal['Good', 'Medium', 'Bad']
    quality_reason: str
    document_type: str
    summary: str
    structure: List[DocNode]



def extract_text_from_pdf(filepath):
    full_text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text: full_text += text + "\n"
    except Exception as e:
        logger.error(f"PDF Extraction failed: {e}")
        raise Exception("File is corrupted or cannot be read.")
    return full_text

def mark_duplicates(nodes: List[DocNode], seen_headings=None):
    if seen_headings is None: seen_headings = set()
    for node in nodes:
        h_clean = node.heading.strip().lower()
        if h_clean in seen_headings and h_clean:
            node.is_duplicate = True
            node.heading += " [Duplicate]"
        else:
            seen_headings.add(h_clean)
        if node.children: mark_duplicates(node.children, seen_headings)
    return nodes

def generate_formatted_text(data: dict) -> str:
   
    lines = []
    lines.append("==================================================")
    lines.append("            DOCUMENT ANALYSIS REPORT              ")
    lines.append("==================================================\n")
    
    lines.append(f"Document Type: {data['document_type']}")
    lines.append(f"Quality:       {data['quality_rating']} ({data['quality_reason']})")
    lines.append("\n------------------ SUMMARY ------------------\n")
    lines.append(data['summary'])
    lines.append("\n------------------ CONTENT ------------------\n")

    def recurse_print(nodes, level=0):
        indent = "    " * level
        for node in nodes:
            heading = node['heading']
            # Add visual marker for duplicates
            if node.get('is_duplicate'):
                heading += " (DUPLICATE)"
            
            lines.append(f"{indent}# {heading}")
            if node['content']:
                lines.append(f"{indent}  {node['content']}")
            lines.append("") # Empty line for spacing
            
            if node.get('children'):
                recurse_print(node['children'], level + 1)

    recurse_print(data['structure'])
    return "\n".join(lines)

def query_ollama(text):
    system_prompt = (
        "You are an intelligent document architect. Your goal is to structure the provided text logically.\n"
        "RULES:\n"
        "1. SEMANTIC GROUPING: If the text lacks clear headings, group paragraphs by MEANING and generate logical headings yourself.\n"
        "2. DUPLICATES: If you see repeated sections, include them.\n"
        "3. QUALITY & SUMMARY: assess quality and write a summary.\n"
        "4. OUTPUT: Return valid JSON matching the schema below.\n\n"
        "Schema:\n"
        "{ 'quality_rating': 'Good'|'Medium'|'Bad', 'quality_reason': 'str', 'document_type': 'str', 'summary': 'str', "
        "'structure': [{'heading': 'str', 'content': 'str', 'children': []}] }"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system_prompt}\n\nDOCUMENT TEXT:\n{text}",
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096}
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get('response', '')
    except Exception as e:
        raise Exception(f"Ollama is not responding: {e}")

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        raw_text = extract_text_from_pdf(filepath)
        if not raw_text or len(raw_text.strip()) < 10:
             return jsonify({'error': 'Document appears empty.'})

        json_str = query_ollama(raw_text)
        
        try:
            data_dict = json.loads(json_str)
            validated_data = DocumentAnalysis(**data_dict)
            validated_data.structure = mark_duplicates(validated_data.structure)
            final_json = validated_data.model_dump()
            
        except (ValidationError, json.JSONDecodeError) as ve:
            return jsonify({'error': "AI Generation Error: The model failed to structure this document."})

        # --- 1. SAVE JSON ---
        base_name = os.path.splitext(filename)[0]
        json_filename = f"{base_name}_analysis.json"
        json_path = os.path.join(app.config['OUTPUT_FOLDER'], json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, indent=4)

        # --- 2. SAVE FORMATTED TEXT ---
        text_content = generate_formatted_text(final_json)
        text_filename = f"{base_name}_report.txt"
        text_path = os.path.join(app.config['OUTPUT_FOLDER'], text_filename)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_content)

        return jsonify({
            'status': 'success', 
            'data': final_json,
            'download_json': f"/download/{json_filename}",
            'download_text': f"/download/{text_filename}"
        })

    except Exception as e:
        return jsonify({'error': str(e).split('\n')[0]})

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)