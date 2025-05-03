from fastapi import FastAPI,Request, Response,APIRouter,UploadFile, Depends, HTTPException, status, File, Form
# from fastapi.responses import JSONResponse, FileResponse
from schemas import QueryIn, GenerateOut#, PDFQuestionResponse, MaskingResponse, ImageMaskingResponse, ErrorResponse
from retriever import generate_predicted_gold_inds
from evaluator import evaluate_program
from generator import infer, build_vocab, PointerProgramGenerator
# from masking import predict_and_mask, run_final_pattern_check, BERTForNER, entity_mapping,mask_predictions
from model_retriever import BertRetriever
# from qa_pipeline.main import convert_pdf_to_json
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from typing import Dict#, List, Any, Union, Optional
# import tempfile
from utils import find_most_relevant_sample, load_dataset, vectorize ,download_and_load_models#,render_text_to_image, extract_text_from_file,transform
# import pdfplumber
import torch
from transformers import BertTokenizerFast#, PegasusTokenizer, PegasusForConditionalGeneration, AutoTokenizer, DetrForObjectDetection, DetrImageProcessor, DetrConfig
import numpy as np
import json
import os
# import io
# import onnxruntime as ort
# import pytesseract
# from PIL import Image#, ImageDraw, ImageFont
# from docx import Document
from fastapi.middleware.cors import CORSMiddleware
os.environ["WANDB_DISABLED"] = "true"



bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
retriver_path,generator_path,vocab_path,train_path,test_path= download_and_load_models()

# Load model and tokenizer
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your client app to make requests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


dataset,test_dataset =load_dataset(train_path=train_path, test_path=test_path)
# Load TF-IDF vectorizer (assuming it's already fitted)
vectorizer = vectorize(dataset,max_features=5000)  # Replace with your vectorizer if different
vocab_size = bert_tokenizer.vocab_size
retriever_model = BertRetriever(vocab_size=vocab_size,embed_size=60, num_layers=4, num_heads=12, hidden_dim=240, num_labels=2)   
retriever_model.load_state_dict(torch.load(retriver_path, map_location=device))
retriever_model.eval()

# Load Generator model
with open(vocab_path, "r") as f:
    vocab_dict = json.load(f)
    vocab = list(vocab_dict.keys())

generator_model = PointerProgramGenerator(vocab_dict)
# print("generator_model: ", generator_model)
generator_model=torch.load(generator_path, map_location=device,weights_only=False)
generator_model.eval()


@app.get("/health")
async def health_check():
    return {"status": "ok"}
##check if the server is running
@app.get("/")
async def root():
    return {"message": "fin_gpt server is running"} 


#input is only question
@app.post("/question")
async def find_relevant(request: Request):
    data = await request.json()
    user_question = data.get('question')
    
    if user_question:
        # Find the most relevant sample
        selected_sample = find_most_relevant_sample(user_question, dataset, vectorizer)

        # Extract the relevant question and any other desired info
        response_data = {
            "qa": {
                "question": selected_sample["qa"]["question"],
            },
            "pre_text": selected_sample["pre_text"],
            "post_text": selected_sample["post_text"],
            "table": selected_sample["table"],
        }
        
        # Pass the dictionary directly to run_pipeline
        return await run_pipeline(QueryIn(**response_data))
    else:
        return {"error": "No question provided"}, 400

#input is question and pdf file

# @app.post("/ask_pdf", response_model=GenerateOut)
# async def ask_pdf(question: str = Form(...),pdf: UploadFile = File(...)):
#     # Step 1: Save the uploaded PDF to a temporary file
#     pdf_path = f"temp_{pdf.filename}"
#     try:
#         with open(pdf_path, "wb") as f:
#             f.write(await pdf.read())

#         # Step 2: Convert the PDF to JSON
#         model_input = convert_pdf_to_json(pdf_path, question)
#         print("model_input: ", model_input)
#         # Step 3: Parse the JSON to extract fields for run_pipeline
#         model_input_data = json.loads(model_input)
#         query_in = QueryIn(
#             qa={"question": question},
#             pre_text=model_input_data.get("pretext", []),
#             post_text=model_input_data.get("posttext", []),
#             table=model_input_data.get("table", [])
#         )

#         # Step 4: Call run_pipeline with the extracted data
#         result = await run_pipeline(query_in)

#         # Step 5: Return the result
#         return result

#     except Exception as e:
#         # Log the error and raise an HTTPException
#         print(f"Error in /ask_pdf: {e}")
#         raise HTTPException(status_code=500, detail="An error occurred while processing the PDF.")

#     finally:
#         # Step 6: Clean up the temporary file
#         if os.path.exists(pdf_path):
#             os.remove(pdf_path)

#input is question and pretext,posttext,table
@app.post("/retrive", response_model=GenerateOut)
async def run_pipeline(data: QueryIn):
    # Step 1: Format input record for retriever
      # Parse the JSON body
    # print("data: ", data)
    record = {
        "qa": {"question": data.qa.question},
        "pre_text": data.pre_text,
        "post_text": data.post_text,
        "table": data.table,
    }
    # record=data
    # Step 2: Use retriever to get gold_inds
    gold_inds_raw = generate_predicted_gold_inds(record, retriever_model, bert_tokenizer,threshold=0,num_candidates=2)
    # print("gold_inds_raw: ", gold_inds_raw)
    gold_inds = {k: v["sentence"] for k, v in gold_inds_raw}
    # return {"gold_inds": gold_inds}
    # Step 3: Use generator to get program
    full_input = data.qa.question + " " + " ".join([v for v in gold_inds.values() if any(c.isdigit() for c in v)])
    print(full_input)
    encoded = bert_tokenizer(full_input, return_tensors="pt", padding="max_length", truncation=True, max_length=512, return_offsets_mapping=True)
    input_tokens = bert_tokenizer.convert_ids_to_tokens(encoded["input_ids"].squeeze(0))
    sample = {
        "input_ids": encoded["input_ids"].squeeze(0),
        "input_mask": encoded["attention_mask"].squeeze(0),
        "input_tokens": input_tokens
    }
    program = infer(generator_model, sample, vocab_dict)
    # Step 4: Evaluate the program
    # print("program: ", program)
    program=" , ".join(program[:-1] if len(program)>1 else program )
    result = evaluate_program(program, data.table)
    gold_inds=list(gold_inds.values())
    # Step 5: Return all
    return GenerateOut(gold_inds=gold_inds, program=program, result=str(result))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
