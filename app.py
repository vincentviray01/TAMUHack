from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
#from werkzeug.datastructures import  FileStorage
from transformers import pipeline#AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from transformers import BartTokenizer, BartForConditionalGeneration
import wikipedia
import os
import shutil

app = Flask(__name__, static_folder="static")

global videoFileName
global transcriptFileName

videoFileName = None
transcriptFileName = None

@app.route('/')
def index():
   return render_template("index.html")
	
@app.route('/upload', methods = ['POST'])
def upload():
   global videoFileName
   global transcriptFileName
   if request.method == 'POST':
      if videoFileName is None or transcriptFileName is None:
         return render_template("index.html", video_file_name = videoFileName, transcript_file_name = transcriptFileName, videoOrTranscriptIsNotMissing = False)

      query = request.form["query"]
      key = request.form["key"]

            
      

      question = query

      model_name = "deepset/roberta-base-squad2"

      # GET ANSWER FROM UPLOADED TRANSCRIPT
      with open(transcriptFileName, "r") as reader:
         transcript_text = reader.read()      

      # a) Get predictions
      nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
      QA_input = {
         'question': question,
         'context': transcript_text
      }
      res = nlp(QA_input)
      modelAnswer = list(res.values())[3]
      transcript_answer = modelAnswer[0].upper() + modelAnswer[1:]
      if transcript_answer[-1] != ".":
         transcript_answer = transcript_answer + "."

      # GET ANSWER FROM WIKIPEDIA
      #wikstr = wikipedia.summary("Quicksort")

      try:
         wikstr = wikipedia.page(key).content
         wikstr = wikstr.replace("\n", " ")
      except:
         wikstr = ""

      # a) Get predictions
      if len(wikstr) > 0:
         nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
         QA_input = {
            'question': question,
            'context': wikstr
         }
         res = nlp(QA_input)
         modelAnswer = list(res.values())[3]
         wikipedia_answer = modelAnswer[0].upper() + modelAnswer[1:]
         if wikipedia_answer[-1] != ".":
            wikipedia_answer = wikipedia_answer + "."
      else:
         wikipedia_answer = ""

      
      model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
      tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
      # GET SUMMARY FOR UPLOADED TRANSCRIPT                  
      ARTICLE_TO_SUMMARIZE = transcript_text
      inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

      summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=1024)
      transcript_summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

      # GET SUMMARY FOR WIKIPEDIA PAGE
      # model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
      # tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
      if len(wikstr) > 0:
         ARTICLE_TO_SUMMARIZE = wikstr
         inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

         summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=1024)
         wikipedia_summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]               
      else:
         wikipedia_summary = ""

      return render_template("answer.html", video_file_name = videoFileName, transcript_file_name = transcriptFileName, key = key, wikipedia_answer=wikipedia_answer, transcript_answer=transcript_answer, query=query, wikipedia_summary=wikipedia_summary,transcript_summary=transcript_summary)

@app.route('/upload_files', methods = ['POST'])
def upload_files():
   global videoFileName
   global transcriptFileName
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))

      try:
         shutil.rmtree('static')
      except:
         pass
      os.mkdir('static')

      try:         
         shutil.rmtree('transcripts')
      except:
         pass
      os.mkdir("transcripts")

      os.rename(f.filename, f"static/{f.filename}")

      f2 = request.files['file2']
      f2.save(secure_filename(f2.filename))

      os.rename(f2.filename, f"transcripts/{f2.filename}")

      videoFileName = f.filename
      transcriptFileName = f"transcripts/{f2.filename}"


      return render_template("index.html", video_file_name = videoFileName, transcript_file_name = transcriptFileName)

		
if __name__ == '__main__':
   app.run(debug = True)