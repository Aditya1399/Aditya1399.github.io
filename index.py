# app.py
from driver_code import main
from flask import Flask,render_template,request,redirect,url_for,jsonify
from langchain.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR
from langchain.vectorstores import Chroma

import os
import tempfile
import requests
from langchain import PromptTemplate, LLMChain,HuggingFaceHub
from config import WHITE, GREEN, RESET_COLOR, model_name

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from nltk.tokenize import word_tokenize
from transformers import pipeline

import os
#from app import GithubUrl
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR
from langchain.vectorstores import Chroma
import os
import uuid
import subprocess
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from langchain.document_loaders import DirectoryLoader, NotebookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import clean_and_tokenize

os.environ['HUGGINGFACEHUB_API_TOKEN']='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'
app=Flask("__name__")
#global response
@app.route("/",methods=["POST","GET"])
def home():
    
    if request.method=='POST':
        global response
        GithubUrl=request.form['GithubUrl']
        username=GithubUrl.split('/')[-1]
        #adding the username to the github api
        api=f'https://api.github.com/users/{username}/repos'.format(username)
        r1=[]
        response=requests.get(api)
        #getting the response dictionary in a json format
        response_dict=response.json()
        
        #extracting the repo urls from the user file and storing it in a list
        for r in response_dict:
            r1.append(r['html_url'])
        
        #Iterating over each url cloning into repository
        
        index1=[]
        document1=[]
        
        repo_path="./repos"
        if not os.path.exists(repo_path):
             os.makedirs(repo_path)
             for singlerepository in r1:
                path=repo_path
                clone='git clone ' + singlerepository
                os.chdir(path)
                os.system(clone)
        else:
             pass
        
        
        #calling the load and index files function to load and split the data 
        
        extensions = ['txt', 'md', 'markdown', 'rst', 'py', 'js', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php', 'scala', 'html', 'htm', 'xml', 'json', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'sh', 'bash', 'css', 'scss', 'sql', 'gitignore', 'dockerignore', 'editorconfig', 'ipynb']

        file_type_counts = {}
        documents_dict = {}

        for ext in extensions:
            glob_pattern = f'**/*.{ext}'
            try:
                loader = None
                if ext == 'ipynb':
                    loader = NotebookLoader(str(repo_path), include_outputs=True, max_output_length=20, remove_newline=True)
                else:
                    loader = DirectoryLoader(repo_path, glob=glob_pattern)

                loaded_documents = loader.load() if callable(loader.load) else []
                if loaded_documents:
                    file_type_counts[ext] = len(loaded_documents)
                    for doc in loaded_documents:
                        file_path = doc.metadata['source']
                        relative_path = os.path.relpath(file_path, repo_path)
                        file_id = str(uuid.uuid4())
                        doc.metadata['source'] = relative_path
                        doc.metadata['file_id'] = file_id

                        documents_dict[file_id] = doc
            except Exception as e:
                print(f"Error loading files with pattern '{glob_pattern}': {e}")
                continue

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        split_documents = []
        for file_id, original_doc in documents_dict.items():
            split_docs = text_splitter.split_documents([original_doc])
            for split_doc in split_docs:
                split_doc.metadata['file_id'] = original_doc.metadata['file_id']
                split_doc.metadata['source'] = original_doc.metadata['source']

            split_documents.extend(split_docs)

        index = None
        if split_documents:
            tokenized_documents = [clean_and_tokenize(doc.page_content) for doc in split_documents]
            index = BM25Okapi(tokenized_documents)
        
        #calling the vectordvb function from database.py to retrive the documents from vector store
        instructor_embeddings=HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large')
        if len(split_documents)*1.33>512:
            split_documents=split_documents[:512]
            persist_directory='db'
            embedding=instructor_embeddings
            vectordb=Chroma.from_documents(documents=split_documents,embedding=embedding,persist_directory=persist_directory)
            retriever=vectordb.as_retriever(search_kwargs={"k":1})
            
            
        
        
        #creating a prompt template
        template = """
                    

                    Instr:
                    1. Answer based on context/docs.
                    2. Focus on repo/code.
                    3. Your task is to tell most technically challenging repository based on the context data 
                    5. Unsure? Say "I am not sure".

                    {question}


                    Answer:

                    Provide repository name in the answer with explanation why you selected that
                    """

        prompt = PromptTemplate(
                        template=template,
                        input_variables=["question"]
                    )
        prompt1=prompt.format(question="Which is the most technically complex and challenging repository and tell the reason for that?")
                    #creating the pipeline for text generation
        pipe=pipeline(
                        'text-generation',
                        model='gpt2-large',
                        max_length=512,
                        top_p=0.95,
                        temperature=0.1,
                        repetition_penalty=1.15
                    )
        local_llm=HuggingFacePipeline(pipeline=pipe)
        llm_chain = RetrievalQA.from_chain_type(llm=local_llm,chain_type="stuff",retriever=retriever,return_source_documents=True)
        response=llm_chain(prompt1)
        return render_template('hello.html',data=response)
    else:
        return render_template('hello.html')
    
        

    

    
    
if __name__ == "__main__":
    app.run(debug=True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      