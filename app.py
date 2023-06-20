# app.py
from driver_code import main
from flask import Flask,render_template,request,redirect,url_for,jsonify

import os
import tempfile
import requests
from langchain import PromptTemplate, LLMChain,HuggingFaceHub
from config import WHITE, GREEN, RESET_COLOR, model_name
from utils import format_user_question
from file_processing import clone_github_repo, load_and_index_files
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from nltk.tokenize import word_tokenize
from transformers import pipeline
from database import vectordb
import os
#from app import GithubUrl
from langchain.llms import HuggingFacePipeline
os.environ['HUGGINGFACEHUB_API_TOKEN']='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'
app=Flask("__name__")

@app.route("/",methods=["POST","GET"])
def home():
    if request.method=='POST':
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
        
        repo_path="C:/Users/Aditya/OneDrive/Desktop/githubautomatedanalysis/RepoReader/repos"

        #clone_github_repo(r1,"C:/Users/Aditya/OneDrive/Desktop/githubautomatedanalysis/RepoReader/repos")
        #calling the load and index files function to load and split the data 
        data2=(load_and_index_files(repo_path))
        #calling the vectordvb function from database.py to retrive the documents from vector store
        vectordb1=vectordb(data2)
        
        
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
        llm_chain = RetrievalQA.from_chain_type(llm=local_llm,chain_type="stuff",retriever=vectordb1,return_source_documents=True)
        response=llm_chain(prompt1)
        return render_template('index.html',data=response)
    
        

    else:
            return render_template('index.html')
    

    
    
if __name__ == "__main__":
    app.run(debug=True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      