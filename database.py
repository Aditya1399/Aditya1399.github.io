from langchain.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR
from langchain.vectorstores import Chroma

instructor_embeddings=HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large')
def vectordb(documents):
    if len(documents)*1.33>512:
        documents=documents[:512]
    persist_directory='db'
    embedding=instructor_embeddings
    vectordb=Chroma.from_documents(documents=documents,embedding=embedding,persist_directory=persist_directory)
    retriever=vectordb.as_retriever(search_kwargs={"k":1})
    return retriever