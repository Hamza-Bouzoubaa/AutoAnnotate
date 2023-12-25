from langchain.vectorstores import FAISS
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from PyPDF2 import PdfReader
from docx import Document
import markdown

from dotenv import load_dotenv
import os
load_dotenv()


## setting up AzureOpenAI
os.environ["openai_api_type"] = "azure"
os.environ["azure_openai_endpoint"] = "https://hamzaopenai.openai.azure.com/"
os.environ["openai_api_version"]  = "2023-07-01-preview"
os.environ["openai_api_key"] = "2225fc60c1e64b289c5a05493f6e1b4c"



def ReadFile(FilePath):
    # Get the file extension
    file_extension = FilePath.split('.')[-1].lower()

    if file_extension == 'pdf':
        doc_reader = PdfReader(FilePath)
        raw_text = ''
        for i, page in enumerate(doc_reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

    elif file_extension == 'docx':
        doc = Document(FilePath)
        raw_text = ''
        for paragraph in doc.paragraphs:
            raw_text += paragraph.text + '\n'

    elif file_extension == 'txt':
        with open(FilePath, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
    elif file_extension == 'md':
        with open(FilePath, 'r', encoding='utf-8') as file:
            md_content = file.read()
            raw_text = markdown.markdown(md_content)

    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    return raw_text




def SplitResumeText(Text):

    chunks = Text.split("\n")

    return chunks





def GenerateEmbeddings(ResumePath):  #Works with PDF DOCX and TXT

    ResumeText = ReadFile(ResumePath)
    
    ResumeSections = SplitResumeText(ResumeText)  # splitting the raw text into sections

    embeddings =  AzureOpenAIEmbeddings(model = "text-embedding-ada-002")  # Setting up OpenAI embedding

    
    dbResume  = FAISS.from_texts(ResumeSections,embeddings)      # Generating and storing Resume embeddings 
    

    return dbResume


def QuerySimilaritySearch(Query,db,k=3):
    similar = db.similarity_search(Query,k)
    page_content_array = [doc.page_content for doc in similar]
    return page_content_array



