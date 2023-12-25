from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI 
from langchain.prompts import PromptTemplate


from openai import AzureOpenAI
from dotenv import load_dotenv
import os
load_dotenv()



def GptAnnotation(Resume,Tags):
    from datetime import date

    
    template = """
        As an adept annotator, your assignment is to carry out a detailed analysis of the provided resume, focusing on the identification of specific elements. 
        These elements correspond to the following categories: {Tags}.

        Your annotations are crucial in the process of training Named Entity Recognition (NER) models using SpaCy. 
        Careful and accurate tagging of these elements in the resume will significantly enhance the model's efficiency and accuracy.

        Please examine the following resume and methodically annotate the elements that match the provided tags.

        Resume:
        {Resume}

        For your guidance, here's an example of how the annotated output should be structured:

        Example of Annotated Resume:

        Please ensure to enclose the tagged text in single quotes ('') like so:

            'John Doe' -<Name>-
            'Stanford University, Computer Science' -<Education>-

            Work Experience:
            'Software Engineer at Google' -<WorkExperience>-
            'Developed a machine learning model for recommendation systems' -<Skills>-

        Please remember to keep the original resume text, formatting, and spacing intact. 
        Your task is to add tags to the existing content without any modifications.
        Ensure to use the appropriate tag from the provided list for each identified element. 
        Attention to detail is paramount in this task.
    """
    prompt = PromptTemplate(
    input_variables=["Resume","Tags"],
    template=template

    ) 

    llm = AzureChatOpenAI(model = "gpt-35-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.run(Tags = Tags ,Resume = Resume)
    return response
