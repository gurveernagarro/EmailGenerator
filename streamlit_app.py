import os
import openai
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.mapreduce import MapReduceChain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from PIL import Image
import base64
from io import BytesIO

st.set_page_config(page_title="Nagarro", page_icon="img/Nagarro_logo.png",)

st.markdown('''<style>.css-1egvi7u {margin-top: -4rem;}</style>''',
    unsafe_allow_html=True)

st.markdown('''<style>.css-znku1x a {color: #9d03fc;}</style>''',
    unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-znku1x a {color: #9d03fc;}</style>''',
    unsafe_allow_html=True)  # lightmode

st.markdown('''<style>.css-qrbaxs {min-height: 0.0rem;}</style>''',
    unsafe_allow_html=True)

st.markdown('''<style>.stSpinner > div > div {border-top-color: #9d03fc;}</style>''',
    unsafe_allow_html=True)

st.markdown('''<style>.css-15tx938{min-height: 0.0rem;}</style>''',
    unsafe_allow_html=True)

hide_decoration_bar_style = '''<style>header {visibility: hidden;}</style>'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

hide_streamlit_footer = """<style>#MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_footer, unsafe_allow_html=True)

# Connect to OpenAI GPT-3

openai.api_type = "azure"
openai.api_base = "https://emailgeneratordemo.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "4461d4ebc79a45bca18557145962a4f3"
#################################################################################################
##################################
###############
#######

def gen_mail_contents(email_contents,sender,recipient,style):     

        openaiq = OpenAI(temperature=0.7,openai_api_key="4461d4ebc79a45bca18557145962a4f3",deployment_id="EmailGeneratorDemo02")                                         
        map_prompt = """Below is a section of a information about {prospect}
       Write a concise summary about {prospect}. If the information is not about {prospect}, exclude it from your summary.{text}
% CONCISE SUMMARY:"""
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "prospect"])

        combine_prompt = """
Your goal is to write the email content for {Reason} by {company} to {prospect}.
A good campaign email combines information about the reason of the campaign.Length of email should be less than 50 words.
% INFORMATION ABOUT {company}:
{company_information}

% INFORMATION ABOUT {prospect}:
{text}

% INCLUDE THE FOLLOWING PIECES IN YOUR RESPONSE:

- Start the email with 'We'
- Keep the email in {style} tone 
- Start with {Reason} of the email

- A 1-2 sentence description about {company}, be brief
- End your content with a call-to-action such as asking them to set up time to talk more


% YOUR RESPONSE:
"""
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["Reason", "company", "prospect", \
                                                                         "text","company_information","style"])

        chain = load_summarize_chain(openaiq,
                             chain_type="map_reduce",
                             map_prompt=map_prompt_template,
                             combine_prompt=combine_prompt_template,
                             verbose=True
                            )
        text_splitter = CharacterTextSplitter()

        ############################ Information about the Prospect
        with open("./BlackBaud.txt") as f:
            state_of_the_union = f.read()
            texts = text_splitter.split_text(state_of_the_union)
       
        docs = [Document(page_content=t) for t in texts[:3]]

        ############################ Information about the Sender
        Filename="./"+sender+".txt"

        with open(Filename) as f:
            state_of_the_Naggaro = f.read()

        ##########################
           
        ############################### Getting into chain    

        output = chain({ "input_documents": docs,##These Docs are basically data about the Targeted customer
                "company": sender, \
                "company_information" :state_of_the_Naggaro ,\
                  "Reason" : email_contents[0], \
                "prospect" : recipient,\
                "style" : style
               })
        langout=output['output_text']
        print(langout)       
        
        return langout
###################################################################
################################
###################

def gen_mail_format(sender, recipient, style, email_contents):
    email_contents = gen_mail_contents(email_contents,sender,recipient,style)
    print(email_contents)
    return email_contents
#########################################################
#####################################
####################

def generate_Variable_content(Variable,Context):
    openaiq = OpenAI(temperature=0.7,openai_api_key="4461d4ebc79a45bca18557145962a4f3",deployment_id="EmailGeneratorDemo02")   
    prompt = PromptTemplate(
    input_variables=["Variable", "Variable_Context"],
    template="""Write a content having {Variable} as heading and give the response considering: {Variable_Context}
    % INCLUDE THE FOLLOWING PIECES IN YOUR RESPONSE:
- Keep the responses really short and crisp 
    % YOUR RESPONSE:
    """,
)
    chain = LLMChain(llm=openaiq, prompt=prompt)
    Langout=(chain.run({
    'Variable': Variable,
    'Variable_Context': Context
    }))
    print(Langout)
    return Langout

def get_base64_from_image(image):
    if isinstance(image, Image.Image):
        if image.mode == "RGBA":
            image = image.convert("RGB")  # Convert RGBA to RGB
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    return ""
def getImageString(uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Read the image file
        # Display the uploaded image
        #Convert the image to base64
        img_base64 = get_base64_from_image(image)
        # Generate HTML string with the image
        image_string = f'<div style="max-width: 100%;"><img src="{img_base64}" alt="Uploaded Image" style="max-width:100%;"></div>'
    return image_string

   
    



##########################################################
#######################################
#####################

def main_gpt3emailgen():

    st.image('img/image_banner.png')  
    st.markdown('Generate professional sounding emails based on your cheap comments - powered by Artificial Intelligence (OpenAI GPT-3)! Implemented by '
        '[Naggaro](https://www.nagarro.com/en/)'
        )
    st.write('\n') 

    st.subheader('\nWhat is your email all about?\n')
    
    
    with st.expander("SECTION - Email Input", expanded=True):

            input_c1 = st.text_input('Enter email contents down below! (currently 2x seperate topics supported)', 'topic 1')
            input_c2 = st.text_input('', 'topic 2 (optional)')
            uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

            email_text = "" 
            col1, col2, col3, space,col7,col8,col5,col9,col10,col4 = st.columns([5, 5, 5, 0.5,5,5,5,5,5,5])
            with col1:
                input_sender = st.text_input('Sender Name', '[rephraise]')
            with col2:
                input_recipient = st.text_input('Recipient Name', '[recipient]')
            with col3:
                input_style = st.selectbox('Writing Style',
                                        ('formal', 'motivated', 'concerned', 'disappointed'),
                                        index=0)
         #   with col6:
          #      input_target=st.selectbox('Customer target',('High Value','Low Value'))
            with col5:
                template = st.selectbox('Template Type',  ('Template1','Template2'))   
                if template!='':
                    variable="Templates"+'\\'+template+".html"
                    with open(variable, 'r') as file:  
                        html_string = file.read()
                    
            with col7:
                Variable2 = st.text_input('First Underheading', 'Enter a Heading')
            with col8:
                Variable2_Context = st.text_input('Context for First Underheading', 'Enter context for the UnderHeading')    


            with col9:
                Variable3 = st.text_input('Second Underheading', 'Enter a Heading')
            with col10:
                Variable3_Context = st.text_input('Context for Second Underheading', 'Enter context for the UnderHeading')    
                

            with col4:
                st.write("\n")  # add spacing
                st.write("\n")  # add spacing
                if st.button('Generate Email'):
                    with st.spinner():

                        input_contents = []  # let the user input all the data
                        if (input_c1 != "") and (input_c1 != 'topic 1'):
                            input_contents.append(str(input_c1))
                        if (input_c2 != "") and (input_c2 != 'topic 2 (optional)'):
                            input_contents.append(str(input_c2))

                        if (len(input_contents) == 0):  # remind user to provide data
                            st.write('Please fill in some contents for your message!')
                        if (len(input_sender) == 0) or (len(input_recipient) == 0):
                            st.write('Sender and Recipient names can not be empty!')

                        if (len(input_contents) >= 1):  # initiate gpt3 mail gen process
                            if (len(input_sender) != 0) and (len(input_recipient) != 0):
                                email_text = gen_mail_format(input_sender,
                                                            input_recipient,
                                                            input_style,
                                                            input_contents)
                                Variable2_Content=generate_Variable_content(
                                                            Variable2,Variable2_Context)
                                Variable3_Content=generate_Variable_content(
                                                            Variable3,Variable3_Context)
                                
            
            
                                
        
          
    
    if email_text != "":
        st.write('\n')  # add spacing
       # with st.container():
         #   st.markdown(Variable2_Content)

        html_string=html_string.replace("DATA1 ",email_text)
        html_string=html_string.replace("IMAGE1",getImageString(uploaded_file))
        html_string=html_string.replace("Variable1",input_c1)  
        html_string=html_string.replace("Variable2",Variable2) 
        html_string=html_string.replace("Context2",Variable2_Content) 
        html_string=html_string.replace("Variable3",Variable3) 
        html_string=html_string.replace("Context3",Variable3_Content)                
        st.components.v1.html(html_string,width=800, height=1500, scrolling=True)



            
            #st.markdown(Variable2_response)
            #Variable3 = st.text_input('Variable3', 'Enter a Heading')
            #Variable3_response=generate_Variable_content(email_text,Variable3)
    
        



      #  st.subheader('\nHere is the Email Body Content\n')
      #  with st.expander("SECTION - Email Output", expanded=True):
      #      st.markdown(email_text)  #output the results
    

    
if __name__ == '__main__':
    # call main function
    main_gpt3emailgen()
