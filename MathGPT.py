import os
import streamlit as st
from PIL import Image
import pytesseract
from paddleocr import PaddleOCR, draw_ocr
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import numpy as np
import pytesseract
import easyocr


# os.environ['SSL_CERT_FILE'] = 'C:\\Users\\RSPRASAD\\AppData\\Local\\.certifi\\cacert.pem'
GROQ_API_KEY = st.secrets('GROQ_API_KEY')
llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")
pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\RSPRASAD\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
ocr = PaddleOCR(use_angle_cls=True, lang='en')
# Function to extract text from an image using OCR
def extract_text_from_image(image):
    try:
        # Convert PIL Image to NumPy array
        image_np = np.array(image)
        
        # Perform OCR using PaddleOCR
        results = ocr.ocr(image_np, cls=True)
        
        # Extract text from results
        extracted_text = ""
        for line in results:
            for word_info in line:
                extracted_text += word_info[1][0] + " "
        
        return extracted_text.strip() if extracted_text else "No text found in the image."
   
    except Exception as e:
        return f"Error processing image: {e}"
    # try:
    #     # Use pytesseract to do OCR on the image
    #     # text = pytesseract.image_to_string(image)
    #     text = pytesseract.image_to_string(image, lang='eng', config='--psm 7')
    #     return text.strip()
    # except Exception as e:
    #     return f"Error processing image: {e}"

# Function to solve a math problem using Groq API and Mistral LLM
def solve_math_problem_with_groq(problem_text):
    try:
        # Initialize the Groq LLM with the Mistral model
       
        # Create a prompt template for solving the math problem
        prompt_template = PromptTemplate(
            input_variables=["problem"],
            template="Solve the following math problem:\n\n{problem}\n\n"
        )

        # Create an LLM chain with the prompt template
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt_template
        )

        # Run the LLM chain with the problem text
        solution = llm_chain.run(problem=problem_text)
        return solution.strip()
    except Exception as e:
        return f"Error solving the problem: {e}"

# Streamlit app
def main():
    st.title("Math Solver using ChatGroq, Mistral, and LangChain")
    st.write('Developed by Ravi Shankar Prasad- https://www.linkedin.com/in/ravi-shankar-prasad-371825101/')

    # Input method selection
    input_method = st.radio("Select input method:", ("Upload Image", "Enter Text"))

    problem_text = ""

    if input_method == "Upload Image":
        # Image upload
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            # Extract text from image
            problem_text = extract_text_from_image(image)
            if problem_text:
                st.write("Extracted Text:", problem_text)
            else:
                st.write("No text found in the image. Please upload a clearer image.")
    else:
        # Text input
        problem_text = st.text_area("Enter math problem:")

    if problem_text:
        if st.button("Solve"):
            # Solve the math problem using Groq and Mistral
            solution = solve_math_problem_with_groq(problem_text)
            st.write("Solution:", solution)

if __name__ == "__main__":
    main()
