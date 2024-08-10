import streamlit as st
from main import *
import os
import json

st.title("Swipe Task: Information Retrieval from Invoices")

uploaded_file = st.file_uploader("Upload an Invoice", type=["pdf"])

if uploaded_file is not None:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    # Save the uploaded file to the temporary directory
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    output_dict = main(file_path=file_path)

    with st.spinner("Processing"):
        output_dict = json.loads(output_dict)
        costumer_deets = output_dict["Costumer Details"]
        products = output_dict["Products"]
        total_amount = output_dict["Total Amount"]

    st.subheader("Information Retrieved:")
    st.text(f"Costumer Details: {costumer_deets}")
    st.text(f"Products: {products}")
    st.text(f"Total Amount: {total_amount}")