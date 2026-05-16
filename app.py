import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Page title
st.title("AI Document Assistant")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    # Read PDF
    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    st.success("PDF successfully loaded!")

    # Better chunking
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    # Load embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create embeddings
    chunk_embeddings = embedding_model.encode(chunks)

    # User question
    user_question = st.text_input("Ask a question about the document")

    if user_question:

        # Embed question
        question_embedding = embedding_model.encode([user_question])

        # Similarity search
        similarities = cosine_similarity(
            question_embedding,
            chunk_embeddings
        )[0]

        # Best chunk
        best_match_index = similarities.argmax()

        context = chunks[best_match_index]

        # Load local language model
        generator = pipeline(
            "text-generation",
            model="google/flan-t5-base"
        )

        # Better prompt
        prompt = f"""
You are an AI assistant answering questions about a document.

Use the context below to answer the question clearly and professionally.

Context:
{context}

Question:
{user_question}

Professional Answer:
"""

        # Generate answer
        result = generator(
            prompt,
            max_new_tokens=100,
            do_sample=False
        )

        answer = result[0]["generated_text"].replace(prompt, "")

        # Display result
        st.subheader("AI Answer")

        st.write("### Context:")
        st.write(context)

        st.write("### Question:")
        st.write(user_question)

        st.write("### Answer:")
        st.write(answer)
