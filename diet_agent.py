import os
import sys
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Define the path for the knowledge base
DIET_DATA_PATH = os.path.join("knowledge_base", "diet_data.txt")

def create_diet_agent():
    """
    Creates the ConversationalRetrievalChain agent using Google Gemini.
    """
    print("Loading knowledge base...")
    loader = TextLoader(DIET_DATA_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    print("Creating vector store with Google embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query")
    vectorstore = FAISS.from_documents(texts, embeddings)

    print("Initializing the agent with Google Gemini...")
    diet_agent = ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0),
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    print("Diet Agent (powered by Google) is ready!")
    return diet_agent

def main():
    """
    The main function to run the chatbot.
    """
    agent = create_diet_agent()
    
    chat_history = []
    
    print("\n--- Welcome to the Diet AI Assistant (Google Edition) ---")
    print("Ask any question about the diet information I have. Type 'exit' to quit.")

    while True:
        try:
            query = input("You: ")
            
            # THE ONLY CHANGE IS HERE: Added .strip() to remove whitespace
            if query.strip().lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            result = agent.invoke({
                "question": query, 
                "chat_history": chat_history
            })
            answer = result["answer"]
            
            chat_history.append((query, answer))
            
            print(f"DietAI: {answer}")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()