from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from utils import get_embedding_model
from config import GOOGLE_API_KEY


class RAG:

    def __init__(self):

        embeddings = get_embedding_model()

        self.db = FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )

        self.llm = ChatGoogleGenerativeAI(
           model="gemini-3.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2
        )

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful assistant.

Use ONLY the information present in the context.

If the answer is not available, reply:
"I couldn't find that information in the document."

Context:
{context}

Question:
{input}
"""
        )

        self.chain = create_stuff_documents_chain(
            self.llm,
            prompt
        )

    def ask(self, question):

        docs = self.db.similarity_search(
            question,
            k=3
        )

        try:
            response = self.chain.invoke(
                {
                    "context": docs,
                    "input": question
                }
            )
            return response

        except Exception as e:
            return f"Error: {str(e)}"