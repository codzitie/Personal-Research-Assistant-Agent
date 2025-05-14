import os
from fastapi import UploadFile
from config import client, embeddings, llm
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Tuple
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
# Global variable to store chat history
global_chat_history = []
# Global variable to store the initialized retriever (to avoid reprocessing the PDF each time)
global_retriever = None

async def save_upload_file_temp(upload_file: UploadFile) -> str:
    temp_file = f"temp_{upload_file.filename}"
    with open(temp_file, "wb") as buffer:
        buffer.write(await upload_file.read())
    return temp_file

def remove_temp_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)

def transcribe(audio_path):
    with open(audio_path, "rb") as file:
        translation = client.audio.translations.create(
            file=(audio_path, file.read()),
            model="whisper-large-v3",
            prompt="Specify context or spelling",
            response_format="json",
            temperature=0.0
        )
    return translation.text

def transcribe_and_rag(content_pdf_path, question_path):
    question = transcribe(question_path)
    print("User Query Transcript", question)
    
    loader = PDFMinerLoader(content_pdf_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)
    result = qa.run({"query": question})
    return result


class ResearchAssistantRAG:
    def __init__(self, google_scholar_search_func):
        """
        Initialize the Research Assistant RAG system
        
        Args:
            google_scholar_search_func: Function to search Google Scholar
        """
        # Global variables for maintaining state
        self.global_chat_history = []
        self.global_retriever = None
        
        # LLM and Embeddings
        self.llm = llm
        self.embeddings = embeddings
        
        # Google Scholar search function
        self.google_scholar_search = google_scholar_search_func
        
        # Create research agent
        self.research_agent = self._create_research_agent()
    
    def _create_research_agent(self):
        """
        Create an agent with research-specific tools
        """
        # Define tools for the agent
        search_tool = Tool(
            name="Google Scholar Search",
            func=self.google_scholar_search,
            description="Searches Google Scholar for academic papers. "
                        "Input is a research query. "
                        "Returns paper names and DOIs matching the query."
        )
        
        # Additional tools can be added here (e.g., PDF download, abstract extraction)
        tools = [search_tool]
        
        # Initialize agent
        agent = initialize_agent(
            tools, 
            self.llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        return agent
    
    def _load_pdf_to_retriever(self, content_pdf_path: str):
        """
        Load PDF and create vector store retriever
        
        Args:
            content_pdf_path: Path to the PDF file
        """
        # Load and process the PDF
        loader = PDFMinerLoader(content_pdf_path)
        data = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(data)
        
        # Create vector store
        db = FAISS.from_documents(docs, self.embeddings)
        self.global_retriever = db.as_retriever()

    def _load_result_to_retriever(self, text_data: str):
        """
        Load result and create vector store retriever
        
        Args:
            content_pdf_path: Path to the PDF file
        """
        
        document = Document(page_content=text_data)
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents([document])
        
        # Create vector store
        db = FAISS.from_documents(docs, self.embeddings)
        self.global_retriever = db.as_retriever()
        
    
    def rag_with_research(self, 
                           question: str, 
                           content_pdf_path: str = None, 
                           reset_history: bool = False,
                           force_research: bool = False) -> Tuple[str, List[Tuple[str, str]]]:
        """
        RAG function with research agent capabilities
        
        Args:
            question: Current user question
            content_pdf_path: Optional path to the PDF file
            reset_history: Flag to reset chat history
        
        Returns:
            answer: The response
            new_chat_history: Updated chat history
        """
        # Reset history if requested
        if reset_history:
            self.global_chat_history = []
            self.global_retriever = None
        
        # Load PDF if provided
        if content_pdf_path:
            print('Processing PDF:', content_pdf_path)
            self._load_pdf_to_retriever(content_pdf_path)
        
        # Determine if research is needed using a more sophisticated approach
        needs_research = force_research
        
        if not needs_research:
            # Use the LLM to determine if the question requires academic research
            research_decision_prompt = f"""
            Determine if the following question requires academic research to answer properly.
            The question might be asking about papers, studies, or recent research,
            OR it might simply state a research topic that should be looked up.
            
            Question: {question}
            
            Respond with only 'YES' if academic research is needed, or 'NO' if not.
            """
            
            research_decision = self.llm.invoke(research_decision_prompt).content.strip().upper()
            print(f"Research decision: {research_decision}")
            needs_research = research_decision == "YES"
        
        # Only perform research if needed
        research_results = ""
        if needs_research:
            print("Initiating research for question...")
            research_results = self.research_agent.run(f"Find academic papers related to: {question}")
            print("Research Results:", research_results)
            self._load_result_to_retriever(research_results)
            
        else:
            print("Skipping research phase - question doesn't appear to require it")

        
        # Prepare prompts for context-aware QA
        condense_question_prompt = PromptTemplate.from_template("""
        Given the following conversation and a follow up question, rephrase the follow up question 
        to be a standalone question that captures all relevant context from the conversation.
        
        Chat History:
        {chat_history}
        
        Follow Up Question: {question}
        
        Standalone Question:
        """)
        
        qa_prompt = PromptTemplate.from_template("""
        You are a helpful research assistant answering questions based on provided documents.
        At present you will recieve title, doi, abstract and some informations of paper rather than it's content.
        Try interpreting with those information and explain it.

        Question: {question}
                                                 
        Context: {context}
        
        Chat History: {chat_history}
        
        Answer with insights from the research and any available context:
        """)
        
        # Create memory object to store chat history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Add existing chat history to memory
        for human_msg, ai_msg in self.global_chat_history:
            memory.chat_memory.add_user_message(human_msg)
            memory.chat_memory.add_ai_message(ai_msg)
        
        # Prepare conversational chain
        # if self.global_retriever:
        qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.global_retriever,
            memory=memory,
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={
                "prompt": qa_prompt
            }
        )
        
        # Run the chain
        result = qa({"question": question})
        answer = result["answer"]
        # else:
        #     # Fallback to agent-only response if no retriever
        #     answer = self.research_agent.run(f"""
        #     Provide a comprehensive answer to the following question, 
        #     incorporating insights from recent research:
        #     {question}
            
        #     Previous Research Results: {research_results}
        #     """)
        
        # Update chat history
        self.global_chat_history.append((question, answer))
        
        return answer, self.global_chat_history

