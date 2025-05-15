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
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import requests
from bs4 import BeautifulSoup
from difflib import get_close_matches
import json
from langchain.schema import Document
import re
import google_scolar.web_search as web_search
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
    def __init__(self, advanced_google_scholar_search=None, scihub_pdf=None):
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
        self.google_scholar_search = advanced_google_scholar_search

        self.paper_download = self._scihub_pdf
        
        # Create research agent
        self.research_agent = self._create_research_agent()

        self.url = None

    def _create_research_agent(self):
        """
        Create an agent with research-specific tools
        """
        # Define tools for the agent
        search_tool = Tool(
            name="Google Scholar Search",
            func=self.google_scholar_search,
            description="Use this tool when you need to find academic papers on a GENERAL TOPIC or get an overview of a research area. "
                        "Input is a research topic or question. "   
                        "Returns paper titles, authors, DOIs, and abstracts. "
                        "Use this tool FIRST when the user asks about a general research topic."
        )
        
        download_tool = Tool(
        name="Download Research Paper",
        func=self.paper_download,
        description=(
            "Use this tool when you need to retrieve a specific academic paper. "
            "This tool supports input as a DOI (e.g., 10.1234/abcd), a direct URL to the paper, "
            "or the paper's title and/or author name (e.g., 'Deep learning by Y LeCun'). "
            "The tool will search for the paper in the previously retrieved list of research papers and return the matching result if found. "
            "Only use this tool when the user asks for a particular paper or its detailed content."
        )
    )
        # Additional tools can be added here (e.g., abstract extraction)
        tools = [search_tool, download_tool]
        
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
    
    def _scihub_pdf(self,query):

        scihub_url = ''
        doi_or_url = ''
        self.download_paper_tool(query)
        # Step 1: Fetch the HTML page from Sci-Hub
        response = requests.get(f"{scihub_url}/{doi_or_url}")
        response.raise_for_status()

        # Step 2: Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Step 3: Find the <embed> tag with the PDF
        embed = soup.find('embed', id='pdf')
        filename = "pdf/"
        if embed and 'src' in embed.attrs:
            pdf_url = embed['src']
            
            # Fix relative URL
            if pdf_url.startswith('/'):
                pdf_url = scihub_url + pdf_url
            
            # Step 4: Download the PDF
            pdf_response = requests.get(pdf_url, stream=True)
            pdf_response.raise_for_status()

            filename += doi_or_url.replace("/", "_") + ".pdf"
            with open(filename, 'wb') as f:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"PDF downloaded as: {filename}")
        else:
            print("PDF embed not found.")

    def download_paper_tool(self,query: str) -> str:
        """
        Tool to download or return the link of a specific paper based on title and/or author.
        """
        if not self.url:
            return "No URLs available to search from. Please run a prior research query first."
        
        matches = []
        for entry in self.url:
            if query.lower() in entry['Title'].lower() or query.lower() in entry['Authors'].lower():
                matches.append(entry)

        # If no direct matches, try fuzzy match
        if not matches:
            titles = [entry['Title'] for entry in self.url]
            close = get_close_matches(query, titles, n=1, cutoff=0.5)
            if close:
                matches = [entry for entry in self.url if entry['Title'] == close[0]]

        if matches:
            paper = matches[0]
            return f"Title: {paper['Title']}\nAuthors: {paper['Authors']}\nURL: {paper['URL']}"
        else:
            return f"No matching paper found for: {query}"

    def _determine_research_approach(self, question: str):
        """
        Determines the appropriate research approach based on the user's question
        
        Args:
            question: User's research question
            
        Returns:
            dict: Contains flags for which tools to use and how
        """
        tool_guidance_prompt = f"""
        Analyze the following research question and determine the appropriate research tool(s) to use:
        
        Question: {question}
        
        Determine which category this question falls into:
        1. GENERAL TOPIC EXPLORATION: User is asking about a broad research area, wants an overview, 
        or needs to find relevant papers on a topic (use Google Scholar Search)
        2. SPECIFIC PAPER REQUEST: User is asking about a specific paper they've identified by author,
        title, year, DOI, or URL and needs detailed information from that paper (use Download Research Paper). In this category definetely search tool must be used
        3. BOTH: User needs both general exploration and specific paper retrieval
        
        Respond with a JSON object with these fields:
        - "category": 1, 2, or 3
        - "use_search": false
        - "use_download": true/false
        - "specific_doi": null or the DOI/URL if explicitly mentioned
        - "explanation": brief reason for this classification
        """
        
        response = self.llm.invoke(tool_guidance_prompt)
        print('ressssssssssssss',response)
        try:
            # First try with markdown-style code block
            match = re.search(r'```(?:json)?\s*({.*?})\s*```', response.content, re.DOTALL)

            if not match:
                # Fallback: try to directly extract JSON-looking object
                match = re.search(r'({.*})', response.content, re.DOTALL)

            if match:
                json_str = match.group(1)
                guidance = json.loads(json_str)
                print('json_str', json_str)
                return guidance
            else:
                print("Could not find valid JSON block in response.")
                return None

        except:
            # Fallback if JSON parsing fails
            print("Warning: Failed to parse tool guidance response")
            return {
                "category": 1,  # Default to search
                "use_search": True,
                "use_download": False,
                "specific_doi": None,
                "explanation": "Default to search due to parsing error"
            }
    
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
            Determine whether the following user query requires academic research tools to be used.

            This includes cases where:
            - The user explicitly mentions a research paper, article, or author
            - The user refers to a study, research result, or scientific claim
            - The user states a research topic that should be looked up (e.g., for background or literature review)

            Question: {question}

            Respond with only 'YES' if academic research tools should be used, or 'NO' if not.
            """
            
            research_decision = self.llm.invoke(research_decision_prompt).content.strip().upper()
            print(f"Research decision: {research_decision}")
            needs_research = research_decision == "YES"
            print('need research >>>>>>>>>>>>>>>>>>>>>>>>>>>')
        
        # Only perform research if needed
        research_results = ""
        if needs_research:
            print("Initiating research for question...")
                # Get guidance on which tools to use
            guidance = self._determine_research_approach(question)
            print(f"Research approach: {guidance['explanation']}")
            
            if guidance['category'] == 1:  # General topic exploration
                research_results = self.research_agent.run(
                    f"Find academic papers related to: {question}. Use ONLY the Google Scholar Search tool."
                )
                
            elif guidance['category'] == 2:  # Specific paper request
            
                research_results = self.research_agent.run(
                    f"This query is about a specific paper related to: {question}. Use the download tool"
                )

            else:  # Both approaches needed
                research_results = self.research_agent.run(
                    f"Research this query thoroughly: {question}. First use Google Scholar Search to find relevant papers, then if a specific important paper is identified, download it using the Download Research Paper tool."
                )
            self.url = web_search.urls
            print('urls^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',self.url)
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
                You are a helpful research assistant. Your task is to answer questions based on provided information about academic papers.

                You will be given the **title**, **DOI**, **abstract**, and other basic details of each paper — not the full content. Use this information to interpret the research and answer the question as best as you can.

                For each relevant paper you refer to in your answer, **list the title and author(s)** to support your explanation.

                Question: {question}

                Context (paper information): {context}

                Chat History: {chat_history}

                Please provide a well-explained answer based on the available context:
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
    
