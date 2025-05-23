import os
from config import client, embeddings, llm
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Tuple
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import requests
from bs4 import BeautifulSoup
import json
from langchain.schema import Document
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google_scolar.web_search as web_search
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# Global variable to store chat history
global_chat_history = []
# Global variable to store the initialized retriever (to avoid reprocessing the PDF each time)
global_retriever = None

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

        self.guidance = None
        
        # Google Scholar search function
        self.google_scholar_search = advanced_google_scholar_search

        self.paper_download = self._scihub_pdf
        
        # Create research agent
        self.research_agent = self._create_research_agent()

        self.url = None

        self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

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
            "Use this tool when the user asks for a specific academic paper. "
            "You can provide the title, author name, or relevant keywords (e.g., 'Deep learning by Y LeCun'). "
            "The tool will internally search for the paper using scholarly databases and return the best matching result. "
            "DOI or URL is not required from the user; the tool will attempt to find and resolve them automatically."
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
        print('in scihub ')
        scihub_url = 'https://sci-hub.se/'
        doi_or_url,title = self._sentence_similarity(query=query,url_list=self.url)
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
            

            # Clean filename: remove illegal characters
            safe_filename = re.sub(r'[^\w\-_.]', '_', title) 
            if '.pdf' not in safe_filename:
                safe_filename += '.pdf'
            filepath = os.path.join("pdf", safe_filename)

            with open(filepath, 'wb') as f:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            filepath = filepath.replace("\\", "/")
            print(f"PDF downloaded as: {filepath}")
            return filepath
        # elif not embed:
        #     pdf_url = '/downloads'
    
        else:
            print("PDF embed not found.")

    def _sentence_similarity(self,query: str, url_list: list, threshold: float = 0.15):
        """
        Returns the best matching entry from url_list based on TF-IDF cosine similarity with the query.
        """
        # url
        titles = [entry['Title'] for entry in url_list]

        # Build TF-IDF vectors
        vectorizer = TfidfVectorizer().fit([query] + titles)
        query_vec = vectorizer.transform([query])
        title_vecs = vectorizer.transform(titles)

        # Compute cosine similarities
        scores = cosine_similarity(query_vec, title_vecs)[0]

        # Get index of best match above threshold
        best_idx = scores.argmax()
        best_score = scores[best_idx]

        if best_score >= threshold:
            paper = url_list[best_idx]
            return paper["URL"],paper["Title"]
            
        else:
            return "No sufficiently similar title found."
    

    
    def _get_prompt_body_for_decision(self):
        # Moved the large string block into a helper method for readability
        return """
            Classify the current user question into one of the following categories.
            Consider if a paper or topic has already been discussed or its content loaded by checking the Conversation History.

            1. GENERAL_TOPIC_EXPLORATION:
            - The user is asking about a broad research area, concept, or method.
            - They are looking for background information or relevant papers on a topic.
            - Action: Use Google Scholar Search tool.

            2. SPECIFIC_PAPER_REQUEST:
            - The user refers to a specific research paper using its title, DOI, URL, or author.
            - Mainly they ask by its title and author
            - They may ask to explain, summarize, or discuss the content of that paper.
            - Action: Use Download Research Paper tool to fetch and process the paper.

            3. BOTH:
            - The user wants both a general topic exploration and a specific paper.
            - Action: Use both tools as appropriate.

            4. NO_TOOL_REQUIRED (Answer from existing retrieved context or general knowledge):
            - The user asks a follow-up question about a paper whose content has *already been loaded and processed* (e.g., available via the existing retriever).
            - Or, the question can be answered from general LLM knowledge or the immediate chat history without needing to consult external academic databases or download new papers.
            - Example: After a paper has been downloaded and summarized:
                Human: "Download and summarize 'Attention is All You Need'."
                AI: (Provides summary based on downloaded PDF)
                Current Question: "What was the key innovation in that paper's methodology?"
                → {{"category": 4, "use_search": false, "use_download": false, "specific_doi": null, "explanation": "Follow-up on an already processed paper. Answer from existing retrieved context."}}
            - Action: Do not use search or download tools; proceed to answer using existing context.

            Return your answer as a JSON object with the following fields:
            - "category": An integer (1, 2, 3, or 4).
            - "use_search": Boolean (true/false, indicates if Google Scholar Search is needed).
            - "use_download": Boolean (true/false, indicates if Download Research Paper is needed).
            - "specific_doi": String (null, or DOI/URL if explicitly mentioned and relevant for category 2 or 3).
            - "explanation": A brief string explaining the classification.

            Examples:

            1. Current User Question: "Explain spectral entropy"
            (Assuming no relevant history)
            → {{"category": 1, "use_search": true, "use_download": false, "specific_doi": null, "explanation": "User is exploring a general concept."}}

            2. Current User Question: "Explain 'Cancer prediction using machine learning' by Michael Jordan"
            (Assuming no relevant history or this is a new request for this paper)
            → {{"category": 2, "use_search": false, "use_download": true, "specific_doi": null, "explanation": "User wants a specific paper by author and title."}}

            3. Current User Question: "Find papers on self-supervised learning and download the one by Yann LeCun"
            → {{"category": 3, "use_search": true, "use_download": true, "specific_doi": null, "explanation": "User is exploring a topic and requesting a specific paper."}}

            4. Conversation History:
            Human: Can you find papers on 'attention mechanisms'?
            AI: (Lists some papers, including 'Attention is All You Need', and this paper was downloaded and processed)
            Current User Question: "Explain the methodology in 'Attention is All You Need' more."
            → {{"category": 4, "use_search": false, "use_download": false, "specific_doi": "DOI_of_Attention_is_All_You_Need_if_known", "explanation": "User is asking a follow-up about a paper already processed. Answer from existing context."}}

            Keep in mind that return only json as output
            """

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
            
            decision_memory = ConversationBufferMemory(
                memory_key="chat_history", # Must match placeholder in prompt
                return_messages=True
            )
            for human_msg, ai_msg in self.global_chat_history:
                decision_memory.chat_memory.add_user_message(human_msg)
                decision_memory.chat_memory.add_ai_message(ai_msg)

            prompt_body = self._get_prompt_body_for_decision()
            research_decision_template_str = f"""
                Analyze the current user question, considering the conversation history, to determine the appropriate tool(s) or approach.

                Conversation History:
                {{chat_history}}

                Current User Question: {{input}}

                {prompt_body}
                """
            # 3. Create PromptTemplate object
            research_decision_prompt_template = PromptTemplate(
                input_variables=["chat_history", "input"], # 'input' is default for ConversationChain
                template=research_decision_template_str
            )

            # 4. Initialize and run the decision-making ConversationChain
            decision_chain = ConversationChain(
                llm=self.llm,
                memory=decision_memory, # Use the dedicated, populated memory
                prompt=research_decision_prompt_template,
                verbose=True # Good for debugging
            )
            
            # Invoke the chain and get the response string
            chain_response_dict = decision_chain.invoke({"input": question})
            self.guidance = chain_response_dict[decision_chain.output_key] 
            needs_research =  "YES"
        
        # Only perform research if needed
        research_results = ""
    
        print('guide',self.guidance)
        json_str = self.guidance
        if not isinstance(json_str, dict):
            # json_str = json.dumps(self.guidance, indent=4)
            cleaned = re.sub(r"```[a-zA-Z]*", "", self.guidance).replace("```", "").strip()
            match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
            if match:
                # Step 2: Fix invalid JSON formatting
                json_text = match.group(0)
                cleaned = (
                    json_text.replace("TRUE", "true")
                        .replace("FALSE", "false")
                        .replace("NULL", "null")
                        .lower()  
                )

                # Step 3: Load as Python dictionary
                json_str = json.loads(cleaned)
                print('json_str',json_str)
        if json_str['category'] == 1:  # General topic exploration
            print('entering search . . . .')
            research_results = self.research_agent.run(
                f"Find academic papers related to: {question}. Use ONLY the Google Scholar Search tool."
            )
            
        elif json_str['category'] == 2:  # Specific paper request
            print('entering scihub search . . . .')
            retriever_check = False
            filename = self._scihub_pdf(question)
            if filename:
                self._load_pdf_to_retriever(filename)
                retriever_check = True

        elif json_str['category'] == 3:  # Both approaches needed
            retriever_check = False
            print('entering both . . . .')
            research_results = self.research_agent.run(
                f"Research this query thoroughly: {question}. First use Google Scholar Search to find relevant papers, then if a specific important paper is identified, download it using the Download Research Paper tool."
            )
            self._load_result_to_retriever(research_results)
            if filename:
                self._load_pdf_to_retriever(filename)
                retriever_check = True
            
            # self._load_result_to_retriever(research_results)
        else:
            retriever_check = True
            print("Skipping research phase - question doesn't appear to require it")

        self.url = web_search.urls
        print("Research Results:", research_results)

        if json_str['category'] != 1:
            if retriever_check == False:
                fail_msg = 'Sorry for the inconvenience.... Access restricted for the paper mentoined'
                return fail_msg, self.global_chat_history
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
            You are a helpful research assistant. You are given the full text or detailed content of a scientific article.

            Your task depends on the user's question:

            1. If the user asks for a general summary, provide a comprehensive overview that includes:
                - **Title of the paper**
                - **Authors** (if available)
                - **Objective** of the research
                - **Methods** used
                - **Key findings and results**
                - **Conclusions**
                - **Applications or implications**
                - **Limitations or future work**, if mentioned

            2. If the user asks about a **specific topic, method, result, or term**, extract and explain that part **in detail**, strictly based on the context. Do **not** provide information that is not explicitly present in the article.

            3. If the question is too vague, default to a detailed summary of the paper.

            4. If the context does not contain enough information to answer the question, respond with: **"Not enough data in the provided context."**

            - Never copy from the text directly; always paraphrase in clear, concise language.
            - Do not speculate or hallucinate. Only use facts and statements found in the provided article.

            ---

            **User Question:** {question}

            **Article Content (Context):** {context}

            **Chat History:** {chat_history}

            ---

            Provide a clear and well-organized response based only on the given article.
            """)
                        
            
            # Add existing chat history to memory
            for human_msg, ai_msg in self.global_chat_history:
                self.memory.chat_memory.add_user_message(human_msg)
                self.memory.chat_memory.add_ai_message(ai_msg)
            
            # Prepare conversational chain
            # if self.global_retriever:
            qa = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.global_retriever,
                memory=self.memory,
                condense_question_prompt=condense_question_prompt,
                combine_docs_chain_kwargs={
                    "prompt": qa_prompt
                }
            )
            # print('self.global_retriever 33333333333333333333333333333333',self.global_retriever)
            # Run the chain
            result = qa({"question": question})
            answer = result["answer"]
            self.global_chat_history.append((question, answer))
            
            return answer, self.global_chat_history
                   
        else:
            formatted_output = ''
            for idx, paper in enumerate(self.url, 1):
                formatted_output += (
                    f"{idx}. Title: {paper['Title']}\n"
                    f"   Authors: {paper['Authors']}\n"
                    f"   Abstract: {paper['Abstract']}\n\n"
                )

            self.global_chat_history.append((question, formatted_output))
            return formatted_output, self.global_chat_history
    
