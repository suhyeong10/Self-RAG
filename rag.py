import os
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer = None

class ReflectionToken(Enum):
    """
    Self-RAG reflection tokens
    """

    RETRIEVE_YES = "yes"
    RETRIEVE_NO = "no"
    RETRIEVE_CONTINUE = "continue"
    ISREL_RELEVANT = "relevant"
    ISREL_IRRELEVANT = "irrelevant"
    ISSUP_FULLY_SUPPORTED = "fully supported"
    ISSUP_PARTIALLY_SUPPORTED = "partially supported"
    ISSUP_NO_SUPPORT = "no support"
    ISUSE_5 = "5"
    ISUSE_4 = "4"
    ISUSE_3 = "3"
    ISUSE_2 = "2"
    ISUSE_1 = "1"

@dataclass
class State:
    query: str
    retrieved_docs: List[Dict[str, Any]] = None
    generated_response: str = ""
    reflection_tokens: List[str] = None
    is_relevant: Optional[bool] = None
    is_supported: Optional[bool] = None
    is_useful: Optional[int] = None
    retrieval_needed: bool = False
    max_retries: int = 3
    current_retry: int = 0

    def __post_init__(self):
        if self.retrieved_docs is None:
            self.retrieved_docs = []
        if self.reflection_tokens is None:
            self.reflection_tokens = []

class SelfRAG:
    """
    Wrapper for Self-RAG model
    """

    def __init__(self, model_name: str = "selfrag/selfrag_llama2_7b"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def load_model(self):
        if self.model is None:
            logger.info(f"Loading Self-RAG model: {self.model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name
                )
                
                global tokenizer
                tokenizer = self.tokenizer

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype='bfloat16'
                )

                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device='cpu',
                    top_k=None,
                    top_p=None,
                    temperature=None
                )

                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
    
    def format_prompt(self, query: str, paragraph: Optional[str] = None) -> str:
        """
        Format the prompt for the Self-RAG model
        """

        prompt = f"### Instruction:\n{query}\n\n### Response:\n"
        if paragraph is not None:
            prompt += f"[Retrieval]<paragraph>{paragraph}</paragraph>"
        
        return prompt

    def predict_retrieve(self, query: str) -> bool:
        self.load_model()
        prompt = self.format_prompt(query)

        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt")
            output_ids = self.model.generate(
                **input_ids,
                max_new_tokens=100,
                temperature=None,
                do_sample=False,
                top_k=None,
                top_p=None,
                pad_token_id=32015
            )

            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
            
            if response.startswith(prompt):
                response = response[len(prompt):]
            else:
                response = response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return True

        return "[Retrieval]" in response

    def generate_w_docs(self, query: str, docs: List[str]) -> Tuple[str, List[str]]:
        self.load_model()

        best_response = ""
        best_score = -1
        all_reflection_tokens = list()

        for doc in docs:
            try:
                prompt = self.format_prompt(query, doc.page_content)    

                response = self.pipeline(
                    prompt, 
                    return_full_text=False,
                    skip_special_tokens=False
                )
        
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                continue
        
            reflection_tokens = self._extract_reflection_tokens(response)
            all_reflection_tokens.extend(reflection_tokens)

            score = self._calculate_response_score(reflection_tokens)

            if score > best_score:
                best_score = score
                best_response = response

        return best_response, all_reflection_tokens

    def generate_wo_docs(self, query: str) -> Tuple[str, List[str]]:
        self.load_model()
        prompt = self.format_prompt(query)

        try:
            response = self.pipeline(
                prompt,
                return_full_text=False,
                skip_special_tokens=False
            )

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating the response.", []

        reflection_tokens = self._extract_reflection_tokens(response)
        return response, reflection_tokens

    def _extract_reflection_tokens(self, response: str) -> List[str]:
        tokens = []

        if "[No Retrieval]" in response:
            tokens.append("no_retrieval")
        elif "[Retrieval]" in response:
            tokens.append("retrieval")
        elif "[Continue to Use Evidence]" in response:
            tokens.append("continue_evidence")

        if "[Relevant]" in response:
            tokens.append("relevant")
        elif "[Irrelevant]" in response:
            tokens.append("irrelevant")

        if "[Fully supported]" in response:
            tokens.append("fully_supported")
        elif "[Partially supported]" in response:
            tokens.append("partially_supported")
        elif "[No support / Contradictory]" in response:
            tokens.append("no_support")

        for i in range(1, 6):
            if f"[Utility:{i}]" in response:
                tokens.append(f"utility_{i}")
                break

        return tokens

    def _calculate_response_score(self, reflection_tokens: List[str]) -> float:
        score = 0.0

        if "relevant" in reflection_tokens:
            score += 1.0
        elif "irrelevant" in reflection_tokens:
            score -= 1.0

        if "fully_supported" in reflection_tokens:
            score += 2.0
        elif "partially_supported" in reflection_tokens:
            score += 1.0
        elif "no_support" in reflection_tokens:
            score -= 2.0
        
        for i in range(1, 6):
            if f"utility_{i}" in reflection_tokens:
                score += i * 0.5
                break
        
        return score

class Retriever:
    def __init__(self, pdf_path: str, save_path: str):
        self.pdf_path = pdf_path
        self.save_path = save_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=256
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )

    def _get_docs(self):
        docs_list = list()

        if os.path.isdir(self.pdf_path):
            pdf_files = [file_name for file_name in os.listdir(self.pdf_path) if file_name.endswith(".pdf")]
            for file_name in tqdm(pdf_files, desc="Loading PDF files", unit="file", ncols=150):
                pdf_file_path = os.path.join(self.pdf_path, file_name)
                docs_list.append(PyPDFLoader(pdf_file_path).load())
                
        documents_list = [item for sublist in docs_list for item in sublist]
        
        return documents_list

    def _load_retriever(self):
        if not os.path.isfile(os.path.join(self.save_path, "docs.pkl")):
            docs = self._get_docs()
            split_docs = self.text_splitter.split_documents(docs)
            pickle.dump(split_docs, open(os.path.join(self.save_path, "docs.pkl"), "wb"))
        else:
            split_docs = pickle.load(open(os.path.join(self.save_path, "docs.pkl"), "rb"))

        if not os.path.isdir(os.path.join(self.save_path, "database")):
            db = FAISS.from_documents(split_docs, self.embeddings)
            db.save_local(os.path.join(self.save_path, "database"))

            retriever = db.as_retriever()
        else:
            db = FAISS.load_local(
                os.path.join(self.save_path, "database"),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )

            retriever = db.as_retriever(
                search_kwargs={
                    "k": 5
                }
            )

        return retriever

RAG_MODEL = SelfRAG()
RETRIEVER = Retriever(
    pdf_path="dataset/pdf",
    save_path="dataset"
)._load_retriever()

def should_retrieve(state: State) -> State:
    logger.info("Checking if retrieval is needed...")

    retrieval_needed = RAG_MODEL.predict_retrieve(state.query)
    state.retrieval_needed = retrieval_needed

    logger.info(f"Retrieval needed: {retrieval_needed}")

    return state

def retrieve_documents(state: State) -> State:
    logger.info("Retrieving documents...")

    state.retrieved_docs = RETRIEVER.invoke(state.query)

    logger.info(f"Retrieved {len(state.retrieved_docs)} documents")

    return state

def generate_w_retrieval(state: State) -> State:
    logger.info("Generating response with retrieved documents...")

    response, reflection_tokens = RAG_MODEL.generate_w_docs(state.query, state.retrieved_docs)

    state.generated_response = response
    state.reflection_tokens = reflection_tokens

    if "relevant" in reflection_tokens:
        state.is_relevant = True
    elif "irrelevant" in reflection_tokens:
        state.is_relevant = False
    
    if "fully_supported" in reflection_tokens:
        state.is_supported = True
    elif "no_support" in reflection_tokens:
        state.is_supported = False
    
    for i in range(1, 6):
        if f"utility_{i}" in reflection_tokens:
            state.is_useful = i
            break
    
    logger.info(f"Generated response with tokens: {reflection_tokens}")

    return state

def generate_wo_retrieval(state: State) -> State:
    logger.info("Generating response without retrieval...")

    response, reflection_tokens = RAG_MODEL.generate_wo_docs(state.query)

    state.generated_response = response
    state.reflection_tokens = reflection_tokens
    
    for i in range(1, 6):
        if f"utility_{i}" in reflection_tokens:
            state.is_useful = i
            break
    
    logger.info(f"Generated response with tokens: {reflection_tokens}")
    
    return state

def check_quality(state: State) -> State:
    logger.info("Checking response quality...")

    if state.retrieval_needed:
        quality_good = (
            state.is_relevant is True and 
            state.is_supported is True and 
            state.is_useful and state.is_useful >= 3
        )
    else:
        quality_good = state.is_useful and state.is_useful >= 3
    
    if not quality_good and state.current_retry < state.max_retries:
        state.current_retry += 1
        logger.info(f"Quality check failed, retrying... (attempt {state.current_retry})")

        state.generated_response = ""
        state.reflection_tokens = []
        state.is_relevant = None
        state.is_supported = None
        state.is_useful = None

    return state

def decide_next_step(state: State) -> str:
    if state.current_retry >= state.max_retries:
        logger.info("Max retries reached, ending")
        return "end"
    
    if state.generated_response and state.is_useful and state.is_useful >= 3:
        if state.retrieval_needed:
            if state.is_relevant and state.is_supported:
                logger.info("High quality response with retrieval, ending")
                return "end"
        else:
            logger.info("High quality response without retrieval, ending")
            return "end"
    
    if state.retrieval_needed:
        return "retrieve"
    else:
        return "generate_wo_retrieval"

def create_rag_graph() -> StateGraph:
    workflow = StateGraph(State)

    workflow.add_node("should_retrieve", should_retrieve)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate_w_retrieval", generate_w_retrieval)
    workflow.add_node("generate_wo_retrieval", generate_wo_retrieval)
    workflow.add_node("check_quality", check_quality)

    workflow.add_edge("should_retrieve", "retrieve")
    workflow.add_edge("retrieve", "generate_w_retrieval")
    workflow.add_edge("generate_w_retrieval", "check_quality")
    workflow.add_edge("generate_wo_retrieval", "check_quality")

    workflow.add_conditional_edges(
        "should_retrieve",
        lambda state: "retrieve" if state.retrieval_needed else "generate_wo_retrieval"
    )

    workflow.add_conditional_edges(
        "check_quality",
        decide_next_step,
        {
            "retrieve": "retrieve",
            "generate_wo_retrieval": "generate_wo_retrieval",
            "end": END
        }
    )

    workflow.set_entry_point("should_retrieve")

    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)

def self_rag(query: str) -> Dict[str, Any]:
    logger.info(f"Running Self-RAG for query: {query}")

    initial_state = State(query=query)

    graph = create_rag_graph()
    result = graph.invoke(
        initial_state,
        config={
            "configurable": {
                "thread_id": "user1"
            }
        }
    )

    logger.info("Self-RAG execution completed")
    return {
        "query": result.query,
        "response": result.generated_response,
        "retrieval_needed": result.retrieval_needed,
        "retrieved_docs": result.retrieved_docs,
        "reflection_tokens": result.reflection_tokens,
        "quality_metrics": {
            "is_relevant": result.is_relevant,
            "is_supported": result.is_supported,
            "is_useful": result.is_useful
        },
        "retries": result.current_retry
    }

if __name__ == "__main__":
    result = self_rag("What is FaithfulRAG?")
    print(result)
