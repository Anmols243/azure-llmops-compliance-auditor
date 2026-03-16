import json
import os
import logging
from typing import Dict, Any, List

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

#State Schema
from backend.src.graph.state import VideoAuditState, ComplianceIssue

from backend.src.services.video_indexer import VideoIndexerService

logger = logging.getLogger("Compliance-auditor")
logging.basicConfig(level=logging.INFO)

# Responsible for Video to Text
def index_video_node(state:VideoAuditState) -> Dict[str,Any]:
    '''
    Downloads the youtube video from the url
    Uploads to Azure Video Indexer
    Extracts the insights
    '''

    video_url = state.get("vid_url")
    video_id_input = state.get("vid_id","video_demo")

    logger.info(f"----[Node:Indexer] Processing : {video_url}")

    local_filename = "temp_audit_video.mp4"

    try:
        vi_service = VideoIndexerService()

        # Download: yt-dlp
        if "youtube.com" in video_url or "youtu.be" in video_url:
            local_path = vi_service.download_youtube_video(video_url, output_path=local_filename)
        else: 
            raise Exception("Please provide a valid YouTube URL for this test.")
        
        # Upload
        azure_video_id = vi_service.upload_video(local_path, video_name = video_id_input)
        logger.info(f"Upload Success. Azure ID : {azure_video_id}")

        # Cleanup
        if os.path.exists(local_path):
            os.remove(local_path)

        # Wait
        raw_insights = vi_service.extract_data(azure_video_id)

        # Extract
        clean_data = vi_service.extract_data(raw_insights)
        logger.info("---[NODE: Indexer] Extraction Complete---")
        return clean_data
    
    except Exception as e:
        logger.error(f"Video Indexer Failed: {e}")
        return {
            "errors" : [str(e)],
            "final_status" : "FAIL",
            "transcript" : "",
            "ocr_test" : []
        }

# Node 2 : Compliance Auditor
def audio_content_node(state:VideoAuditState) -> Dict[str,Any]:
    '''
    Performs RAG to audit the content
    '''

    logger.info("----[Node: Auditor] querying Knowledge base & LLM")
    transcript = state.get("transcript","")
    if not transcript:
        logger.warning("No transcript available. Skipping Aduit....")
        return {
            "final_status" : "FAIL",
            "final_report" : "Audit skipped because video processing failed (No transcript)"
        }        
    
    llm = AzureChatOpenAI(
        azure_deployment= os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        azure_api_version= os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.0
    )

    embedding = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_api_version= os.getenv("AZURE_OPENAI_API_VERSION")
    )

    vector_store = AzureSearch(
        azure_search_endpoint= os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key= os.getenv("AZURE_SEARCH_API_KEY"),
        index_name= os.getenv("AZURE_SEARCH_INDEX_NAME"),
        embedding_function= embedding.embed_query
    )

    #RAG
    ocr_text = state.get("ocr_text", [])
    query_text = f"{transcript} {''.join(ocr_text)}"
    docs = vector_store.similarity_search(query_text,k=3)
    retrieved_rules = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""
            You are a senior brand compliance officer.
            OFFICIAL REGULATORY RULES:
            {retrieved_rules}
            INSTRUCTIONS:
            1. Analyze the Transcript and OCR text below.
            2. Indetify ANY violations of the rules.
            3. Return strictly JSON in the following format:
                {
                "compliance_results": [
                    {
                    "category": "Claim Validation",
                    "severity": "CRITICAL",
                    "description": "Explanation of the violation..."
                    }
                ],
                "status": "FAIL",
                "final_report": "Summary of findings..."
                }
            if no violations are found, set "status" to "PASS" and "compliance_results" to [].
            """
    
    user_message = f"""
                VIDEO_METADATA : {state.get('video_metadata',{})}
                TRANSCRIPT : {transcript}
                ON-SCREEN TEST (OCR) : {ocr_text}
                """
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])

        content = response.content
        return{
            "compliance_results" : content.get("complicance_results", []),
            "final_status" : content.get("status","FAIL"),
            "final_report" : content.get("final_report","No report generated")
        }
    
    except Exception as e:
        logger.error(f"System Error in Auditor Node : {str(e)}")
        # Logging raw response
        logger.error(f"Raw LLm response: {response.content if 'response' in locals() else 'None'}")
        return {
            "errors" : [str(e)],
            "final_status" : "FAILED"
        }