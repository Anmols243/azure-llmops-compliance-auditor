import operator
from typing import Annotated, List, Dict, Optional, Any, TypedDict

class ComplianceIssue(TypedDict):
    category : str
    description : str
    severity : str
    timestamp : Optional[str] 

class VideoAuditState(TypedDict):
    '''
    Defines the data schema for langgraph execution content.
    Main container : Holds all the information about the audit
    right from the initial URL to the final report.
    '''
    vid_url : str
    vid_id : str

    local_path : Optional[str]
    video_metadata : Dict[str,Any]
    transcript : Optional[str]
    ocr_test : List[str]

    compliance_results : Annotated[List[ComplianceIssue], operator.add]

    final_status : str
    final_report : str

    errors : Annotated[List[str], operator.add]