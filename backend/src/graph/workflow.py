'''
This module defines the DAG(Directed Acyclic Graph) that orchestrastes the video compliance audit process.
It connects the nodes using the StateGraph from LangGraph

START -> index_video_node -> audit_content_node -> END
'''

from langgraph.graph import StateGraph, END
from backend.src.graph.state import VideoAuditState

from backend.src.graph.nodes import (
    index_video_node,
    audit_content_node
)

def create_graph():
    workflow = StateGraph(VideoAuditState)

    #nodes
    workflow.add_node("indexer", index_video_node)
    workflow.add_node("auditor", audit_content_node)

    workflow.set_entry_point("indexer")

    #edges
    workflow.add_edge("indexer","auditor")
    workflow.add_edge("auditor",END)

    #compile
    app = workflow.compile()

    return app

app = create_graph()