#!/usr/bin/env python3
"""
Chat Interaction Routes - Text-based chat with cognitive architecture
Implements chat endpoints with provenance, session management, and safety filters
"""

import logging
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.cognitive_architecture import CognitiveArchitecture

router = APIRouter()
logger = logging.getLogger(__name__)

# Global cognitive architecture instance (will be initialized in main.py)
_cognitive_architecture: Optional[CognitiveArchitecture] = None

# Session management
_chat_sessions: Dict[str, Dict[str, Any]] = defaultdict(dict)


def get_cognitive_architecture() -> CognitiveArchitecture:
    """Dependency injection for cognitive architecture"""
    global _cognitive_architecture
    if _cognitive_architecture is None:
        raise HTTPException(status_code=503, detail="Cognitive architecture not initialized")
    return _cognitive_architecture


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""

    message: str = Field(..., description="User message text", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(
        None, description="Chat session ID (auto-generated if not provided)"
    )
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for processing")


class ProvenanceInfo(BaseModel):
    """Provenance information for chat response"""

    episode_ids: List[str] = Field(default_factory=list, description="Related episode trace IDs")
    semantic_links: List[str] = Field(
        default_factory=list, description="Related semantic memory IDs"
    )
    similar_traces_count: int = Field(0, description="Number of similar traces found")
    consolidation_occurred: bool = Field(False, description="Whether consolidation occurred")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""

    success: bool = Field(True, description="Whether the request was successful")
    response_text: str = Field(..., description="Generated response text")
    session_id: str = Field(..., description="Chat session ID")
    provenance: ProvenanceInfo = Field(..., description="Provenance information")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class SessionContextResponse(BaseModel):
    """Response model for session context endpoint"""

    session_id: str
    message_count: int
    working_memory_items: int
    created_at: datetime
    last_activity: datetime


class SafetyFilter:
    """Safety filter for PII and NSFW content (stub implementation)"""

    # Basic PII patterns (stub - should be more comprehensive)
    PII_PATTERNS = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{16}\b",  # Credit card
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}-\d{3}-\d{4}\b",  # Phone
    ]

    # Basic NSFW patterns (stub - should be more comprehensive)
    NSFW_PATTERNS = [
        # Add patterns as needed
    ]

    @classmethod
    def filter_content(cls, text: str) -> tuple[str, List[str]]:
        """
        Filter content for PII and NSFW

        Returns:
            tuple: (filtered_text, detected_issues)
        """
        detected_issues = []
        filtered_text = text

        # Check for PII
        for pattern in cls.PII_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                detected_issues.append(f"PII detected: {pattern}")
                # Replace with placeholder
                filtered_text = re.sub(pattern, "[REDACTED]", filtered_text)

        # Check for NSFW (stub)
        for pattern in cls.NSFW_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                detected_issues.append("NSFW content detected")
                # Could filter or flag here

        return filtered_text, detected_issues


def _generate_response_text(cognitive_result: Dict[str, Any], user_message: str) -> str:
    """
    Generate response text from cognitive architecture result

    This is a simple response generator. In a production system,
    this would use more sophisticated NLP techniques.
    """
    decision = cognitive_result.get("arbitration_decision", "process")

    if decision == "recall":
        recall_count = cognitive_result.get("recall_count", 0)
        if recall_count > 0:
            return f"I recall {recall_count} similar experiences. How can I help you with '{user_message}'?"
        else:
            return f"I'm processing your message: '{user_message}'. How can I assist you?"

    elif decision == "consolidate":
        consolidation = cognitive_result.get("consolidation_result", {})
        if consolidation.get("consolidated"):
            concept = consolidation.get("concept", "information")
            return f"I've consolidated knowledge about '{concept}'. What would you like to know?"
        else:
            return f"I'm learning from your message: '{user_message}'. What else can I help with?"

    else:
        # Default response
        return f"I understand you said: '{user_message}'. How can I assist you further?"


def _extract_provenance(cognitive_result: Dict[str, Any]) -> ProvenanceInfo:
    """Extract provenance information from cognitive architecture result"""
    episode_ids = []
    semantic_links = []

    # Extract episode trace ID
    trace_id = cognitive_result.get("episodic_trace_id")
    if trace_id:
        episode_ids.append(trace_id)

    # Extract similar traces from recall
    similar_traces = cognitive_result.get("similar_traces", [])
    if isinstance(similar_traces, list):
        for trace in similar_traces:
            if isinstance(trace, dict) and "trace_id" in trace:
                episode_ids.append(trace["trace_id"])
            elif isinstance(trace, str):
                episode_ids.append(trace)

    # Extract semantic links from consolidation
    consolidation_result = cognitive_result.get("consolidation_result", {})
    if consolidation_result.get("consolidated"):
        semantic_id = consolidation_result.get("semantic_id")
        if semantic_id:
            semantic_links.append(semantic_id)

    return ProvenanceInfo(
        episode_ids=episode_ids,
        semantic_links=semantic_links,
        similar_traces_count=len(similar_traces) if isinstance(similar_traces, list) else 0,
        consolidation_occurred=consolidation_result.get("consolidated", False),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, architecture: CognitiveArchitecture = Depends(get_cognitive_architecture)
) -> ChatResponse:
    """
    Chat endpoint - Process user message through cognitive architecture

    Returns response with provenance information including episode IDs and semantic links.
    """
    start_time = time.time()

    try:
        # Generate or use session ID
        session_id = request.session_id or str(uuid.uuid4())

        # Safety filter
        filtered_message, safety_issues = SafetyFilter.filter_content(request.message)
        if safety_issues:
            logger.warning(f"Safety issues detected in message: {safety_issues}")

        # Prepare context
        context = request.context or {}
        context["session_id"] = session_id
        context["chat_message"] = True
        context["timestamp"] = datetime.now().isoformat()

        # Process through cognitive architecture
        cognitive_result = architecture.process_input(filtered_message, context=context)

        # Generate response text
        response_text = _generate_response_text(cognitive_result, filtered_message)

        # Extract provenance
        provenance = _extract_provenance(cognitive_result)

        # Update session
        if session_id not in _chat_sessions:
            _chat_sessions[session_id] = {
                "created_at": datetime.now(),
                "message_count": 0,
                "working_memory_ids": [],
            }

        _chat_sessions[session_id]["message_count"] += 1
        _chat_sessions[session_id]["last_activity"] = datetime.now()

        # Track working memory IDs for this session
        working_memory_id = cognitive_result.get("working_memory_id")
        if working_memory_id:
            if "working_memory_ids" not in _chat_sessions[session_id]:
                _chat_sessions[session_id]["working_memory_ids"] = []
            _chat_sessions[session_id]["working_memory_ids"].append(working_memory_id)

        processing_time = (time.time() - start_time) * 1000

        return ChatResponse(
            success=True,
            response_text=response_text,
            session_id=session_id,
            provenance=provenance,
            processing_time_ms=processing_time,
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.get("/chat/{session_id}/context", response_model=SessionContextResponse)
async def get_session_context(
    session_id: str, architecture: CognitiveArchitecture = Depends(get_cognitive_architecture)
) -> SessionContextResponse:
    """
    Get context for a chat session

    Returns session information including message count and working memory state.
    """
    if session_id not in _chat_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    session = _chat_sessions[session_id]

    # Get working memory item count
    working_memory_items = len(architecture.working_memory.items)

    return SessionContextResponse(
        session_id=session_id,
        message_count=session.get("message_count", 0),
        working_memory_items=working_memory_items,
        created_at=session.get("created_at", datetime.now()),
        last_activity=session.get("last_activity", datetime.now()),
    )


@router.post("/chat/{session_id}/reset")
async def reset_session(
    session_id: str, architecture: CognitiveArchitecture = Depends(get_cognitive_architecture)
) -> Dict[str, Any]:
    """
    Reset a chat session

    Clears working memory items associated with the session.
    """
    if session_id not in _chat_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    try:
        session = _chat_sessions[session_id]
        items_cleared = session.get("message_count", 0)

        # Clear working memory items associated with this session
        working_memory_ids = session.get("working_memory_ids", [])
        items_removed = 0
        for item_id in working_memory_ids:
            try:
                architecture.working_memory.remove_item(item_id)
                items_removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove working memory item {item_id}: {e}")

        # Reset session
        _chat_sessions[session_id] = {
            "created_at": datetime.now(),
            "message_count": 0,
            "working_memory_ids": [],
        }

        logger.info(
            f"Session {session_id} reset, cleared {items_cleared} messages, removed {items_removed} working memory items"
        )

        return {
            "success": True,
            "session_id": session_id,
            "messages_cleared": items_cleared,
            "working_memory_items_removed": items_removed,
            "message": "Session reset successfully",
        }

    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset session: {str(e)}")
