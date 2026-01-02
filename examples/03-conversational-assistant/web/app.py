# web/app.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import asyncio

from assistant.graph.assistant import AIAssistant

app = FastAPI(title="AI Assistant API")
assistant = AIAssistant()
assistant.compile()


# 연결된 WebSocket 관리
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)


manager = ConnectionManager()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool


# REST API 엔드포인트
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """채팅 API"""
    try:
        response = await assistant.achat(
            request.message,
            session_id=request.session_id
        )
        return ChatResponse(
            response=response,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/approve")
async def approve_endpoint(request: ApprovalRequest):
    """도구 실행 승인 API"""
    assistant.approve_tool_execution(
        request.session_id,
        request.approved
    )
    return {"status": "approved" if request.approved else "rejected"}


@app.get("/api/history/{session_id}")
async def history_endpoint(session_id: str):
    """대화 기록 API"""
    history = assistant.get_conversation_history(session_id)
    return {"history": history}


@app.delete("/api/history/{session_id}")
async def clear_history_endpoint(session_id: str):
    """대화 기록 삭제 API"""
    assistant.clear_conversation(session_id)
    return {"status": "cleared"}


# WebSocket 엔드포인트
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket 채팅"""
    await manager.connect(websocket, session_id)

    try:
        while True:
            # 메시지 수신
            data = await websocket.receive_text()
            message_data = json.loads(data)

            if message_data.get("type") == "chat":
                user_message = message_data.get("message", "")

                # 스트리밍 응답
                async for event in stream_response(user_message, session_id):
                    await manager.send_message(event, session_id)

            elif message_data.get("type") == "approve":
                approved = message_data.get("approved", False)
                assistant.approve_tool_execution(session_id, approved)
                await manager.send_message(
                    {"type": "approval_processed", "approved": approved},
                    session_id
                )

    except WebSocketDisconnect:
        manager.disconnect(session_id)


async def stream_response(message: str, session_id: str):
    """스트리밍 응답 생성"""
    for event in assistant.stream_chat(message, session_id):
        yield event
        await asyncio.sleep(0.01)  # 약간의 지연

    yield {"type": "done"}


# HTML 페이지
@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지"""
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Assistant</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .chat-container {
            width: 100%;
            max-width: 800px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .chat-messages {
            height: 500px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
            flex-direction: column;
        }
        .message.user { align-items: flex-end; }
        .message.assistant { align-items: flex-start; }
        .message-content {
            max-width: 80%;
            padding: 12px 18px;
            border-radius: 18px;
            line-height: 1.5;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .message.assistant .message-content {
            background: white;
            color: #333;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .chat-input {
            display: flex;
            padding: 20px;
            background: white;
            border-top: 1px solid #eee;
        }
        .chat-input input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #eee;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
        }
        .chat-input input:focus { border-color: #667eea; }
        .chat-input button {
            margin-left: 10px;
            padding: 15px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
        }
        .typing-indicator { display: none; padding: 10px; }
        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #667eea;
            border-radius: 50%;
            margin: 0 2px;
            animation: bounce 1.4s infinite ease-in-out;
        }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🤖 AI Assistant</h1>
            <p>무엇이든 물어보세요!</p>
        </div>
        <div class="chat-messages" id="messages">
            <div class="message assistant">
                <div class="message-content">
                    안녕하세요! 저는 AI 어시스턴트입니다. 웹 검색, 계산, 코드 실행 등 다양한 작업을 도와드릴 수 있습니다.
                </div>
            </div>
        </div>
        <div class="typing-indicator" id="typing">
            <span></span><span></span><span></span>
        </div>
        <div class="chat-input">
            <input type="text" id="input" placeholder="메시지를 입력하세요..." autocomplete="off">
            <button onclick="sendMessage()">전송</button>
        </div>
    </div>
    <script>
        const sessionId = 'session-' + Math.random().toString(36).substr(2, 9);
        const messagesDiv = document.getElementById('messages');
        const input = document.getElementById('input');
        const typing = document.getElementById('typing');

        const ws = new WebSocket(`ws://${window.location.host}/ws/${sessionId}`);

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === 'message') {
                typing.style.display = 'none';
                addMessage(data.content, 'assistant');
            } else if (data.type === 'done') {
                typing.style.display = 'none';
            }
        };

        function sendMessage() {
            const message = input.value.trim();
            if (!message) return;
            addMessage(message, 'user');
            input.value = '';
            typing.style.display = 'block';
            ws.send(JSON.stringify({ type: 'chat', message: message }));
        }

        function addMessage(content, type) {
            const div = document.createElement('div');
            div.className = `message ${type}`;
            div.innerHTML = `<div class="message-content">${content.replace(/\\n/g, '<br>')}</div>`;
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        input.addEventListener('keypress', e => { if (e.key === 'Enter') sendMessage(); });
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
