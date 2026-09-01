from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
import time

from agent import agente_sigem

# ══════════════════════════════════════════════════════════════
# MODELOS PYDANTIC
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    mensaje: str = Field(min_length=1)
    thread_id: str = Field(default="default_user", min_length=1)

class PasoAgente(BaseModel):
    agente: str
    accion: str
    icono: str

class ChatResponse(BaseModel):
    respuesta: str
    thread_id: str
    pasos: list[PasoAgente]
    tiempo_ms: float

# ══════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Agente de respuesta ciudadana - SIGEM",
    description="API del agente RAG (LangGraph + Ollama) que responde preguntas sobre procesos, formatos y normatividad del municipio de Marinilla, con base en el repositorio SIGEM.",
    version="0.1.0",
)


@app.get("/health", tags=["sistema"])
async def health_check():
    """Indica si el proceso del API está disponible."""
    return {"status": "ok"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def endpoint_chat(request: ChatRequest):
    """
    Chatea con el agente pasándole un thread_id para mantener la memoria conversacional.
    """
    inicio = time.time()
    try:
        config_graph = {"configurable": {"thread_id": request.thread_id}}
        
        resultado = agente_sigem.invoke(
            {"messages": [HumanMessage(content=request.mensaje)]},
            config=config_graph
        )
        
        respuesta = resultado["messages"][-1].content
        
        pasos = []
        for msg in resultado["messages"]:
            if type(msg).__name__ == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    pasos.append(PasoAgente(
                        agente="herramienta",
                        accion=f"Buscando en SIGEM: {tc.get('args', {}).get('consulta', '')}",
                        icono="🔎"
                    ))
        
        tiempo_ms = (time.time() - inicio) * 1000
        
        return ChatResponse(
            respuesta=respuesta,
            thread_id=request.thread_id,
            pasos=pasos,
            tiempo_ms=round(tiempo_ms, 1)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Iniciando API del agente SIGEM en http://localhost:8001")
    print("Swagger Docs: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)