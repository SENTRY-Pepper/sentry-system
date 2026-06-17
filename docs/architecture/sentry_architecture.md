# SENTRY Architecture Diagram

This is the simplified Mermaid source diagram for the current SENTRY system.
It shows the main runtime boundaries and data flow without internal route,
class, or table-level detail.

```mermaid
flowchart LR
    subgraph clients["Client Interfaces"]
        android["Android App<br/>Trainee, Manager, Admin"]
        pepper["Pepper Interfaces<br/>SENTRYPepper + NAOqi client"]
    end

    subgraph middleware["FastAPI Middleware"]
        api["REST API<br/>Auth, sessions, queries, analytics"]
    end

    subgraph ai["AI Layer"]
        rag["RAG Pipeline<br/>Grounded and baseline responses"]
        openai["OpenAI LLM"]
    end

    subgraph knowledge["Knowledge Base"]
        raw["OWASP + Legal Documents"]
        chroma["ChromaDB Vector Store"]
    end

    subgraph storage["Persistence"]
        postgres["PostgreSQL<br/>Users, organisations, sessions, analytics logs"]
        local["Android DataStore<br/>Auth state and local training progress"]
    end

    subgraph evaluation["Research Evaluation"]
        eval["Evaluation Scripts<br/>Grounding, hallucination, study metrics"]
    end

    android -->|"HTTP / Ktor"| api
    pepper -->|"HTTP"| api
    android -->|"local progress"| local

    api -->|"session and analytics writes"| postgres
    api -->|"grounded or baseline query"| rag
    api -->|"analytics reads"| postgres

    raw -->|"ingestion and embeddings"| chroma
    rag -->|"retrieve grounded context"| chroma
    rag -->|"generate answer"| openai

    eval -->|"runs study sessions"| api
    eval -->|"stores scored results"| postgres

    classDef client fill:#E8F3FF,stroke:#2F6FA3,color:#14324A
    classDef apiLayer fill:#EAF8EF,stroke:#2D8A57,color:#163D28
    classDef aiLayer fill:#FFF3D6,stroke:#B77D00,color:#4F3500
    classDef data fill:#FDECEC,stroke:#B84A4A,color:#4A1E1E
    classDef evalLayer fill:#EEF2F6,stroke:#61758A,color:#24313D

    class android,pepper client
    class api apiLayer
    class rag,openai aiLayer
    class raw,chroma,postgres,local data
    class eval evalLayer
```

## Notes

- Clients interact with SENTRY through the FastAPI middleware.
- The middleware owns authentication, session lifecycle, query handling, and
  analytics endpoints.
- Grounded answers use ChromaDB retrieval plus OpenAI generation. Baseline
  answers skip retrieval and use the LLM directly.
- PostgreSQL stores backend analytics and research data. Android DataStore keeps
  local app state and offline training progress.
