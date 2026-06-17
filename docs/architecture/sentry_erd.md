# SENTRY Entity Relationship Diagram

This ERD represents the current persisted backend schema defined in
`backend/database/models.py`. Android DTOs, RAG vector-store documents, and
local Android DataStore state are intentionally excluded because they are not
relational database tables in the current system.

```mermaid
erDiagram
    ORGANISATIONS {
        string id PK "UUID string"
        string name
        string canonical_id UK "Canonical organisation key"
        boolean is_active
        datetime created_at
    }

    USERS {
        string id PK "UUID string"
        string participant_id
        string display_name
        string role "admin | manager | trainee"
        string pin_hash
        string organisation_id FK "references organisations.canonical_id"
        string department
        string position
        boolean is_active
        datetime created_at
    }

    TRAINING_SESSIONS {
        string id PK "UUID string"
        string participant_id
        string user_id FK "nullable; references users.id"
        string condition "grounded | baseline"
        string organisation_id "logical canonical organisation key"
        datetime started_at
        datetime completed_at
        int duration_seconds
        float pre_assessment_score
        float post_assessment_score
        float knowledge_gain
        boolean is_complete
    }

    SCENARIO_INTERACTIONS {
        string id PK "UUID string"
        string session_id FK "references training_sessions.id"
        string scenario_id
        string scenario_type
        string decision "correct | risky"
        text employee_response
        int response_time_ms
        int correction_loops
        float ai_latency_ms
        text ai_sources "comma-separated source names"
        datetime created_at
    }

    ASSESSMENT_RESULTS {
        string id PK "UUID string"
        string session_id FK_UK "unique; references training_sessions.id"
        float pre_score
        float post_score
        float knowledge_gain
        float relative_improvement_pct
        datetime pre_taken_at
        datetime post_taken_at
    }

    EVALUATION_LOGS {
        string id PK "UUID string"
        string session_id FK "references training_sessions.id"
        string scenario_id
        text query
        string mode "grounded | baseline"
        text response
        float grounding_accuracy
        float hallucination_rate
        float grounding_improvement
        float retrieval_ms
        float generation_ms
        float total_ms
        int prompt_tokens
        int completion_tokens
        text sources "comma-separated source names"
        datetime logged_at
    }

    ORGANISATIONS ||--o{ USERS : "has active/inactive accounts"
    USERS ||--o{ TRAINING_SESSIONS : "owns optional persisted sessions"
    TRAINING_SESSIONS ||--o{ SCENARIO_INTERACTIONS : "records scenario decisions"
    TRAINING_SESSIONS ||--o{ EVALUATION_LOGS : "records RAG evaluation metrics"
    TRAINING_SESSIONS ||--o| ASSESSMENT_RESULTS : "has final assessment"
    ORGANISATIONS ||..o{ TRAINING_SESSIONS : "logical analytics scope"
```

## Relationship Notes

- `organisations.canonical_id` is the referenced organisation key for users,
  not `organisations.id`.
- `training_sessions.organisation_id` stores the same canonical organisation
  identifier for analytics, but it is not declared as a physical foreign key in
  SQLAlchemy.
- `training_sessions.user_id` is nullable and uses `ON DELETE SET NULL`, so
  historical sessions can survive account removal.
- `scenario_interactions`, `assessment_results`, and `evaluation_logs` use
  `ON DELETE CASCADE` from `training_sessions`.
- `assessment_results.session_id` is unique, giving each persisted training
  session at most one assessment result row.
- Scenario catalog data, departments, token/session auth state, RAG knowledge
  chunks, and AI source documents are not normalized into separate SQL tables in
  the current repository.
