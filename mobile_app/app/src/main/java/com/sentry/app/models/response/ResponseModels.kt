package com.sentry.app.models.response

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class HealthResponse(
    @SerialName("status")         val status: String,
    @SerialName("pipeline_ready") val pipelineReady: Boolean,
    @SerialName("knowledge_base") val knowledgeBase: KnowledgeBaseStats,
)

@Serializable
data class KnowledgeBaseStats(
    @SerialName("collection_name") val collectionName: String,
    @SerialName("total_chunks")    val totalChunks: Int,
    @SerialName("top_k_default")   val topKDefault: Int,
)

@Serializable
data class QueryResponse(
    @SerialName("query")             val query: String,
    @SerialName("mode")              val mode: String,
    @SerialName("response")          val response: String,
    @SerialName("sources")           val sources: List<String>,
    @SerialName("retrieved_chunks")  val retrievedChunks: List<RetrievedChunk>,
    @SerialName("chunks_used")       val chunksUsed: Int,
    @SerialName("context_tokens")    val contextTokens: Int,
    @SerialName("prompt_tokens")     val promptTokens: Int,
    @SerialName("completion_tokens") val completionTokens: Int,
    @SerialName("retrieval_ms")      val retrievalMs: Double,
    @SerialName("generation_ms")     val generationMs: Double,
    @SerialName("total_ms")          val totalMs: Double,
    @SerialName("model")             val model: String,
    @SerialName("scenario_id")       val scenarioId: String?,
)

@Serializable
data class RetrievedChunk(
    @SerialName("source")      val source: String,
    @SerialName("doc_type")    val docType: String,
    @SerialName("score")       val score: Double,
    @SerialName("chunk_index") val chunkIndex: Int,
)

@Serializable
data class SessionStartResponse(
    @SerialName("session_id")     val sessionId: String,
    @SerialName("participant_id") val participantId: String,
    @SerialName("condition")      val condition: String,
    @SerialName("started_at")     val startedAt: String,
)

@Serializable
data class SessionEndResponse(
    @SerialName("session_id")               val sessionId: String,
    @SerialName("participant_id")           val participantId: String,
    @SerialName("duration_seconds")         val durationSeconds: Int?,
    @SerialName("pre_assessment_score")     val preAssessmentScore: Float?,
    @SerialName("post_assessment_score")    val postAssessmentScore: Float?,
    @SerialName("knowledge_gain")           val knowledgeGain: Float?,
    @SerialName("relative_improvement_pct") val relativeImprovementPct: Float?,
    @SerialName("is_complete")              val isComplete: Boolean,
)

@Serializable
data class SessionSummary(
    @SerialName("session_id")            val sessionId: String,
    @SerialName("participant_id")        val participantId: String,
    @SerialName("condition")             val condition: String,
    @SerialName("is_complete")           val isComplete: Boolean,
    @SerialName("duration_seconds")      val durationSeconds: Int?,
    @SerialName("pre_assessment_score")  val preAssessmentScore: Float?,
    @SerialName("post_assessment_score") val postAssessmentScore: Float?,
    @SerialName("knowledge_gain")        val knowledgeGain: Float?,
    @SerialName("started_at")            val startedAt: String,
)

@Serializable
data class InteractionLogResponse(
    @SerialName("interaction_id") val interactionId: String,
    @SerialName("session_id")     val sessionId: String,
    @SerialName("scenario_id")    val scenarioId: String,
    @SerialName("decision")       val decision: String,
    @SerialName("logged")         val logged: Boolean,
)

@Serializable
data class StudyAnalytics(
    @SerialName("total_sessions")                    val totalSessions: Int,
    @SerialName("grounded_sessions")                 val groundedSessions: Int,
    @SerialName("baseline_sessions")                 val baselineSessions: Int,
    @SerialName("mean_grounding_accuracy_grounded")  val meanGroundingAccuracyGrounded: Float?,
    @SerialName("mean_grounding_accuracy_baseline")  val meanGroundingAccuracyBaseline: Float?,
    @SerialName("mean_hallucination_rate_grounded")  val meanHallucinationRateGrounded: Float?,
    @SerialName("mean_hallucination_rate_baseline")  val meanHallucinationRateBaseline: Float?,
    @SerialName("mean_grounding_improvement")        val meanGroundingImprovement: Float?,
    @SerialName("mean_knowledge_gain_grounded")      val meanKnowledgeGainGrounded: Float?,
    @SerialName("mean_knowledge_gain_baseline")      val meanKnowledgeGainBaseline: Float?,
)

@Serializable
data class OrganisationAnalytics(
    @SerialName("organisation_id")        val organisationId: String,
    @SerialName("total_sessions")         val totalSessions: Int,
    @SerialName("completed_sessions")     val completedSessions: Int,
    @SerialName("mean_pre_score")         val meanPreScore: Float?,
    @SerialName("mean_post_score")        val meanPostScore: Float?,
    @SerialName("mean_knowledge_gain")    val meanKnowledgeGain: Float?,
    @SerialName("mean_grounding_accuracy") val meanGroundingAccuracy: Float?,
    @SerialName("mean_hallucination_rate") val meanHallucinationRate: Float?,
)

data class AuthState(
    val isAuthenticated: Boolean = false,
    val participantId: String    = "",
    val role: String             = "trainee",
    val organisationId: String   = "",
    val token: String            = "",
)
