package com.sentry.app.data.models.response

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class HealthResponse(
    @SerialName("status")         val status: String,
    @SerialName("pipeline_ready") val pipelineReady: Boolean,
)

@Serializable
data class QueryResponse(
    @SerialName("query")             val query: String,
    @SerialName("mode")              val mode: String,
    @SerialName("response")          val response: String,
    @SerialName("sources")           val sources: List<String>,
    @SerialName("chunks_used")       val chunksUsed: Int,
    @SerialName("retrieval_ms")      val retrievalMs: Double,
    @SerialName("generation_ms")     val generationMs: Double,
    @SerialName("total_ms")          val totalMs: Double,
    @SerialName("scenario_id")       val scenarioId: String?,
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
    @SerialName("decision")       val decision: String,
    @SerialName("logged")         val logged: Boolean,
)

@Serializable
data class StudyAnalytics(
    @SerialName("total_sessions")                   val totalSessions: Int,
    @SerialName("grounded_sessions")                val groundedSessions: Int,
    @SerialName("baseline_sessions")                val baselineSessions: Int,
    @SerialName("mean_grounding_accuracy_grounded") val meanGroundingAccuracyGrounded: Float?,
    @SerialName("mean_grounding_accuracy_baseline") val meanGroundingAccuracyBaseline: Float?,
    @SerialName("mean_hallucination_rate_grounded") val meanHallucinationRateGrounded: Float?,
    @SerialName("mean_hallucination_rate_baseline") val meanHallucinationRateBaseline: Float?,
    @SerialName("mean_grounding_improvement")       val meanGroundingImprovement: Float?,
    @SerialName("mean_knowledge_gain_grounded")     val meanKnowledgeGainGrounded: Float?,
    @SerialName("mean_knowledge_gain_baseline")     val meanKnowledgeGainBaseline: Float?,
)

@Serializable
data class OrganisationAnalytics(
    @SerialName("organisation_id")         val organisationId: String,
    @SerialName("total_sessions")          val totalSessions: Int,
    @SerialName("completed_sessions")      val completedSessions: Int,
    @SerialName("mean_pre_score")          val meanPreScore: Float?,
    @SerialName("mean_post_score")         val meanPostScore: Float?,
    @SerialName("mean_knowledge_gain")     val meanKnowledgeGain: Float?,
    @SerialName("mean_grounding_accuracy") val meanGroundingAccuracy: Float?,
)

@Serializable
data class UserResponse(
    @SerialName("id")              val id: String,
    @SerialName("participant_id")  val participantId: String,
    @SerialName("display_name")    val displayName: String,
    @SerialName("role")            val role: String,
    @SerialName("organisation_id") val organisationId: String?,
    @SerialName("department")      val department: String?,
    @SerialName("position")        val position: String?,
    @SerialName("is_active")       val isActive: Boolean,
    @SerialName("created_at")      val createdAt: String,
)

@Serializable
data class UserLoginResponse(
    @SerialName("token") val token: String,
    @SerialName("user")  val user: UserResponse,
)

@Serializable
data class TraineeAnalytics(
    @SerialName("user_id")            val userId: String?,
    @SerialName("participant_id")     val participantId: String,
    @SerialName("display_name")       val displayName: String,
    @SerialName("department")         val department: String?,
    @SerialName("position")           val position: String?,
    @SerialName("is_active")          val isActive: Boolean,
    @SerialName("session_count")      val sessionCount: Int,
    @SerialName("completed_sessions") val completedSessions: Int,
    @SerialName("average_score")      val averageScore: Float?,
    @SerialName("best_score")         val bestScore: Float?,
    @SerialName("last_score")         val lastScore: Float?,
    @SerialName("last_session_at")    val lastSessionAt: String?,
    @SerialName("risky_answers")      val riskyAnswers: Int,
    @SerialName("weakest_categories") val weakestCategories: List<String>,
)

@Serializable
data class WeaknessAnalytics(
    @SerialName("scenario_id")     val scenarioId: String,
    @SerialName("scenario_type")   val scenarioType: String,
    @SerialName("risky_answers")   val riskyAnswers: Int,
    @SerialName("correct_answers") val correctAnswers: Int,
    @SerialName("total_answers")   val totalAnswers: Int,
    @SerialName("risk_rate")       val riskRate: Float,
)

@Serializable
data class DepartmentAnalytics(
    @SerialName("department")         val department: String,
    @SerialName("trainee_count")      val traineeCount: Int,
    @SerialName("completed_sessions") val completedSessions: Int,
    @SerialName("average_score")      val averageScore: Float?,
    @SerialName("risky_answers")      val riskyAnswers: Int,
)

@Serializable
data class ManagerOverviewAnalytics(
    @SerialName("organisation_id")     val organisationId: String,
    @SerialName("trainee_count")       val traineeCount: Int,
    @SerialName("active_trainees")     val activeTrainees: Int,
    @SerialName("total_sessions")      val totalSessions: Int,
    @SerialName("completed_sessions")  val completedSessions: Int,
    @SerialName("average_score")       val averageScore: Float?,
    @SerialName("completion_rate")     val completionRate: Float,
    @SerialName("risky_answers")       val riskyAnswers: Int,
    @SerialName("top_weaknesses")      val topWeaknesses: List<WeaknessAnalytics>,
    @SerialName("departments")         val departments: List<DepartmentAnalytics>,
)

// Local only — never serialized to network
data class AuthState(
    val isAuthenticated: Boolean = false,
    val participantId: String    = "",
    val role: String             = "trainee",
    val organisationId: String   = "",
)
