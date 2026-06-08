package com.sentry.app.data.models.request

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class QueryRequest(
    @SerialName("query")           val query: String,
    @SerialName("scenario_id")     val scenarioId: String?    = null,
    @SerialName("doc_type_filter") val docTypeFilter: String? = null,
)

@Serializable
data class SessionStartRequest(
    @SerialName("participant_id")  val participantId: String,
    @SerialName("condition")       val condition: String,
    @SerialName("organisation_id") val organisationId: String,
)

@Serializable
data class SessionEndRequest(
    @SerialName("session_id")            val sessionId: String,
    @SerialName("pre_assessment_score")  val preAssessmentScore: Float?,
    @SerialName("post_assessment_score") val postAssessmentScore: Float?,
    @SerialName("duration_seconds")      val durationSeconds: Int?,
)

@Serializable
data class InteractionLogRequest(
    @SerialName("session_id")        val sessionId: String,
    @SerialName("scenario_id")       val scenarioId: String,
    @SerialName("scenario_type")     val scenarioType: String,
    @SerialName("decision")          val decision: String,
    @SerialName("employee_response") val employeeResponse: String?,
    @SerialName("response_time_ms")  val responseTimeMs: Int?,
    @SerialName("correction_loops")  val correctionLoops: Int = 0,
    @SerialName("ai_latency_ms")     val aiLatencyMs: Float?,
    @SerialName("ai_sources")        val aiSources: String?,
)

@Serializable
data class UserLoginRequest(
    @SerialName("participant_id")  val participantId: String,
    @SerialName("pin")             val pin: String,
    @SerialName("role")            val role: String,
    @SerialName("organisation_id") val organisationId: String?,
)

@Serializable
data class UserCreateRequest(
    @SerialName("participant_id")  val participantId: String,
    @SerialName("display_name")    val displayName: String,
    @SerialName("role")            val role: String = "trainee",
    @SerialName("pin")             val pin: String,
    @SerialName("organisation_id") val organisationId: String,
    @SerialName("department")      val department: String?,
    @SerialName("position")        val position: String?,
)
