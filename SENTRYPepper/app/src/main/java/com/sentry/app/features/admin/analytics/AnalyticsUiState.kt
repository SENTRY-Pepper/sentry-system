package com.sentry.app.features.admin.analytics


import com.sentry.app.data.models.response.SessionSummary

data class AnalyticsUiState(
    val loading: Boolean = false,
    val error: String = "",
    val sessions: List<SessionSummary> = emptyList(),
    val selectedCondition: String? = null,
    val totalSessions: Int = 0,
    val groundedSessionCount: Int = 0,
    val baselineSessionCount: Int = 0,
    val groundedAccuracy: Float = 0f,
    val baselineAccuracy: Float = 0f,
    val groundedHallucination: Float = 0f,
    val baselineHallucination: Float = 0f,
    val groundingImprovement: Float = 0f,
    val groundedKnowledgeGain: Float = 0f,
    val baselineKnowledgeGain: Float = 0f
)
