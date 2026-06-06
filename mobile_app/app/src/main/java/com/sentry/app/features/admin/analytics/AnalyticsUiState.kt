package com.sentry.app.features.admin.analytics


import com.sentry.app.data.models.response.SessionSummary

data class AnalyticsUiState(
    val loading: Boolean = false,
    val error: String = "",
    val sessions: List<SessionSummary> = emptyList(),
    val selectedCondition: String? = null,
    // computed from sessions list
    val avgAccuracy: Float = 0f,
    val avgKnowledgeGain: Float = 0f,
    val ragSessionCount: Int = 0,
    val baselineSessionCount: Int = 0,
    val ragAvgAccuracy: Float = 0f,
    val baselineAvgAccuracy: Float = 0f
)