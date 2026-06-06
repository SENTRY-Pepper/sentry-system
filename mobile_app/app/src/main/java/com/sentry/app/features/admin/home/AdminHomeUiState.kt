package com.sentry.app.features.admin.home

import com.sentry.app.data.models.response.SessionSummary

data class AdminHomeUiState(
    val loading: Boolean = false,
    val error: String = "",
    // from OrganisationAnalytics
    val totalSessions: Int = 0,
    val completedSessions: Int = 0,
    val meanPreScore: Float = 0f,
    val meanPostScore: Float = 0f,
    val meanKnowledgeGain: Float = 0f,
    val meanGroundingAccuracy: Float = 0f,
    // from listSessions (5 most recent)
    val recentSessions: List<SessionSummary> = emptyList()
)