package com.sentry.app.features.admin.home

import com.sentry.app.data.models.response.SessionSummary
import com.sentry.app.data.models.response.TraineeAnalytics

data class AdminHomeUiState(
    val loading: Boolean = false,
    val saving: Boolean = false,
    val error: String = "",
    val message: String = "",
    val totalSessions: Int = 0,
    val groundedSessions: Int = 0,
    val baselineSessions: Int = 0,
    val groundedAccuracy: Float = 0f,
    val baselineAccuracy: Float = 0f,
    val groundedHallucination: Float = 0f,
    val baselineHallucination: Float = 0f,
    val groundingImprovement: Float = 0f,
    val groundedKnowledgeGain: Float = 0f,
    val baselineKnowledgeGain: Float = 0f,
    val recentSessions: List<SessionSummary> = emptyList(),
    val trainees: List<TraineeAnalytics> = emptyList(),
)
