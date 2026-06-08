package com.sentry.app.features.manager.home

import com.sentry.app.data.models.response.DepartmentAnalytics
import com.sentry.app.data.models.response.TraineeAnalytics
import com.sentry.app.data.models.response.WeaknessAnalytics

data class ManagerHomeUiState(
    val loading: Boolean = false,
    val saving: Boolean = false,
    val error: String = "",
    val message: String = "",
    val organisationId: String = "",
    val traineeCount: Int = 0,
    val activeTrainees: Int = 0,
    val totalSessions: Int = 0,
    val completedSessions: Int = 0,
    val averageScore: Float = 0f,
    val completionRate: Float = 0f,
    val riskyAnswers: Int = 0,
    val weaknesses: List<WeaknessAnalytics> = emptyList(),
    val departments: List<DepartmentAnalytics> = emptyList(),
    val trainees: List<TraineeAnalytics> = emptyList(),
)
