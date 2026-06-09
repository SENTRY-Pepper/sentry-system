package com.sentry.app.data.repository

import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.models.request.UserCreateRequest
import com.sentry.app.data.models.response.ManagerOverviewAnalytics
import com.sentry.app.data.models.response.OrganisationAnalytics
import com.sentry.app.data.models.response.SessionSummary
import com.sentry.app.data.models.response.StudyAnalytics
import com.sentry.app.data.models.response.TraineeAnalytics
import com.sentry.app.data.models.response.UserResponse
import com.sentry.app.data.models.response.WeaknessAnalytics

interface AnalyticsRepository {

    suspend fun getStudyAnalytics(): NetworkResult<StudyAnalytics>

    suspend fun getOrganisationAnalytics(): NetworkResult<OrganisationAnalytics>

    suspend fun listSessions(
        condition: String? = null,
        organisationId: String? = null,
        limit: Int = 50,
    ): NetworkResult<List<SessionSummary>>

    suspend fun getManagerOverview(): NetworkResult<ManagerOverviewAnalytics>

    suspend fun getManagerWeaknesses(limit: Int = 10): NetworkResult<List<WeaknessAnalytics>>

    suspend fun listTrainees(): NetworkResult<List<TraineeAnalytics>>

    suspend fun createTrainee(request: UserCreateRequest): NetworkResult<UserResponse>

    suspend fun deactivateTrainee(userId: String): NetworkResult<UserResponse>
}

