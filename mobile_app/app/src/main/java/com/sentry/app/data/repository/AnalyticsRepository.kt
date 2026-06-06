package com.sentry.app.data.repository

import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.models.response.OrganisationAnalytics
import com.sentry.app.data.models.response.SessionSummary
import com.sentry.app.data.models.response.StudyAnalytics

interface AnalyticsRepository {

    suspend fun getStudyAnalytics(): NetworkResult<StudyAnalytics>

    suspend fun getOrganisationAnalytics(): NetworkResult<OrganisationAnalytics>

    suspend fun listSessions(
        condition: String? = null,
        organisationId: String? = null,
        limit: Int = 50,
    ): NetworkResult<List<SessionSummary>>
}

