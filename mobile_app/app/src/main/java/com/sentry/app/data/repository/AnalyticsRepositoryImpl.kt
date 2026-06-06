package com.sentry.app.data.repository

import com.sentry.app.core.network.NetworkResult
import com.sentry.app.core.network.SentryKtorClient
import com.sentry.app.data.local.TokenManager
import com.sentry.app.data.models.response.OrganisationAnalytics
import com.sentry.app.data.models.response.SessionSummary
import com.sentry.app.data.models.response.StudyAnalytics
import com.sentry.app.data.remote.api.getOrganisationAnalytics
import com.sentry.app.data.remote.api.getStudyAnalytics
import com.sentry.app.data.remote.api.listSessions
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class AnalyticsRepositoryImpl @Inject constructor(
    private val ktorClient: SentryKtorClient,
    private val tokenManager: TokenManager,
) : AnalyticsRepository {

    override suspend fun getStudyAnalytics(): NetworkResult<StudyAnalytics> =
        ktorClient.getStudyAnalytics()

    override suspend fun getOrganisationAnalytics(): NetworkResult<OrganisationAnalytics> {
        val orgId = tokenManager.getOrganisationId()
            ?: return NetworkResult.Error(message = "No organisation ID")
        return ktorClient.getOrganisationAnalytics(orgId)
    }

    override suspend fun listSessions(
        condition: String?,
        organisationId: String?,
        limit: Int,
    ): NetworkResult<List<SessionSummary>> {
        val orgId = organisationId ?: tokenManager.getOrganisationId()
        return ktorClient.listSessions(
            condition = condition,
            organisationId = orgId,
            limit = limit,
        )
    }
}