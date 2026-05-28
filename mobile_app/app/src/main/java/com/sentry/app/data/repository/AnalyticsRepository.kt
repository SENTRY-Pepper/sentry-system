package com.sentry.app.data.repository

import com.sentry.app.core.network.NetworkResult
import com.sentry.app.core.network.safeApiCall
import com.sentry.app.data.models.response.*
import com.sentry.app.data.remote.api.SentryApiService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton

// Admin-only repository — AnalyticsViewModel calls requireAdmin() before using this
@Singleton
class AnalyticsRepository @Inject constructor(private val api: SentryApiService) {

    suspend fun getStudyAnalytics(): NetworkResult<StudyAnalytics> =
        io { safeApiCall("Analytics.study") { api.getStudyAnalytics() } }

    suspend fun listSessions(
        condition: String? = null, organisationId: String? = null,
    ): NetworkResult<List<SessionSummary>> =
        io { safeApiCall("Analytics.sessions") { api.listSessions(condition, organisationId) } }

    suspend fun getOrganisationAnalytics(
        organisationId: String,
    ): NetworkResult<OrganisationAnalytics> =
        io { safeApiCall("Analytics.org") { api.getOrganisationAnalytics(organisationId) } }

    private suspend fun <T> io(block: suspend () -> T): T =
        withContext(Dispatchers.IO) { block() }
}