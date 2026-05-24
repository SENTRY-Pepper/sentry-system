package com.sentry.app.repository

import com.sentry.app.models.request.*
import com.sentry.app.models.response.*
import com.sentry.app.network.api.SentryApiService
import com.sentry.app.utils.NetworkResult
import com.sentry.app.utils.safeApiCall
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import timber.log.Timber
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class SessionRepository @Inject constructor(private val api: SentryApiService) {

    suspend fun healthCheck(): NetworkResult<HealthResponse> =
        withContext(Dispatchers.IO) { safeApiCall { api.healthCheck() } }

    suspend fun groundedQuery(query: String, scenarioId: String? = null): NetworkResult<QueryResponse> =
        withContext(Dispatchers.IO) {
            Timber.d("groundedQuery: $query")
            safeApiCall { api.groundedQuery(QueryRequest(query = query, scenarioId = scenarioId)) }
        }

    suspend fun baselineQuery(query: String, scenarioId: String? = null): NetworkResult<QueryResponse> =
        withContext(Dispatchers.IO) {
            safeApiCall { api.baselineQuery(QueryRequest(query = query, scenarioId = scenarioId)) }
        }

    suspend fun startSession(participantId: String, condition: String, organisationId: String): NetworkResult<SessionStartResponse> =
        withContext(Dispatchers.IO) {
            safeApiCall { api.startSession(SessionStartRequest(participantId, condition, organisationId)) }
        }

    suspend fun endSession(sessionId: String, preScore: Float?, postScore: Float?, durationSeconds: Int?): NetworkResult<SessionEndResponse> =
        withContext(Dispatchers.IO) {
            safeApiCall { api.endSession(SessionEndRequest(sessionId, preScore, postScore, durationSeconds)) }
        }

    suspend fun getSession(sessionId: String): NetworkResult<SessionSummary> =
        withContext(Dispatchers.IO) { safeApiCall { api.getSession(sessionId) } }

    suspend fun logInteraction(
        sessionId: String, scenarioId: String, scenarioType: String,
        decision: String, employeeResponse: String?, responseTimeMs: Int?,
        correctionLoops: Int, aiLatencyMs: Float?, aiSources: String?,
    ): NetworkResult<InteractionLogResponse> =
        withContext(Dispatchers.IO) {
            safeApiCall {
                api.logInteraction(
                    InteractionLogRequest(sessionId, scenarioId, scenarioType,
                        decision, employeeResponse, responseTimeMs,
                        correctionLoops, aiLatencyMs, aiSources)
                )
            }
        }

    suspend fun getStudyAnalytics(): NetworkResult<StudyAnalytics> =
        withContext(Dispatchers.IO) { safeApiCall { api.getStudyAnalytics() } }

    suspend fun listSessions(condition: String? = null, organisationId: String? = null): NetworkResult<List<SessionSummary>> =
        withContext(Dispatchers.IO) { safeApiCall { api.listSessions(condition, organisationId) } }

    suspend fun getOrganisationAnalytics(organisationId: String): NetworkResult<OrganisationAnalytics> =
        withContext(Dispatchers.IO) { safeApiCall { api.getOrganisationAnalytics(organisationId) } }
}
