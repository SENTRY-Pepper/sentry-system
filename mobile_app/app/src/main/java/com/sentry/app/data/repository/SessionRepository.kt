package com.sentry.app.data.repository

import com.sentry.app.core.network.NetworkResult
import com.sentry.app.core.network.safeApiCall
import com.sentry.app.data.models.request.*
import com.sentry.app.data.models.response.*
import com.sentry.app.data.remote.api.SentryApiService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class SessionRepository @Inject constructor(private val api: SentryApiService) {

    suspend fun startSession(
        participantId: String, condition: String, organisationId: String,
    ): NetworkResult<SessionStartResponse> = io {
        safeApiCall("Session.start") {
            api.startSession(SessionStartRequest(participantId, condition, organisationId))
        }
    }

    suspend fun endSession(
        sessionId: String, preScore: Float?, postScore: Float?, durationSeconds: Int?,
    ): NetworkResult<SessionEndResponse> = io {
        safeApiCall("Session.end") {
            api.endSession(SessionEndRequest(sessionId, preScore, postScore, durationSeconds))
        }
    }

    suspend fun getSession(sessionId: String): NetworkResult<SessionSummary> = io {
        safeApiCall("Session.get") { api.getSession(sessionId) }
    }

    // Fire-and-forget — non-fatal if it fails
    suspend fun logInteraction(
        sessionId: String, scenarioId: String, scenarioType: String,
        decision: String, employeeResponse: String?, responseTimeMs: Int?,
        correctionLoops: Int = 0, aiLatencyMs: Float?, aiSources: String?,
    ): NetworkResult<InteractionLogResponse> = io {
        safeApiCall("Session.log") {
            api.logInteraction(
                InteractionLogRequest(
                    sessionId, scenarioId, scenarioType,
                    decision, employeeResponse, responseTimeMs,
                    correctionLoops, aiLatencyMs, aiSources,
                )
            )
        }
    }

    private suspend fun <T> io(block: suspend () -> T): T =
        withContext(Dispatchers.IO) { block() }
}