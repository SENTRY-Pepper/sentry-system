package com.sentry.app.data.repository

import com.sentry.app.core.network.NetworkResult
import com.sentry.app.core.network.SentryKtorClient
import com.sentry.app.data.local.TokenManager
import com.sentry.app.data.models.request.InteractionLogRequest
import com.sentry.app.data.models.request.SessionEndRequest
import com.sentry.app.data.models.request.SessionStartRequest
import com.sentry.app.data.models.response.SessionEndResponse
import com.sentry.app.data.models.response.SessionStartResponse
import com.sentry.app.data.models.response.SessionSummary
import com.sentry.app.data.remote.api.endSession
import com.sentry.app.data.remote.api.getSession
import com.sentry.app.data.remote.api.logInteraction
import com.sentry.app.data.remote.api.startSession
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import timber.log.Timber
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class SessionRepositoryImpl @Inject constructor(
    private val ktorClient: SentryKtorClient,
    private val tokenManager: TokenManager,
) : SessionRepository {

    override suspend fun startSession(
        condition: String,
    ): NetworkResult<SessionStartResponse> {
        val participantId = tokenManager.getParticipantId()
            ?: return NetworkResult.Error(message = "Not authenticated")
        val organisationId = tokenManager.getOrganisationId() ?: "SENTRY_STUDY"

        return ktorClient.startSession(
            SessionStartRequest(
                participantId = participantId,
                condition = condition,
                organisationId = organisationId,
            )
        )
    }

    override suspend fun endSession(
        sessionId: String,
        preAssessmentScore: Float?,
        postAssessmentScore: Float?,
        durationSeconds: Int?,
    ): NetworkResult<SessionEndResponse> =
        ktorClient.endSession(
            SessionEndRequest(
                sessionId = sessionId,
                preAssessmentScore = preAssessmentScore,
                postAssessmentScore = postAssessmentScore,
                durationSeconds = durationSeconds,
            )
        )

    override suspend fun getSession(
        sessionId: String,
    ): NetworkResult<SessionSummary> =
        ktorClient.getSession(sessionId)

    // fire-and-forget — launched on IO, failure is logged but never surfaced to UI
    override fun logInteraction(
        sessionId: String,
        scenarioId: String,
        scenarioType: String,
        decision: String,
        employeeResponse: String?,
        responseTimeMs: Int?,
        correctionLoops: Int,
        aiLatencyMs: Float?,
        aiSources: String?,
    ) {
        CoroutineScope(Dispatchers.IO).launch {
            val result = ktorClient.logInteraction(
                InteractionLogRequest(
                    sessionId = sessionId,
                    scenarioId = scenarioId,
                    scenarioType = scenarioType,
                    decision = decision,
                    employeeResponse = employeeResponse,
                    responseTimeMs = responseTimeMs,
                    correctionLoops = correctionLoops,
                    aiLatencyMs = aiLatencyMs,
                    aiSources = aiSources,
                )
            )
            if (result.isError) {
                Timber.w("SessionRepository: logInteraction failed — non-fatal")
            }
        }
    }
}