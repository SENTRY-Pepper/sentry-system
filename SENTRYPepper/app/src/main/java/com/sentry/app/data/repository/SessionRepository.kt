package com.sentry.app.data.repository

import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.models.response.SessionEndResponse
import com.sentry.app.data.models.response.SessionStartResponse
import com.sentry.app.data.models.response.SessionSummary

interface SessionRepository {

    suspend fun startSession(
        condition: String,
    ): NetworkResult<SessionStartResponse>

    suspend fun endSession(
        sessionId: String,
        preAssessmentScore: Float?,
        postAssessmentScore: Float?,
        durationSeconds: Int?,
    ): NetworkResult<SessionEndResponse>

    suspend fun getSession(
        sessionId: String,
    ): NetworkResult<SessionSummary>

    // fire-and-forget — failures are non-fatal and must not block UI
    fun logInteraction(
        sessionId: String,
        scenarioId: String,
        scenarioType: String,
        decision: String,
        employeeResponse: String?,
        responseTimeMs: Int?,
        correctionLoops: Int,
        aiLatencyMs: Float?,
        aiSources: String?,
    )
}
