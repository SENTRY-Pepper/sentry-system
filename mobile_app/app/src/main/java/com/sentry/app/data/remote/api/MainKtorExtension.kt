package com.sentry.app.data.remote.api

import com.sentry.app.core.network.SentryKtorClient
import com.sentry.app.core.network.NetworkResult
import com.sentry.app.core.network.safeKtorCall
import com.sentry.app.data.models.request.InteractionLogRequest
import com.sentry.app.data.models.request.QueryRequest
import com.sentry.app.data.models.request.SessionEndRequest
import com.sentry.app.data.models.request.SessionStartRequest
import com.sentry.app.data.models.response.HealthResponse
import com.sentry.app.data.models.response.InteractionLogResponse
import com.sentry.app.data.models.response.OrganisationAnalytics
import com.sentry.app.data.models.response.QueryResponse
import com.sentry.app.data.models.response.SessionEndResponse
import com.sentry.app.data.models.response.SessionStartResponse
import com.sentry.app.data.models.response.SessionSummary
import com.sentry.app.data.models.response.StudyAnalytics
import io.ktor.client.call.body
import io.ktor.client.request.get
import io.ktor.client.request.parameter
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.http.encodeURLPathPart

// health

suspend fun SentryKtorClient.healthCheck(): NetworkResult<HealthResponse> =
    safeKtorCall("healthCheck") {
        client.get("health").body()
    }

// ai query

suspend fun SentryKtorClient.groundedQuery(
    request: QueryRequest,
): NetworkResult<QueryResponse> =
    safeKtorCall("groundedQuery") {
        client.post("api/v1/query") {
            setBody(request)
        }.body()
    }

suspend fun SentryKtorClient.baselineQuery(
    request: QueryRequest,
): NetworkResult<QueryResponse> =
    safeKtorCall("baselineQuery") {
        client.post("api/v1/query/baseline") {
            setBody(request)
        }.body()
    }

// session lifecycle

suspend fun SentryKtorClient.startSession(
    request: SessionStartRequest,
): NetworkResult<SessionStartResponse> =
    safeKtorCall("startSession") {
        client.post("api/v1/sessions/start") {
            setBody(request)
        }.body()
    }

suspend fun SentryKtorClient.endSession(
    request: SessionEndRequest,
): NetworkResult<SessionEndResponse> =
    safeKtorCall("endSession") {
        client.post("api/v1/sessions/end") {
            setBody(request)
        }.body()
    }

suspend fun SentryKtorClient.getSession(
    sessionId: String,
): NetworkResult<SessionSummary> =
    safeKtorCall("getSession") {
        client.get("api/v1/sessions/$sessionId").body()
    }

// logging — fire-and-forget, caller does not await result

suspend fun SentryKtorClient.logInteraction(
    request: InteractionLogRequest,
): NetworkResult<InteractionLogResponse> =
    safeKtorCall("logInteraction") {
        client.post("api/v1/sessions/interaction") {
            setBody(request)
        }.body()
    }

// analytics — admin only

suspend fun SentryKtorClient.getStudyAnalytics(): NetworkResult<StudyAnalytics> =
    safeKtorCall("getStudyAnalytics") {
        client.get("api/v1/analytics/study").body()
    }

suspend fun SentryKtorClient.listSessions(
    condition: String?      = null,
    organisationId: String? = null,
    limit: Int              = 50,
): NetworkResult<List<SessionSummary>> =
    safeKtorCall("listSessions") {
        client.get("api/v1/analytics/sessions") {
            condition?.let      { parameter("condition", it) }
            organisationId?.let { parameter("organisation_id", it) }
            parameter("limit", limit)
        }.body()
    }

suspend fun SentryKtorClient.getOrganisationAnalytics(
    organisationId: String,
): NetworkResult<OrganisationAnalytics> =
    safeKtorCall("getOrganisationAnalytics") {
        client.get("api/v1/analytics/organisation/${organisationId.encodeURLPathPart()}").body()
    }
