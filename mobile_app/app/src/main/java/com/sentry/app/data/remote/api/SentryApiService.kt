package com.sentry.app.data.remote.api

import com.sentry.app.data.models.request.*
import com.sentry.app.data.models.response.*
import retrofit2.Response
import retrofit2.http.*

interface SentryApiService {

    // Health
    @GET("health")
    suspend fun healthCheck(): Response<HealthResponse>

    // AI Query — only fetch what is needed per scenario
    @POST("api/v1/query")
    suspend fun groundedQuery(@Body request: QueryRequest): Response<QueryResponse>

    @POST("api/v1/query/baseline")
    suspend fun baselineQuery(@Body request: QueryRequest): Response<QueryResponse>

    // Session lifecycle
    @POST("api/v1/sessions/start")
    suspend fun startSession(@Body request: SessionStartRequest): Response<SessionStartResponse>

    @POST("api/v1/sessions/end")
    suspend fun endSession(@Body request: SessionEndRequest): Response<SessionEndResponse>

    @GET("api/v1/sessions/{sessionId}")
    suspend fun getSession(@Path("sessionId") sessionId: String): Response<SessionSummary>

    // Logging — fire-and-forget, failures are non-fatal
    @POST("api/v1/sessions/interaction")
    suspend fun logInteraction(@Body request: InteractionLogRequest): Response<InteractionLogResponse>

    // Analytics — admin only, fetched lazily never on startup
    @GET("api/v1/analytics/study")
    suspend fun getStudyAnalytics(): Response<StudyAnalytics>

    @GET("api/v1/analytics/sessions")
    suspend fun listSessions(
        @Query("condition")       condition: String?      = null,
        @Query("organisation_id") organisationId: String? = null,
        @Query("limit")           limit: Int              = 50,
    ): Response<List<SessionSummary>>

    @GET("api/v1/analytics/organisation/{orgId}")
    suspend fun getOrganisationAnalytics(
        @Path("orgId") organisationId: String,
    ): Response<OrganisationAnalytics>
}