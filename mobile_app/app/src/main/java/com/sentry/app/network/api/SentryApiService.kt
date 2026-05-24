package com.sentry.app.network.api

import com.sentry.app.models.request.*
import com.sentry.app.models.response.*
import retrofit2.Response
import retrofit2.http.*

interface SentryApiService {

    @GET("health")
    suspend fun healthCheck(): Response<HealthResponse>

    @GET("api/v1/knowledge-base/status")
    suspend fun knowledgeBaseStatus(): Response<KnowledgeBaseStats>

    @POST("api/v1/query")
    suspend fun groundedQuery(@Body request: QueryRequest): Response<QueryResponse>

    @POST("api/v1/query/baseline")
    suspend fun baselineQuery(@Body request: QueryRequest): Response<QueryResponse>

    @POST("api/v1/sessions/start")
    suspend fun startSession(@Body request: SessionStartRequest): Response<SessionStartResponse>

    @POST("api/v1/sessions/end")
    suspend fun endSession(@Body request: SessionEndRequest): Response<SessionEndResponse>

    @GET("api/v1/sessions/{sessionId}")
    suspend fun getSession(@Path("sessionId") sessionId: String): Response<SessionSummary>

    @POST("api/v1/sessions/interaction")
    suspend fun logInteraction(@Body request: InteractionLogRequest): Response<InteractionLogResponse>

    @POST("api/v1/sessions/eval-log")
    suspend fun logEvaluation(@Body request: EvalLogRequest): Response<Map<String, Boolean>>

    @GET("api/v1/analytics/study")
    suspend fun getStudyAnalytics(): Response<StudyAnalytics>

    @GET("api/v1/analytics/sessions")
    suspend fun listSessions(
        @Query("condition") condition: String? = null,
        @Query("organisation_id") organisationId: String? = null,
        @Query("limit") limit: Int = 100,
    ): Response<List<SessionSummary>>

    @GET("api/v1/analytics/organisation/{organisationId}")
    suspend fun getOrganisationAnalytics(
        @Path("organisationId") organisationId: String,
    ): Response<OrganisationAnalytics>
}
