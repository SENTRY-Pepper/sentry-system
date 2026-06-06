package com.sentry.app.data.repository

import com.sentry.app.core.network.NetworkResult
import com.sentry.app.core.network.SentryKtorClient
import com.sentry.app.data.models.request.QueryRequest
import com.sentry.app.data.models.response.QueryResponse
import com.sentry.app.data.remote.api.baselineQuery
import com.sentry.app.data.remote.api.groundedQuery
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class QueryRepositoryImpl @Inject constructor(
    private val ktorClient: SentryKtorClient,
) : QueryRepository {

    override suspend fun groundedQuery(
        query: String,
        scenarioId: String?,
    ): NetworkResult<QueryResponse> =
        ktorClient.groundedQuery(
            QueryRequest(
                query = query,
                scenarioId = scenarioId,
            )
        )

    override suspend fun baselineQuery(
        query: String,
        scenarioId: String?,
    ): NetworkResult<QueryResponse> =
        ktorClient.baselineQuery(
            QueryRequest(
                query = query,
                scenarioId = scenarioId,
            )
        )
}