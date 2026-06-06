package com.sentry.app.data.repository

import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.models.response.QueryResponse

interface QueryRepository {

    suspend fun groundedQuery(
        query: String,
        scenarioId: String?,
    ): NetworkResult<QueryResponse>

    suspend fun baselineQuery(
        query: String,
        scenarioId: String?,
    ): NetworkResult<QueryResponse>
}
