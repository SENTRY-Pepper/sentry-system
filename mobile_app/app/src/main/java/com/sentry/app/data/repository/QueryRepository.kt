package com.sentry.app.data.repository

import com.sentry.app.core.network.NetworkResult
import com.sentry.app.core.network.safeApiCall
import com.sentry.app.data.models.request.QueryRequest
import com.sentry.app.data.models.response.QueryResponse
import com.sentry.app.data.remote.api.SentryApiService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import timber.log.Timber
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class QueryRepository @Inject constructor(private val api: SentryApiService) {

    suspend fun groundedQuery(
        query: String, scenarioId: String? = null,
    ): NetworkResult<QueryResponse> = withContext(Dispatchers.IO) {
        Timber.d("QueryRepository: grounded — $query")
        safeApiCall("Query.grounded") {
            api.groundedQuery(QueryRequest(query = query, scenarioId = scenarioId))
        }
    }

    suspend fun baselineQuery(
        query: String, scenarioId: String? = null,
    ): NetworkResult<QueryResponse> = withContext(Dispatchers.IO) {
        Timber.d("QueryRepository: baseline — $query")
        safeApiCall("Query.baseline") {
            api.baselineQuery(QueryRequest(query = query, scenarioId = scenarioId))
        }
    }
}