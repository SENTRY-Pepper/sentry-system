package com.sentry.app.core.network

import retrofit2.Response
import timber.log.Timber

suspend fun <T> safeApiCall(
    tag: String = "API",
    call: suspend () -> Response<T>,
): NetworkResult<T> = try {
    val response = call()
    if (response.isSuccessful) {
        val body = response.body()
        if (body != null) {
            NetworkResult.Success(body)
        } else {
            NetworkResult.Error(response.code(), "Empty response body")
        }
    } else {
        Timber.w("[$tag] HTTP ${response.code()} — ${response.message()}")
        NetworkResult.Error(response.code(), response.message())
    }
} catch (e: Exception) {
    Timber.e(e, "[$tag] Network exception")
    NetworkResult.Exception(e)
}