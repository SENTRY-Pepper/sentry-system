package com.sentry.app.core.network

import io.ktor.client.plugins.ClientRequestException
import io.ktor.client.plugins.ServerResponseException
import timber.log.Timber

suspend fun <T> safeKtorCall(
    tag: String = "KtorCall",
    call: suspend () -> T,
): NetworkResult<T> {
    return try {
        NetworkResult.Success(call())
    } catch (e: ClientRequestException) {
        // 4xx
        val code = e.response.status.value
        val message = e.response.status.description
        Timber.tag(tag).e(e, "Client error $code")
        NetworkResult.Error(code, message)
    } catch (e: ServerResponseException) {
        // 5xx
        val code = e.response.status.value
        val message = e.response.status.description
        Timber.tag(tag).e(e, "Server error $code")
        NetworkResult.Error(code, message)
    } catch (e: kotlin.Exception) {
        Timber.tag(tag).e(e, "Unexpected error")
        NetworkResult.Exception(e)
    }
}
