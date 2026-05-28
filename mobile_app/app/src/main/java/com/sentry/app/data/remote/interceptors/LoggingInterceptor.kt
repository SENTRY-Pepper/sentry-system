package com.sentry.app.data.remote.interceptors

import okhttp3.Interceptor
import okhttp3.Response
import timber.log.Timber
import javax.inject.Inject

/**
 * Logs outgoing requests and incoming responses in debug builds only.
 * OkHttp's built-in HttpLoggingInterceptor is used in NetworkModule —
 * this interceptor exists for any custom structured logging we add later.
 * Currently a pass-through so the file is not empty.
 */
class LoggingInterceptor @Inject constructor() : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val request = chain.request()
        Timber.d("→ ${request.method} ${request.url}")
        val response = chain.proceed(request)
        Timber.d("← ${response.code} ${request.url}")
        return response
    }
}