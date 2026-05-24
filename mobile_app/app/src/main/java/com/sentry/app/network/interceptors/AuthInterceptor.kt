package com.sentry.app.network.interceptors

import okhttp3.Interceptor
import okhttp3.Response
import timber.log.Timber
import javax.inject.Inject
import javax.inject.Provider

class AuthInterceptor @Inject constructor(
    private val tokenProvider: Provider<TokenManager>,
) : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val token = tokenProvider.get().getToken()
        val request = if (token != null) {
            chain.request().newBuilder()
                .addHeader("Authorization", "Bearer $token")
                .addHeader("X-App-Version", "1.0.0")
                .build()
        } else {
            chain.request()
        }
        val response = chain.proceed(request)
        if (response.code == 401) {
            Timber.w("AuthInterceptor: 401 — clearing token")
            tokenProvider.get().clearToken()
        }
        return response
    }
}
