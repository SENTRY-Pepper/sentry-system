package com.sentry.app.data.remote.interceptors

import com.sentry.app.data.local.TokenManager
import okhttp3.Interceptor
import okhttp3.Response
import timber.log.Timber
import javax.inject.Inject
import javax.inject.Provider

class AuthInterceptor @Inject constructor(
    // Provider<T> = lazy injection — breaks circular dependency
    private val tokenProvider: Provider<TokenManager>,
) : Interceptor {

    override fun intercept(chain: Interceptor.Chain): Response {
        val token = tokenProvider.get().getToken()

        val request = chain.request().newBuilder().apply {
            if (token != null) addHeader("Authorization", "Bearer $token")
            addHeader("X-App-Version", "1.0.0")
            addHeader("X-Client", "android")
        }.build()

        val response = chain.proceed(request)

        if (response.code == 401) {
            Timber.w("AuthInterceptor: 401 — clearing session")
            tokenProvider.get().clearTokenBlocking()
        }

        return response
    }
}