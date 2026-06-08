package com.sentry.app.core.network

import com.sentry.app.BuildConfig
import com.sentry.app.data.local.TokenManager
import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.plugins.defaultRequest
import io.ktor.client.plugins.logging.LogLevel
import io.ktor.client.plugins.logging.Logger
import io.ktor.client.plugins.logging.Logging
import io.ktor.client.plugins.HttpTimeout
import io.ktor.http.ContentType
import io.ktor.http.contentType
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.json.Json
import timber.log.Timber

class SentryKtorClient(
    private val tokenManager: TokenManager,
    private val baseUrl: String,
) {
    val client = HttpClient(CIO) {
        expectSuccess = true

        install(HttpTimeout) {
            connectTimeoutMillis = 10_000   // 10s connect
            requestTimeoutMillis = 60_000   // 60s — GPT-4 worst case
            socketTimeoutMillis  = 60_000
        }

        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
                coerceInputValues  = true
                isLenient          = true
            })
        }

        install(Logging) {
            logger = object : Logger {
                override fun log(message: String) {
                    Timber.tag("SentryKtor").d(message)
                }
            }
            level = if (BuildConfig.DEBUG) LogLevel.BODY else LogLevel.NONE
        }

        defaultRequest {
            url(baseUrl)
            contentType(ContentType.Application.Json)

            // same pattern as Odit — read token synchronously off main thread
            val token = tokenManager.getToken()
            if (token != null) {
                headers.append("Authorization", "Bearer $token")
            }
        }
    }
}
