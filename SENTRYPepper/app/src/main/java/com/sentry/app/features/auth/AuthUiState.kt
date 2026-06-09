package com.sentry.app.features.auth
data class AuthUiState(
    val loading: Boolean = false,
    val error: String    = "",
)
