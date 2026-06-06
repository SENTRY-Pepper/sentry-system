package com.sentry.app.features.auth

import com.sentry.app.core.navigation.UserRole

// one-shot event — emitted once on success, screen collects and navigates
sealed class AuthEvent {
    data class NavigateToHome(val role: UserRole) : AuthEvent()
}
