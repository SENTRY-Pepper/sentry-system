package com.sentry.app.core.navigation

import kotlinx.serialization.Serializable

sealed class Routes {

    @Serializable
    data object Splash : Routes()

    @Serializable
    data object Auth : Routes()

    @Serializable
    data object TraineeHome : Routes()

    @Serializable
    data object AdminHome : Routes()

    @Serializable
    data object ManagerHome : Routes()

    @Serializable
    data object Analytics : Routes()

    @Serializable
    data object Chat : Routes()

    @Serializable
    data object Settings : Routes()

    // routes with args
    @Serializable
    data class Session(val sessionId: String) : Routes()

    @Serializable
    data class Results(val sessionId: String) : Routes()
}

enum class UserRole {
    TRAINEE, MANAGER, ADMIN;

    companion object {
        fun from(v: String) = entries.firstOrNull {
            it.name.equals(v, ignoreCase = true)
        } ?: TRAINEE
    }
}
