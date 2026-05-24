package com.sentry.app

object Routes {
    const val SPLASH             = "splash"
    const val AUTH               = "auth"
    const val TRAINEE_HOME       = "trainee_home"
    const val SESSION            = "session/{sessionId}"
    const val SESSION_RESULTS    = "session_results/{sessionId}"
    const val ADMIN_HOME         = "admin_home"
    const val ANALYTICS          = "analytics"
    const val PARTICIPANT_DETAIL = "participant/{participantId}"
    const val CHAT               = "chat"
    const val SETTINGS           = "settings"

    fun session(sessionId: String)               = "session/$sessionId"
    fun sessionResults(sessionId: String)        = "session_results/$sessionId"
    fun participantDetail(participantId: String) = "participant/$participantId"
}

enum class UserRole {
    TRAINEE, ADMIN;
    companion object {
        fun fromString(value: String): UserRole =
            entries.firstOrNull { it.name.equals(value, ignoreCase = true) } ?: TRAINEE
    }
}
