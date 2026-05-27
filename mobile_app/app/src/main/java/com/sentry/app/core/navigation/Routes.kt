package com.sentry.app.core.navigation

object Routes {
    const val SPLASH        = "splash"
    const val AUTH          = "auth"
    const val TRAINEE_HOME  = "trainee_home"
    const val SESSION       = "session/{sessionId}"
    const val RESULTS       = "results/{sessionId}"
    const val ADMIN_HOME    = "admin_home"
    const val ANALYTICS     = "analytics"
    const val CHAT          = "chat"
    const val SETTINGS      = "settings"

    fun session(id: String)  = "session/$id"
    fun results(id: String)  = "results/$id"
}

enum class UserRole { TRAINEE, ADMIN;
    companion object {
        fun from(v: String) = entries.firstOrNull {
            it.name.equals(v, ignoreCase = true)
        } ?: TRAINEE
    }
}