package com.sentry.app.features.settings
sealed class SettingsEvent {
    data object LoggedOut : SettingsEvent()
}
