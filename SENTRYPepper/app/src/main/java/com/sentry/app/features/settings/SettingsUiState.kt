package com.sentry.app.features.settings

data class SettingsUiState(
    val participantId:  String = "",
    val organisation:   String = "",
    val role:           String = "",
    // local UI prefs — not persisted to network in this study
    val groundedAi:     Boolean = true,
    val showSources:    Boolean = true,
    val notifications:  Boolean = false,
    val serverUrl:      String = "",
)
