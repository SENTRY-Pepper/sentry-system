package com.sentry.app.features.trainee.home
data class TraineeHomeUiState(
    val participantId:  String              = "",
    val organisation:   String              = "",
    val loading:        Boolean             = false,
    val error:          String              = "",
    val modules:        List<ModuleProgress> = defaultModules(),
    val sessionsCompleted: Int              = 0,
    val avgAccuracy:    String              = "--",
    val modulesLeft:    Int                 = 10,
    val quickAskOpen:   Boolean             = false,
    val quickAskTranscript: String          = "",
    val quickAskResponse: String            = "",
    val quickAskLoading: Boolean            = false,
    val quickAskError: String               = "",
)
