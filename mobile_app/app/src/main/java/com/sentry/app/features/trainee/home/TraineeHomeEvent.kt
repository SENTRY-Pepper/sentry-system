package com.sentry.app.features.trainee.home

sealed class TraineeHomeEvent {
    data class NavigateToSession(val sessionId: String, val moduleId: String?) : TraineeHomeEvent()
}
