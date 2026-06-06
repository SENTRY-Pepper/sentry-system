package com.sentry.app.features.trainee.home
data class ModuleProgress(
    val name: String,
    val progress: Float,
    val status: String,
    val isComplete: Boolean,
)
