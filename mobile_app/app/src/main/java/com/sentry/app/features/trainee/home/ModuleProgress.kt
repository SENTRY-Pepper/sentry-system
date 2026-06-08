package com.sentry.app.features.trainee.home
data class ModuleProgress(
    val id: String,
    val name: String,
    val progress: Float,
    val scorePercent: Int?,
    val status: String,
    val isComplete: Boolean,
)
