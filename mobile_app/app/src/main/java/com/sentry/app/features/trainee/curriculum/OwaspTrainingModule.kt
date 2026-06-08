package com.sentry.app.features.trainee.curriculum

data class OwaspAnswerOption(
    val id: String,
    val label: String,
    val text: String,
    val isCorrect: Boolean,
    val feedback: String,
)

data class OwaspTrainingModule(
    val id: String,
    val owaspId: String,
    val title: String,
    val difficulty: String,
    val summary: String,
    val workplaceTakeaway: String,
    val scenario: String,
    val options: List<OwaspAnswerOption>,
    val correctAnswerId: String,
    val sourceReference: String,
)

val OwaspTrainingModule.correctOption: OwaspAnswerOption
    get() = options.first { it.id == correctAnswerId }
