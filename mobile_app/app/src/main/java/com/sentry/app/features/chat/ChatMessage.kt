package com.sentry.app.features.chat
data class ChatMessage(
    val text: String,
    val isUser: Boolean,
    val sources: List<String> = emptyList(),
)
