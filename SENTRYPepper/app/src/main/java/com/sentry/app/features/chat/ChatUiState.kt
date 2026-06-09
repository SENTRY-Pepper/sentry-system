package com.sentry.app.features.chat
data class ChatUiState(
    val messages: List<ChatMessage> = listOf(
        ChatMessage(
            text   = "Hello! I am Pepper, your SENTRY cybersecurity assistant. I can answer questions about cybersecurity threats, best practices, and Kenyan cyber law. What would you like to know?",
            isUser = false,
        )
    ),
    val inputText: String = "",
    val loading: Boolean  = false,
    val error: String     = "",
)