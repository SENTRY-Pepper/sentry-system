package com.sentry.app.features.chat

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.repository.QueryRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ChatMessage(
    val text: String,
    val isUser: Boolean,
    val sources: List<String> = emptyList(),
)

data class ChatUiState(
    val messages: List<ChatMessage> = listOf(
        ChatMessage(
            text   = "Hello! I am Pepper, your SENTRY cybersecurity assistant. I can answer questions about cybersecurity threats, best practices, and Kenyan cyber law. What would you like to know?",
            isUser = false,
        )
    ),
    val inputText: String  = "",
    val loading: Boolean   = false,
    val error: String      = "",
)

@HiltViewModel
class ChatViewModel @Inject constructor(
    private val queryRepository: QueryRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState = _uiState.asStateFlow()

    fun onInputChanged(text: String) {
        _uiState.value = _uiState.value.copy(inputText = text)
    }

    fun sendMessage() {
        val text = _uiState.value.inputText.trim()
        if (text.isBlank() || _uiState.value.loading) return

        // Add user message immediately
        val userMsg = ChatMessage(text = text, isUser = true)
        _uiState.value = _uiState.value.copy(
            messages  = _uiState.value.messages + userMsg,
            inputText = "",
            loading   = true,
            error     = "",
        )

        viewModelScope.launch {
            when (val result = queryRepository.groundedQuery(query = text)) {
                is NetworkResult.Success -> {
                    val pepperMsg = ChatMessage(
                        text    = result.data.response,
                        isUser  = false,
                        sources = result.data.sources,
                    )
                    _uiState.value = _uiState.value.copy(
                        messages = _uiState.value.messages + pepperMsg,
                        loading  = false,
                    )
                }
                is NetworkResult.Error -> {
                    val errorMsg = ChatMessage(
                        text   = "I encountered an issue retrieving a grounded response. Please check that the SENTRY server is running.",
                        isUser = false,
                    )
                    _uiState.value = _uiState.value.copy(
                        messages = _uiState.value.messages + errorMsg,
                        loading  = false,
                    )
                }
                is NetworkResult.Exception -> {
                    val offlineMsg = ChatMessage(
                        text   = "I cannot reach the server right now. Please ensure the SENTRY middleware is running on your network.",
                        isUser = false,
                    )
                    _uiState.value = _uiState.value.copy(
                        messages = _uiState.value.messages + offlineMsg,
                        loading  = false,
                    )
                }
                is NetworkResult.Loading -> Unit
            }
        }
    }
}