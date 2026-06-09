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
        sendText(_uiState.value.inputText)
    }

    fun sendText(rawText: String) {
        val text = rawText.trim()
        if (text.isBlank() || _uiState.value.loading) return

        val userMsg = ChatMessage(text = text, isUser = true)
        _uiState.value = _uiState.value.copy(
            messages = _uiState.value.messages + userMsg,
            inputText = "",
            loading = true,
            error = "",
        )

        viewModelScope.launch {
            // scenarioId is null for free-form chat — not tied to a specific scenario
            when (val result = queryRepository.groundedQuery(
                query = text,
                scenarioId = null,
            )) {
                is NetworkResult.Success -> {
                    _uiState.value = _uiState.value.copy(
                        messages = _uiState.value.messages + ChatMessage(
                            text = result.data.response,
                            isUser = false,
                            sources = result.data.sources,
                        ),
                        loading = false,
                    )
                }

                is NetworkResult.Error -> {
                    _uiState.value = _uiState.value.copy(
                        messages = _uiState.value.messages + ChatMessage(
                            text = "I encountered an issue retrieving a grounded response. Please check that the SENTRY server is running.",
                            isUser = false,
                        ),
                        loading = false,
                    )
                }

                is NetworkResult.Exception -> {
                    _uiState.value = _uiState.value.copy(
                        messages = _uiState.value.messages + ChatMessage(
                            text = "I cannot reach the server right now. Please ensure the SENTRY middleware is running on your network.",
                            isUser = false,
                        ),
                        loading = false,
                    )
                }

                is NetworkResult.Loading -> Unit
            }
        }
    }
}
