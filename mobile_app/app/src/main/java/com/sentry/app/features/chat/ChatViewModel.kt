package com.sentry.app.features.chat

import androidx.lifecycle.ViewModel
import com.sentry.app.data.repository.QueryRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject

data class ChatUiState(
    val messages: List<Pair<String, String>> = emptyList(), // role to content
    val loading: Boolean                     = false,
    val error: String                        = "",
)

@HiltViewModel
class ChatViewModel @Inject constructor(
    private val queryRepository: QueryRepository,
) : ViewModel() {
    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState = _uiState.asStateFlow()
}