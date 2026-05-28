package com.sentry.app.features.trainee.results

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import com.sentry.app.data.repository.SessionRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject

data class ResultsUiState(
    val sessionId: String  = "",
    val loading: Boolean   = false,
    val error: String      = "",
)

@HiltViewModel
class ResultsViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val sessionRepository: SessionRepository,
) : ViewModel() {
    private val sessionId: String = checkNotNull(savedStateHandle["sessionId"])
    private val _uiState = MutableStateFlow(ResultsUiState(sessionId = sessionId))
    val uiState = _uiState.asStateFlow()
}