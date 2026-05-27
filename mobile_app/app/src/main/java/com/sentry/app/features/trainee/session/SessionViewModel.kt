package com.sentry.app.features.trainee.session

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import com.sentry.app.data.repository.QueryRepository
import com.sentry.app.data.repository.SessionRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject

data class SessionUiState(
    val sessionId: String     = "",
    val currentScenario: Int  = 0,
    val totalScenarios: Int   = 5,
    val loading: Boolean      = false,
    val error: String         = "",
    val isComplete: Boolean   = false,
)

@HiltViewModel
class SessionViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val sessionRepository: SessionRepository,
    private val queryRepository: QueryRepository,
) : ViewModel() {

    private val sessionId: String = checkNotNull(savedStateHandle["sessionId"])

    private val _uiState = MutableStateFlow(SessionUiState(sessionId = sessionId))
    val uiState = _uiState.asStateFlow()

    // Scenario logic and AI query calls added when we build this screen
}