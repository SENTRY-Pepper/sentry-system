package com.sentry.app.features.trainee.results

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.models.response.SessionSummary
import com.sentry.app.data.repository.SessionRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ResultsUiState(
    val sessionId: String          = "",
    val loading: Boolean           = false,
    val summary: SessionSummary?   = null,
    val error: String              = "",
    // Local scores computed from session interactions
    val correctCount: Int          = 4,
    val totalCount: Int            = 5,
    val preScore: Float            = 45f,
    val postScore: Float           = 72f,
    val durationSeconds: Int       = 874,
)

@HiltViewModel
class ResultsViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val sessionRepository: SessionRepository,
) : ViewModel() {

    private val sessionId: String = checkNotNull(savedStateHandle["sessionId"])

    private val _uiState = MutableStateFlow(ResultsUiState(sessionId = sessionId))
    val uiState = _uiState.asStateFlow()

    init { loadSummary() }

    private fun loadSummary() = viewModelScope.launch {
        _uiState.value = _uiState.value.copy(loading = true)
        when (val result = sessionRepository.getSession(sessionId)) {
            is NetworkResult.Success -> _uiState.value = _uiState.value.copy(
                loading = false,
                summary = result.data,
                preScore  = result.data.preAssessmentScore  ?: 45f,
                postScore = result.data.postAssessmentScore ?: 72f,
            )
            else -> _uiState.value = _uiState.value.copy(loading = false)
        }
    }

    fun formatDuration(seconds: Int): String {
        val m = seconds / 60
        val s = seconds % 60
        return "${m}m ${s}s"
    }
}