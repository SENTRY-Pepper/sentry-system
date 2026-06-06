package com.sentry.app.features.admin.analytics

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.models.response.SessionSummary
import com.sentry.app.data.repository.AnalyticsRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class AnalyticsViewModel @Inject constructor(
    private val analyticsRepository: AnalyticsRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(AnalyticsUiState())
    val uiState = _uiState.asStateFlow()

    init {
        loadSessions()
    }

    fun refresh() {
        loadSessions()
    }

    fun filterByCondition(condition: String?) {
        _uiState.value = _uiState.value.copy(selectedCondition = condition)
    }

    private fun loadSessions() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(loading = true, error = "")

            when (val result = analyticsRepository.listSessions()) {
                is NetworkResult.Success -> {
                    val sessions = result.data
                    _uiState.value = _uiState.value.copy(
                        loading = false,
                        sessions = sessions,
                        avgAccuracy = avgPostScore(sessions),
                        avgKnowledgeGain = avgKnowledgeGain(sessions),
                        ragSessionCount = sessions.count { it.condition == "grounded" },
                        baselineSessionCount = sessions.count { it.condition == "baseline" },
                        ragAvgAccuracy = avgPostScore(sessions.filter { it.condition == "grounded" }),
                        baselineAvgAccuracy = avgPostScore(sessions.filter { it.condition == "baseline" })
                    )
                }

                is NetworkResult.Error -> {
                    _uiState.value = _uiState.value.copy(
                        loading = false,
                        error = result.message
                    )
                }

                is NetworkResult.Exception -> {
                    _uiState.value = _uiState.value.copy(
                        loading = false,
                        error = result.e.message ?: "unexpected error"
                    )
                }

                is NetworkResult.Loading -> Unit
            }
        }
    }

    private fun avgPostScore(sessions: List<SessionSummary>): Float {
        val scored = sessions.mapNotNull { it.postAssessmentScore }
        if (scored.isEmpty()) return 0f
        return scored.average().toFloat()
    }

    private fun avgKnowledgeGain(sessions: List<SessionSummary>): Float {
        val gained = sessions.mapNotNull { it.knowledgeGain }
        if (gained.isEmpty()) return 0f
        return gained.average().toFloat()
    }
}