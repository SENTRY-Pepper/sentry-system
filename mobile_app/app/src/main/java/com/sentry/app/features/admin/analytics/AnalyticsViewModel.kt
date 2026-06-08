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

            when (val studyResult = analyticsRepository.getStudyAnalytics()) {
                is NetworkResult.Success -> {
                    val study = studyResult.data
                    _uiState.value = _uiState.value.copy(
                        totalSessions = study.totalSessions,
                        groundedSessionCount = study.groundedSessions,
                        baselineSessionCount = study.baselineSessions,
                        groundedAccuracy = study.meanGroundingAccuracyGrounded ?: 0f,
                        baselineAccuracy = study.meanGroundingAccuracyBaseline ?: 0f,
                        groundedHallucination = study.meanHallucinationRateGrounded ?: 0f,
                        baselineHallucination = study.meanHallucinationRateBaseline ?: 0f,
                        groundingImprovement = study.meanGroundingImprovement ?: 0f,
                        groundedKnowledgeGain = study.meanKnowledgeGainGrounded ?: 0f,
                        baselineKnowledgeGain = study.meanKnowledgeGainBaseline ?: 0f,
                    )
                }

                is NetworkResult.Error -> {
                    _uiState.value = _uiState.value.copy(
                        loading = false,
                        error = studyResult.message
                    )
                    return@launch
                }

                is NetworkResult.Exception -> {
                    _uiState.value = _uiState.value.copy(
                        loading = false,
                        error = studyResult.e.message ?: "unexpected error"
                    )
                    return@launch
                }

                is NetworkResult.Loading -> Unit
            }

            when (val sessionsResult = analyticsRepository.listSessions(limit = 100)) {
                is NetworkResult.Success -> {
                    _uiState.value = _uiState.value.copy(
                        loading = false,
                        sessions = sessionsResult.data,
                    )
                }

                is NetworkResult.Error -> {
                    _uiState.value = _uiState.value.copy(
                        loading = false,
                        error = sessionsResult.message
                    )
                }

                is NetworkResult.Exception -> {
                    _uiState.value = _uiState.value.copy(
                        loading = false,
                        error = sessionsResult.e.message ?: "unexpected error"
                    )
                }

                is NetworkResult.Loading -> Unit
            }
        }
    }
}
