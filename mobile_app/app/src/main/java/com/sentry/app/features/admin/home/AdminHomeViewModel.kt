package com.sentry.app.features.admin.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.repository.AnalyticsRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class AdminHomeViewModel @Inject constructor(
    private val analyticsRepository: AnalyticsRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(AdminHomeUiState())
    val uiState = _uiState.asStateFlow()

    init {
        loadDashboard()
    }

    fun loadDashboard() {
        loadStudyMetrics()
        loadRecentSessions()
    }

    fun dismissError() {
        _uiState.value = _uiState.value.copy(error = "")
    }

    private fun loadStudyMetrics() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(loading = true, error = "")

            when (val result = analyticsRepository.getStudyAnalytics()) {
                is NetworkResult.Success -> {
                    val data = result.data
                    _uiState.value = _uiState.value.copy(
                        loading = false,
                        totalSessions = data.totalSessions,
                        groundedSessions = data.groundedSessions,
                        baselineSessions = data.baselineSessions,
                        groundedAccuracy = data.meanGroundingAccuracyGrounded ?: 0f,
                        baselineAccuracy = data.meanGroundingAccuracyBaseline ?: 0f,
                        groundedHallucination = data.meanHallucinationRateGrounded ?: 0f,
                        baselineHallucination = data.meanHallucinationRateBaseline ?: 0f,
                        groundingImprovement = data.meanGroundingImprovement ?: 0f,
                        groundedKnowledgeGain = data.meanKnowledgeGainGrounded ?: 0f,
                        baselineKnowledgeGain = data.meanKnowledgeGainBaseline ?: 0f,
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

    private fun loadRecentSessions() {
        viewModelScope.launch {
            when (val result = analyticsRepository.listSessions(limit = 5)) {
                is NetworkResult.Success -> {
                    _uiState.value = _uiState.value.copy(
                        recentSessions = result.data
                    )
                }

                else -> Unit
            }
        }
    }
}

