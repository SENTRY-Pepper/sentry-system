package com.sentry.app.features.manager.home

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
class ManagerHomeViewModel @Inject constructor(
    private val analyticsRepository: AnalyticsRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(ManagerHomeUiState())
    val uiState = _uiState.asStateFlow()

    init {
        refresh()
    }

    fun refresh() {
        loadOverview()
        loadTrainees()
    }

    private fun loadOverview() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(loading = true, error = "")

            when (val result = analyticsRepository.getManagerOverview()) {
                is NetworkResult.Success -> {
                    val overview = result.data
                    _uiState.value = _uiState.value.copy(
                        loading = false,
                        organisationId = overview.organisationId,
                        traineeCount = overview.traineeCount,
                        activeTrainees = overview.activeTrainees,
                        totalSessions = overview.totalSessions,
                        completedSessions = overview.completedSessions,
                        averageScore = overview.averageScore ?: 0f,
                        completionRate = overview.completionRate,
                        riskyAnswers = overview.riskyAnswers,
                        weaknesses = overview.topWeaknesses,
                        departments = overview.departments,
                    )
                }

                is NetworkResult.Error -> {
                    _uiState.value = _uiState.value.copy(
                        loading = false,
                        error = result.message,
                    )
                }

                is NetworkResult.Exception -> {
                    _uiState.value = _uiState.value.copy(
                        loading = false,
                        error = result.e.message ?: "Could not load manager analytics.",
                    )
                }

                is NetworkResult.Loading -> Unit
            }
        }
    }

    private fun loadTrainees() {
        viewModelScope.launch {
            when (val result = analyticsRepository.listTrainees()) {
                is NetworkResult.Success -> {
                    _uiState.value = _uiState.value.copy(trainees = result.data)
                }

                else -> Unit
            }
        }
    }
}
