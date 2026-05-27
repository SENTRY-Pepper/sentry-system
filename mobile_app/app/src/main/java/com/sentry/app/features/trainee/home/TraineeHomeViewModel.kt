package com.sentry.app.features.trainee.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.models.response.SessionStartResponse
import com.sentry.app.data.repository.AuthRepository
import com.sentry.app.data.repository.SessionRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class TraineeHomeUiState(
    val participantId: String         = "",
    val loading: Boolean              = false,
    val error: String                 = "",
    val sessionStarted: String?       = null, // sessionId on success
)

@HiltViewModel
class TraineeHomeViewModel @Inject constructor(
    private val authRepository: AuthRepository,
    private val sessionRepository: SessionRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(TraineeHomeUiState())
    val uiState = _uiState.asStateFlow()

    init {
        // Load participant info without a network call
        val id = authRepository.getCurrentRole()
        _uiState.value = _uiState.value.copy(participantId = id.name)
    }

    fun startSession(condition: String = "grounded") = viewModelScope.launch {
        _uiState.value = _uiState.value.copy(loading = true, error = "")
        val participantId  = authRepository.getCurrentRole().name
        val organisationId = "SENTRY_STUDY"

        when (val result = sessionRepository.startSession(participantId, condition, organisationId)) {
            is NetworkResult.Success -> {
                _uiState.value = _uiState.value.copy(
                    loading        = false,
                    sessionStarted = result.data.sessionId,
                )
            }
            is NetworkResult.Error -> {
                _uiState.value = _uiState.value.copy(
                    loading = false,
                    error   = result.message,
                )
            }
            is NetworkResult.Exception -> {
                _uiState.value = _uiState.value.copy(
                    loading = false,
                    error   = "Cannot reach server. Check your connection.",
                )
            }
            is NetworkResult.Loading -> Unit
        }
    }
}