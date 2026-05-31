package com.sentry.app.features.trainee.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.repository.AuthRepository
import com.sentry.app.data.repository.SessionRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class TraineeHomeUiState(
    val participantId: String  = "",
    val loading: Boolean       = false,
    val error: String          = "",
    val sessionStarted: String? = null,
)

@HiltViewModel
class TraineeHomeViewModel @Inject constructor(
    private val authRepository: AuthRepository,
    private val sessionRepository: SessionRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(TraineeHomeUiState())
    val uiState = _uiState.asStateFlow()

    init {
        val id = authRepository.getCurrentRole().name
        _uiState.value = _uiState.value.copy(participantId = id)
    }

    fun startSession(condition: String = "grounded") = viewModelScope.launch {
        _uiState.value = _uiState.value.copy(loading = true, error = "")

        when (val result = sessionRepository.startSession(
            participantId  = "EMP_042",
            condition      = condition,
            organisationId = "SENTRY_STUDY",
        )) {
            is NetworkResult.Success -> _uiState.value = _uiState.value.copy(
                loading        = false,
                sessionStarted = result.data.sessionId,
            )
            is NetworkResult.Error -> _uiState.value = _uiState.value.copy(
                loading = false,
                error   = "Server error. Check your connection.",
            )
            is NetworkResult.Exception -> _uiState.value = _uiState.value.copy(
                loading = false,
                error   = "Cannot reach server. Is the middleware running?",
            )
            is NetworkResult.Loading -> Unit
        }
    }
}