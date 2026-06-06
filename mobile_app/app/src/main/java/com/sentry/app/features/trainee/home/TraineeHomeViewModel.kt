package com.sentry.app.features.trainee.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.local.TokenManager
import com.sentry.app.data.repository.AuthRepository
import com.sentry.app.data.repository.SessionRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

fun defaultModules() = listOf(
    ModuleProgress("Phishing Detection",       1.0f, "Complete",    true),
    ModuleProgress("USB Drop Simulation",      1.0f, "Complete",    true),
    ModuleProgress("Password Hygiene",         0.4f, "In Progress", false),
    ModuleProgress("Voice Social Engineering", 0.0f, "Not Started", false),
    ModuleProgress("Network Hygiene",          0.0f, "Not Started", false),
)

@HiltViewModel
class TraineeHomeViewModel @Inject constructor(
    private val authRepository: AuthRepository,
    private val sessionRepository: SessionRepository,
    private val tokenManager: TokenManager,
) : ViewModel() {

    private val _uiState = MutableStateFlow(TraineeHomeUiState())
    val uiState = _uiState.asStateFlow()

    private val _events = MutableSharedFlow<TraineeHomeEvent>()
    val events = _events.asSharedFlow()

    init { loadProfile() }

    private fun loadProfile() {
        val participantId = tokenManager.getParticipantId() ?: ""
        val organisation  = tokenManager.getOrganisationId() ?: ""
        val modulesLeft   = defaultModules().count { !it.isComplete }

        _uiState.value = _uiState.value.copy(
            participantId = participantId,
            organisation  = organisation,
            modulesLeft   = modulesLeft,
        )
    }

    fun startSession(condition: String = "grounded") = viewModelScope.launch {
        _uiState.value = _uiState.value.copy(loading = true, error = "")

        // SessionRepository reads participantId + organisationId from TokenManager internally
        when (val result = sessionRepository.startSession(condition = condition)) {
            is NetworkResult.Success -> {
                _uiState.value = _uiState.value.copy(loading = false)
                _events.emit(TraineeHomeEvent.NavigateToSession(result.data.sessionId))
            }
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
