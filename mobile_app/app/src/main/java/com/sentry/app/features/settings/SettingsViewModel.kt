package com.sentry.app.features.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sentry.app.data.local.TokenManager
import com.sentry.app.data.repository.AuthRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val authRepository: AuthRepository,
    private val tokenManager: TokenManager,
) : ViewModel() {

    private val _uiState = MutableStateFlow(SettingsUiState())
    val uiState = _uiState.asStateFlow()

    private val _events = MutableSharedFlow<SettingsEvent>()
    val events = _events.asSharedFlow()

    init {
        loadProfile()
    }

    private fun loadProfile() {
        _uiState.value = _uiState.value.copy(
            participantId = tokenManager.getParticipantId() ?: "",
            organisation = tokenManager.getOrganisationId() ?: "",
            role = tokenManager.getRole()?.replaceFirstChar { it.uppercase() } ?: "",
        )
    }

    fun setGroundedAi(enabled: Boolean) {
        _uiState.value = _uiState.value.copy(groundedAi = enabled)
    }

    fun setShowSources(enabled: Boolean) {
        _uiState.value = _uiState.value.copy(showSources = enabled)
    }

    fun setNotifications(enabled: Boolean) {
        _uiState.value = _uiState.value.copy(notifications = enabled)
    }

    fun setServerUrl(url: String) {
        _uiState.value = _uiState.value.copy(serverUrl = url)
    }

    fun logout() = viewModelScope.launch {
        authRepository.logout()
        _events.emit(SettingsEvent.LoggedOut)
    }
}