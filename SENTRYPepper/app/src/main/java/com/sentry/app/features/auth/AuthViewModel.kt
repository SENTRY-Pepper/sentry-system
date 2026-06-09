package com.sentry.app.features.auth

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sentry.app.core.navigation.UserRole
import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.repository.AuthRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class AuthViewModel @Inject constructor(
    private val authRepository: AuthRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(AuthUiState())
    val uiState = _uiState.asStateFlow()

    private val _events = MutableSharedFlow<AuthEvent>()
    val events = _events.asSharedFlow()

    fun login(
        participantId: String,
        pin: String,
        role: String,
        organisationId: String,
    ) = viewModelScope.launch {
        _uiState.value = AuthUiState(loading = true)

        when (val result = authRepository.login(participantId, pin, role, organisationId)) {
            is NetworkResult.Success -> {
                _uiState.value = AuthUiState()
                _events.emit(AuthEvent.NavigateToHome(UserRole.from(result.data.role)))
            }

            is NetworkResult.Error -> {
                _uiState.value = AuthUiState(error = result.message ?: "Login failed")
            }

            is NetworkResult.Exception -> {
                _uiState.value = AuthUiState(error = result.e.message ?: "Unexpected error")
            }

            is NetworkResult.Loading -> Unit
        }
    }
}