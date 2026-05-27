package com.sentry.app.features.auth

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sentry.app.core.navigation.UserRole
import com.sentry.app.data.repository.AuthRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class AuthUiState(
    val loading: Boolean       = false,
    val error: String          = "",
    val success: UserRole?     = null,
)

@HiltViewModel
class AuthViewModel @Inject constructor(
    private val authRepository: AuthRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(AuthUiState())
    val uiState = _uiState.asStateFlow()

    fun login(
        participantId: String,
        pin: String,
        role: String,
        organisationId: String,
    ) = viewModelScope.launch {
        _uiState.value = AuthUiState(loading = true)

        authRepository.login(participantId, pin, role, organisationId)
            .onSuccess { state ->
                _uiState.value = AuthUiState(success = UserRole.from(state.role))
            }
            .onFailure { e ->
                _uiState.value = AuthUiState(error = e.message ?: "Login failed")
            }
    }
}