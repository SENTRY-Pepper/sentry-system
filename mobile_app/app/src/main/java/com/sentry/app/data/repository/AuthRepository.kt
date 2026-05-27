package com.sentry.app.data.repository

import com.sentry.app.core.navigation.UserRole
import com.sentry.app.data.local.TokenManager
import com.sentry.app.data.models.response.AuthState
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine
import timber.log.Timber
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class AuthRepository @Inject constructor(
    private val tokenManager: TokenManager,
) {
    val authStateFlow: Flow<AuthState> = combine(
        tokenManager.tokenFlow,
        tokenManager.roleFlow,
    ) { token, role ->
        AuthState(
            isAuthenticated = token != null,
            participantId   = tokenManager.getParticipantId() ?: "",
            role            = role ?: UserRole.TRAINEE.name.lowercase(),
            organisationId  = tokenManager.getOrganisationId() ?: "",
        )
    }

    suspend fun login(
        participantId: String,
        pin: String,
        role: String,
        organisationId: String,
    ): Result<AuthState> = runCatching {
        require(participantId.isNotBlank()) { "Participant ID is required" }
        require(pin.length >= 4) { "PIN must be at least 4 digits" }

        val token = "${participantId}_${System.currentTimeMillis()}"
        tokenManager.saveSession(token, participantId, role, organisationId)

        Timber.i("AuthRepository: login — $participantId ($role)")
        AuthState(true, participantId, role, organisationId)
    }

    suspend fun logout() {
        Timber.i("AuthRepository: logout")
        tokenManager.clearToken()
    }

    fun isAuthenticated() = tokenManager.isAuthenticated()
    fun getCurrentRole()  = UserRole.from(tokenManager.getRole() ?: "trainee")
}