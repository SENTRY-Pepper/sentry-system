package com.sentry.app.repository

import com.sentry.app.UserRole
import com.sentry.app.models.response.AuthState
import com.sentry.app.network.interceptors.TokenManager
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine
import timber.log.Timber
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class AuthRepository @Inject constructor(private val tokenManager: TokenManager) {

    val authStateFlow: Flow<AuthState> = combine(
        tokenManager.tokenFlow,
        tokenManager.roleFlow,
    ) { token, role ->
        AuthState(
            isAuthenticated = token != null,
            participantId   = tokenManager.getParticipantId() ?: "",
            role            = role ?: "trainee",
            organisationId  = tokenManager.getOrganisationId() ?: "",
            token           = token ?: "",
        )
    }

    suspend fun login(participantId: String, pin: String, role: String, organisationId: String): Result<AuthState> {
        return try {
            if (participantId.isBlank()) return Result.failure(IllegalArgumentException("Participant ID is required"))
            if (pin.length < 4) return Result.failure(IllegalArgumentException("PIN must be at least 4 digits"))
            val token = "${participantId}_${System.currentTimeMillis()}"
            tokenManager.saveCredentials(token, participantId, role, organisationId)
            Timber.d("AuthRepository: login success — $participantId ($role)")
            Result.success(AuthState(true, participantId, role, organisationId, token))
        } catch (e: Exception) {
            Timber.e(e, "AuthRepository: login failed")
            Result.failure(e)
        }
    }

    suspend fun logout() {
        Timber.d("AuthRepository: logout")
        tokenManager.clearToken()
    }

    fun isAuthenticated(): Boolean = tokenManager.isAuthenticated()
    fun getCurrentRole(): UserRole = UserRole.fromString(tokenManager.getRole() ?: "trainee")
}
