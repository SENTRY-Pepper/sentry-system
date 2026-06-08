package com.sentry.app.data.repository

import com.sentry.app.core.navigation.UserRole
import com.sentry.app.core.network.NetworkResult
import com.sentry.app.core.network.SentryKtorClient
import com.sentry.app.core.organisation.normaliseOrganisationId
import com.sentry.app.data.local.TokenManager
import com.sentry.app.data.models.request.UserLoginRequest
import com.sentry.app.data.models.response.AuthState
import com.sentry.app.data.remote.api.loginUser
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine
import timber.log.Timber
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class AuthRepository @Inject constructor(
    private val tokenManager: TokenManager,
    private val ktorClient: SentryKtorClient,
) {
    val authStateFlow: Flow<AuthState> = combine(
        tokenManager.tokenFlow,
        tokenManager.roleFlow,
    ) { token, role ->
        AuthState(
            isAuthenticated = token != null,
            participantId = tokenManager.getParticipantId() ?: "",
            role = role ?: UserRole.TRAINEE.name.lowercase(),
            organisationId = tokenManager.getOrganisationId() ?: "",
        )
    }

    suspend fun login(
        participantId: String,
        pin: String,
        role: String,
        organisationId: String,
    ): NetworkResult<AuthState> {
        return try {
            require(participantId.isNotBlank()) { "Participant ID is required" }
            require(pin.length >= 4) { "PIN must be at least 4 digits" }
            require(role == UserRole.TRAINEE.name.lowercase() || organisationId.isNotBlank()) {
                "Organisation ID is required for manager/admin login"
            }

            val canonicalOrganisationId = normaliseOrganisationId(organisationId)
                .ifEmpty { "SENTRY_STUDY" }
            when (
                val loginResult = ktorClient.loginUser(
                    UserLoginRequest(
                        participantId = participantId.trim(),
                        pin = pin.trim(),
                        role = role,
                        organisationId = canonicalOrganisationId,
                    )
                )
            ) {
                is NetworkResult.Success -> {
                    val user = loginResult.data.user
                    val resolvedOrg = user.organisationId ?: canonicalOrganisationId
                    tokenManager.saveSession(
                        token = loginResult.data.token,
                        participantId = user.participantId,
                        role = user.role,
                        organisationId = resolvedOrg,
                    )

                    Timber.i("AuthRepository: backend login - ${user.participantId} (${user.role})")
                    NetworkResult.Success(
                        AuthState(
                            isAuthenticated = true,
                            participantId = user.participantId,
                            role = user.role,
                            organisationId = resolvedOrg,
                        )
                    )
                }

                is NetworkResult.Error -> loginResult
                is NetworkResult.Exception -> loginResult
                is NetworkResult.Loading -> NetworkResult.Loading
            }
        } catch (e: IllegalArgumentException) {
            NetworkResult.Error(
                message = e.message ?: "Validation failed"
            )
        } catch (e: Exception) {
            NetworkResult.Exception(e)
        }
    }

    suspend fun logout() {
        Timber.i("AuthRepository: logout")
        tokenManager.clearToken()
    }

    fun isAuthenticated() = tokenManager.isAuthenticated()
    fun getCurrentRole() = UserRole.from(tokenManager.getRole() ?: "trainee")
}
