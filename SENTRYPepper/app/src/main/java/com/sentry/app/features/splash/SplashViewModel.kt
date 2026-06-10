package com.sentry.app.features.splash

import androidx.lifecycle.ViewModel
import com.sentry.app.BuildConfig
import com.sentry.app.core.navigation.UserRole
import com.sentry.app.data.repository.AuthRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
class SplashViewModel @Inject constructor(
    private val authRepository: AuthRepository,
) : ViewModel() {
    fun isAuthenticated() = authRepository.isAuthenticated()
    fun getRole()         = authRepository.getCurrentRole()
    fun isPepperKioskMode() = BuildConfig.PEPPER_KIOSK_MODE

    suspend fun enablePepperKioskSession() {
        authRepository.enablePepperKioskSession(
            participantId = BuildConfig.PEPPER_PARTICIPANT_ID,
            organisationId = BuildConfig.PEPPER_ORGANISATION_ID,
        )
    }
}
