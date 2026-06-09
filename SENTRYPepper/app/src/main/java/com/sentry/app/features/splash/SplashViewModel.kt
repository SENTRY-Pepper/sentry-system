package com.sentry.app.features.splash

import androidx.lifecycle.ViewModel
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
}
