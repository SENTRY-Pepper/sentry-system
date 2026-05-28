package com.sentry.app.features.admin.analytics

import androidx.lifecycle.ViewModel
import com.sentry.app.core.rbac.RoleGuard
import com.sentry.app.data.repository.AnalyticsRepository
import com.sentry.app.data.repository.AuthRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject

data class AnalyticsUiState(
    val loading: Boolean = false,
    val error: String    = "",
)

@HiltViewModel
class AnalyticsViewModel @Inject constructor(
    private val authRepository: AuthRepository,
    private val analyticsRepository: AnalyticsRepository,
) : ViewModel() {
    init { RoleGuard.requireAdmin(authRepository.getCurrentRole()) }
    private val _uiState = MutableStateFlow(AnalyticsUiState())
    val uiState = _uiState.asStateFlow()
}