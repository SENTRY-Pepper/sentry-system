package com.sentry.app.features.trainee.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.local.TokenManager
import com.sentry.app.data.repository.SessionRepository
import com.sentry.app.features.trainee.curriculum.OwaspCurriculum
import com.sentry.app.features.trainee.curriculum.TrainingProgressStore
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

fun defaultModules() = listOf(
    ModuleProgress("A01 Broken Access Control", 0.35f, "In Progress", false),
    ModuleProgress("A02 Security Misconfiguration", 0.0f, "Not Started", false),
    ModuleProgress("A03 Software Supply Chain", 0.0f, "Not Started", false),
    ModuleProgress("A04 Cryptographic Failures", 0.0f, "Not Started", false),
    ModuleProgress("A05 Injection", 0.0f, "Not Started", false),
    ModuleProgress("A06 Insecure Design", 0.0f, "Not Started", false),
    ModuleProgress("A07 Authentication Failures", 0.0f, "Not Started", false),
    ModuleProgress("A08 Software/Data Integrity", 0.0f, "Not Started", false),
    ModuleProgress("A09 Logging and Alerting", 0.0f, "Not Started", false),
    ModuleProgress("A10 Exceptional Conditions", 0.0f, "Not Started", false),
)

@HiltViewModel
class TraineeHomeViewModel @Inject constructor(
    private val sessionRepository: SessionRepository,
    private val tokenManager: TokenManager,
    private val progressStore: TrainingProgressStore,
) : ViewModel() {

    private val _uiState = MutableStateFlow(TraineeHomeUiState())
    val uiState = _uiState.asStateFlow()

    private val _events = MutableSharedFlow<TraineeHomeEvent>()
    val events = _events.asSharedFlow()

    init { refreshProfile() }

    fun refreshProfile() = viewModelScope.launch {
        val participantId = tokenManager.getParticipantId() ?: ""
        val organisation  = tokenManager.getOrganisationId() ?: ""
        val completedIds = progressStore.completedModuleIds()
        val firstIncomplete = OwaspCurriculum.modules.firstOrNull {
            it.id !in completedIds
        }?.id
        val modules = OwaspCurriculum.modules.map { module ->
            val complete = module.id in completedIds
            val inProgress = module.id == firstIncomplete
            ModuleProgress(
                name = "${module.owaspId} ${module.title}",
                progress = when {
                    complete -> 1.0f
                    inProgress -> 0.35f
                    else -> 0.0f
                },
                status = when {
                    complete -> "Complete"
                    inProgress -> "In Progress"
                    else -> "Not Started"
                },
                isComplete = complete,
            )
        }
        val sessionsCompleted = progressStore.sessionsCompleted()
        val averageAccuracy = progressStore.averageAccuracy()
        val modulesLeft = modules.count { !it.isComplete }

        _uiState.value = _uiState.value.copy(
            participantId = participantId,
            organisation  = organisation,
            modules = modules,
            sessionsCompleted = sessionsCompleted,
            avgAccuracy = averageAccuracy?.let { "${it.toInt()}%" } ?: "--",
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
                error = "Offline mode: training will continue locally and sync is skipped.",
            ).also {
                _events.emit(
                    TraineeHomeEvent.NavigateToSession(
                        "offline-${System.currentTimeMillis()}"
                    )
                )
            }
            is NetworkResult.Loading -> Unit
        }
    }
}
