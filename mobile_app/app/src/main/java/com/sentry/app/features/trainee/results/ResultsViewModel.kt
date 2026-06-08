package com.sentry.app.features.trainee.results

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.models.response.SessionSummary
import com.sentry.app.data.repository.SessionRepository
import com.sentry.app.features.trainee.curriculum.OwaspCurriculum
import com.sentry.app.features.trainee.curriculum.TrainingProgressStore
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ResultsUiState(
    val sessionId: String = "",
    val loading: Boolean = false,
    val summary: SessionSummary? = null,
    val error: String = "",
    val correctCount: Int = 0,
    val totalCount: Int = OwaspCurriculum.totalModules,
    val postScore: Float = 0f,
    val durationSeconds: Int = 0,
    val missedModuleIds: List<String> = emptyList(),
)

@HiltViewModel
class ResultsViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val sessionRepository: SessionRepository,
    private val progressStore: TrainingProgressStore,
) : ViewModel() {

    private val sessionId: String = checkNotNull(savedStateHandle["sessionId"])

    private val _uiState = MutableStateFlow(ResultsUiState(sessionId = sessionId))
    val uiState = _uiState.asStateFlow()

    init { loadSummary() }

    private fun loadSummary() = viewModelScope.launch {
        _uiState.value = _uiState.value.copy(loading = true, error = "")

        progressStore.lastResult(sessionId)?.let { local ->
            _uiState.value = _uiState.value.copy(
                correctCount = local.correctCount,
                totalCount = local.totalCount,
                postScore = local.postScore,
                durationSeconds = local.durationSeconds,
                missedModuleIds = local.missedModuleIds,
            )
        }

        when (val result = sessionRepository.getSession(sessionId)) {
            is NetworkResult.Success -> _uiState.value = _uiState.value.copy(
                loading = false,
                summary = result.data,
                postScore = result.data.postAssessmentScore ?: _uiState.value.postScore,
                durationSeconds = result.data.durationSeconds ?: _uiState.value.durationSeconds,
            )
            is NetworkResult.Error -> _uiState.value = _uiState.value.copy(
                loading = false,
                error = result.message,
            )
            is NetworkResult.Exception -> _uiState.value = _uiState.value.copy(
                loading = false,
                error = result.e.message ?: "Loaded local result only",
            )
            is NetworkResult.Loading -> Unit
        }
    }

    fun moduleTitle(moduleId: String): String =
        OwaspCurriculum.findModule(moduleId)?.let { "${it.owaspId}: ${it.title}" }
            ?: moduleId

    fun formatDuration(seconds: Int): String {
        val m = seconds / 60
        val s = seconds % 60
        return "${m}m ${s}s"
    }
}
