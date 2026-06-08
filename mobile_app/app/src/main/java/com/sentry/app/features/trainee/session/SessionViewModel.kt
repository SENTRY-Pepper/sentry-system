package com.sentry.app.features.trainee.session

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.repository.SessionRepository
import com.sentry.app.features.trainee.curriculum.LocalSessionResult
import com.sentry.app.features.trainee.curriculum.OwaspCurriculum
import com.sentry.app.features.trainee.curriculum.OwaspTrainingModule
import com.sentry.app.features.trainee.curriculum.TrainingProgressStore
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class SessionUiState(
    val sessionId: String = "",
    val currentIndex: Int = 0,
    val totalScenarios: Int = OwaspCurriculum.totalModules,
    val selectedChoiceId: String? = null,
    val isAnswered: Boolean = false,
    val isCorrect: Boolean = false,
    val aiResponse: String = "",
    val aiSources: List<String> = emptyList(),
    val aiLoading: Boolean = false,
    val isFinishing: Boolean = false,
    val isComplete: Boolean = false,
    val correctCount: Int = 0,
    val error: String = "",
)

@HiltViewModel
class SessionViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val sessionRepository: SessionRepository,
    private val progressStore: TrainingProgressStore,
) : ViewModel() {

    private val sessionId: String = checkNotNull(savedStateHandle["sessionId"])
    private val sessionStartedAtMs: Long = System.currentTimeMillis()
    private val answeredModuleIds = linkedSetOf<String>()
    private val missedModuleIds = linkedSetOf<String>()

    private val _uiState = MutableStateFlow(
        SessionUiState(
            sessionId = sessionId,
            totalScenarios = OwaspCurriculum.modules.size,
        )
    )
    val uiState = _uiState.asStateFlow()

    fun getCurrentScenario(): OwaspTrainingModule =
        OwaspCurriculum.modules[_uiState.value.currentIndex]

    fun selectChoice(choiceId: String) {
        val state = _uiState.value
        if (state.isAnswered || state.isFinishing) return

        val module = getCurrentScenario()
        val choice = module.options.first { it.id == choiceId }
        val sources = listOf(module.sourceReference)

        answeredModuleIds.add(module.id)
        if (!choice.isCorrect) {
            missedModuleIds.add(module.id)
        }

        _uiState.value = state.copy(
            selectedChoiceId = choiceId,
            isAnswered = true,
            isCorrect = choice.isCorrect,
            aiResponse = choice.feedback,
            aiSources = sources,
            aiLoading = false,
            correctCount = state.correctCount + if (choice.isCorrect) 1 else 0,
        )

        sessionRepository.logInteraction(
            sessionId = sessionId,
            scenarioId = module.id,
            scenarioType = module.owaspId,
            decision = if (choice.isCorrect) "correct" else "risky",
            employeeResponse = "${choice.label}. ${choice.text}",
            responseTimeMs = null,
            correctionLoops = 0,
            aiLatencyMs = null,
            aiSources = sources.joinToString(","),
        )
    }

    fun nextScenario() {
        val state = _uiState.value
        if (state.isFinishing) return

        val nextIndex = state.currentIndex + 1
        if (nextIndex >= OwaspCurriculum.modules.size) {
            finishSession()
        } else {
            _uiState.value = state.copy(
                currentIndex = nextIndex,
                selectedChoiceId = null,
                isAnswered = false,
                isCorrect = false,
                aiResponse = "",
                aiSources = emptyList(),
                aiLoading = false,
                error = "",
            )
        }
    }

    private fun finishSession() {
        val state = _uiState.value
        val total = OwaspCurriculum.modules.size
        val score = if (total > 0) {
            (state.correctCount.toFloat() / total.toFloat()) * 100f
        } else {
            0f
        }
        val durationSeconds = ((System.currentTimeMillis() - sessionStartedAtMs) / 1000)
            .toInt()
            .coerceAtLeast(0)

        _uiState.value = state.copy(isFinishing = true, aiLoading = true, error = "")

        viewModelScope.launch {
            val endResult = sessionRepository.endSession(
                sessionId = sessionId,
                preAssessmentScore = null,
                postAssessmentScore = score,
                durationSeconds = durationSeconds,
            )

            val completedModules = progressStore.completedModuleIds() + answeredModuleIds
            progressStore.recordSession(
                result = LocalSessionResult(
                    sessionId = sessionId,
                    correctCount = state.correctCount,
                    totalCount = total,
                    postScore = score,
                    durationSeconds = durationSeconds,
                    missedModuleIds = missedModuleIds.toList(),
                ),
                completedModuleIds = completedModules,
            )

            val endError = when (endResult) {
                is NetworkResult.Success -> ""
                is NetworkResult.Error -> endResult.message
                is NetworkResult.Exception -> endResult.e.message ?: "Session saved locally only"
                is NetworkResult.Loading -> ""
            }

            _uiState.value = _uiState.value.copy(
                aiLoading = false,
                isFinishing = false,
                isComplete = true,
                error = endError,
            )
        }
    }
}
