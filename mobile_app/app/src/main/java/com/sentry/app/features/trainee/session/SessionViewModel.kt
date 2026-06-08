package com.sentry.app.features.trainee.session

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sentry.app.core.network.NetworkResult
import com.sentry.app.data.repository.SessionRepository
import com.sentry.app.features.trainee.curriculum.LocalSessionResult
import com.sentry.app.features.trainee.curriculum.OwaspCurriculum
import com.sentry.app.features.trainee.curriculum.OwaspQuestion
import com.sentry.app.features.trainee.curriculum.OwaspTrainingModule
import com.sentry.app.features.trainee.curriculum.TrainingProgressStore
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class SessionUiState(
    val sessionId: String = "",
    val currentModuleIndex: Int = 0,
    val currentQuestionIndex: Int = 0,
    val totalModules: Int = OwaspCurriculum.totalModules,
    val totalQuestions: Int = OwaspCurriculum.totalQuestions,
    val selectedChoiceId: String? = null,
    val isAnswered: Boolean = false,
    val isCorrect: Boolean = false,
    val isModuleBreak: Boolean = false,
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
            totalModules = OwaspCurriculum.modules.size,
            totalQuestions = OwaspCurriculum.totalQuestions,
        )
    )
    val uiState = _uiState.asStateFlow()

    fun getCurrentScenario(): OwaspTrainingModule =
        OwaspCurriculum.modules[_uiState.value.currentModuleIndex]

    fun getCurrentQuestion(): OwaspQuestion =
        getCurrentScenario().questions[_uiState.value.currentQuestionIndex]

    fun selectChoice(choiceId: String) {
        val state = _uiState.value
        if (state.isAnswered || state.isFinishing || state.isModuleBreak) return

        val module = getCurrentScenario()
        val question = getCurrentQuestion()
        val choice = question.options.first { it.id == choiceId }
        val sources = listOf(module.sourceReference)

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
            scenarioId = question.id,
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

        if (state.isModuleBreak) {
            val nextModuleIndex = state.currentModuleIndex + 1
            if (nextModuleIndex >= OwaspCurriculum.modules.size) {
                finishSession()
            } else {
                _uiState.value = state.copy(
                    currentModuleIndex = nextModuleIndex,
                    currentQuestionIndex = 0,
                    selectedChoiceId = null,
                    isAnswered = false,
                    isCorrect = false,
                    isModuleBreak = false,
                    aiResponse = "",
                    aiSources = emptyList(),
                    aiLoading = false,
                    error = "",
                )
            }
            return
        }

        val module = getCurrentScenario()
        val nextQuestionIndex = state.currentQuestionIndex + 1
        if (nextQuestionIndex >= module.questions.size) {
            answeredModuleIds.add(module.id)
            _uiState.value = state.copy(
                selectedChoiceId = null,
                isAnswered = false,
                isCorrect = false,
                isModuleBreak = true,
                aiResponse = "",
                aiSources = emptyList(),
                aiLoading = false,
                error = "",
            )
        } else {
            _uiState.value = state.copy(
                currentQuestionIndex = nextQuestionIndex,
                selectedChoiceId = null,
                isAnswered = false,
                isCorrect = false,
                isModuleBreak = false,
                aiResponse = "",
                aiSources = emptyList(),
                aiLoading = false,
                error = "",
            )
        }
    }

    private fun finishSession() {
        val state = _uiState.value
        val total = OwaspCurriculum.totalQuestions
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
