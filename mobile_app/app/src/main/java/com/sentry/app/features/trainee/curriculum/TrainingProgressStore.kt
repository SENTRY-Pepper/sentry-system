package com.sentry.app.features.trainee.curriculum

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.core.stringSetPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.sentry.app.data.local.TokenManager
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.first
import javax.inject.Inject
import javax.inject.Singleton

private val Context.trainingProgressDataStore: DataStore<Preferences>
        by preferencesDataStore(name = "sentry_training_progress")

data class LocalSessionResult(
    val sessionId: String,
    val correctCount: Int,
    val totalCount: Int,
    val postScore: Float,
    val durationSeconds: Int,
    val missedModuleIds: List<String>,
)

data class ModuleResult(
    val moduleId: String,
    val correctCount: Int,
    val totalCount: Int,
) {
    val progress: Float
        get() = if (totalCount > 0) 1f else 0f

    val scorePercent: Int
        get() = if (totalCount > 0) {
            ((correctCount.toFloat() / totalCount.toFloat()) * 100f).toInt()
        } else {
            0
        }
}

@Singleton
class TrainingProgressStore @Inject constructor(
    @param:ApplicationContext private val context: Context,
    private val tokenManager: TokenManager,
) {
    suspend fun completedModuleIds(): Set<String> {
        val prefs = context.trainingProgressDataStore.data.first()
        return prefs[keys().completedModules] ?: emptySet()
    }

    suspend fun sessionsCompleted(): Int {
        val prefs = context.trainingProgressDataStore.data.first()
        return prefs[keys().sessionsCompleted] ?: 0
    }

    suspend fun averageAccuracy(): Float? {
        val prefs = context.trainingProgressDataStore.data.first()
        return prefs[keys().averageAccuracy]
    }

    suspend fun activeSessionId(): String? {
        val prefs = context.trainingProgressDataStore.data.first()
        return prefs[keys().activeSessionId]
    }

    suspend fun lastModuleId(): String? {
        val prefs = context.trainingProgressDataStore.data.first()
        return prefs[keys().lastModuleId]
    }

    suspend fun moduleResults(): Map<String, ModuleResult> {
        val prefs = context.trainingProgressDataStore.data.first()
        return decodeModuleResults(prefs[keys().moduleResults])
    }

    suspend fun setActiveSession(sessionId: String) {
        val keys = keys()
        context.trainingProgressDataStore.edit { prefs ->
            prefs[keys.activeSessionId] = sessionId
        }
    }

    suspend fun recordModuleResult(moduleId: String, correctCount: Int, totalCount: Int) {
        val keys = keys()
        val current = moduleResults().toMutableMap()
        current[moduleId] = ModuleResult(
            moduleId = moduleId,
            correctCount = correctCount,
            totalCount = totalCount,
        )
        context.trainingProgressDataStore.edit { prefs ->
            prefs[keys.moduleResults] = encodeModuleResults(current.values)
            prefs[keys.lastModuleId] = moduleId
            prefs[keys.completedModules] = (prefs[keys.completedModules] ?: emptySet()) + moduleId
        }
    }

    suspend fun lastResult(sessionId: String): LocalSessionResult? {
        val keys = keys()
        val prefs = context.trainingProgressDataStore.data.first()
        if (prefs[keys.lastSessionId] != sessionId) return null

        val total = prefs[keys.lastTotal] ?: return null
        val correct = prefs[keys.lastCorrect] ?: 0
        val postScore = prefs[keys.lastPostScore] ?: 0f
        val duration = prefs[keys.lastDuration] ?: 0
        val missed = prefs[keys.lastMissed]
            ?.split(",")
            ?.filter { it.isNotBlank() }
            ?: emptyList()

        return LocalSessionResult(
            sessionId = sessionId,
            correctCount = correct,
            totalCount = total,
            postScore = postScore,
            durationSeconds = duration,
            missedModuleIds = missed,
        )
    }

    suspend fun recordSession(result: LocalSessionResult, completedModuleIds: Set<String>) {
        val keys = keys()
        val currentSessions = sessionsCompleted()
        val currentAverage = averageAccuracy()
        val prefsSnapshot = context.trainingProgressDataStore.data.first()
        val isSameSession = prefsSnapshot[keys.lastSessionId] == result.sessionId
        val newSessionCount = if (isSameSession) currentSessions else currentSessions + 1
        val newAverage = when {
            isSameSession -> currentAverage ?: result.postScore
            currentAverage == null -> result.postScore
            else -> ((currentAverage * currentSessions) + result.postScore) / newSessionCount
        }

        context.trainingProgressDataStore.edit { prefs ->
            prefs[keys.completedModules] = completedModuleIds
            prefs[keys.sessionsCompleted] = newSessionCount
            prefs[keys.averageAccuracy] = newAverage
            prefs[keys.lastSessionId] = result.sessionId
            prefs[keys.lastCorrect] = result.correctCount
            prefs[keys.lastTotal] = result.totalCount
            prefs[keys.lastPostScore] = result.postScore
            prefs[keys.lastDuration] = result.durationSeconds
            prefs[keys.lastMissed] = result.missedModuleIds.joinToString(",")
        }
    }

    private fun keys(): ProgressKeys {
        val participant = tokenManager.getParticipantId()
            ?.ifBlank { null }
            ?: "anonymous"
        val organisation = tokenManager.getOrganisationId()
            ?.ifBlank { null }
            ?: "no_org"
        val owner = "${participant}_$organisation"
            .lowercase()
            .replace(Regex("[^a-z0-9_]+"), "_")
            .trim('_')
            .ifBlank { "anonymous" }
        return ProgressKeys(owner)
    }

    private data class ProgressKeys(
        val owner: String,
    ) {
        val completedModules = stringSetPreferencesKey("${owner}_completed_modules")
        val sessionsCompleted = intPreferencesKey("${owner}_sessions_completed")
        val averageAccuracy = floatPreferencesKey("${owner}_average_accuracy")
        val lastSessionId = stringPreferencesKey("${owner}_last_session_id")
        val lastCorrect = intPreferencesKey("${owner}_last_correct")
        val lastTotal = intPreferencesKey("${owner}_last_total")
        val lastPostScore = floatPreferencesKey("${owner}_last_post_score")
        val lastDuration = intPreferencesKey("${owner}_last_duration")
        val lastMissed = stringPreferencesKey("${owner}_last_missed")
        val activeSessionId = stringPreferencesKey("${owner}_active_session_id")
        val lastModuleId = stringPreferencesKey("${owner}_last_module_id")
        val moduleResults = stringPreferencesKey("${owner}_module_results")
    }

    private fun decodeModuleResults(raw: String?): Map<String, ModuleResult> =
        raw
            ?.split(";")
            ?.mapNotNull { entry ->
                val parts = entry.split(":")
                if (parts.size != 3) return@mapNotNull null
                val correct = parts[1].toIntOrNull() ?: return@mapNotNull null
                val total = parts[2].toIntOrNull() ?: return@mapNotNull null
                ModuleResult(parts[0], correct, total)
            }
            ?.associateBy { it.moduleId }
            ?: emptyMap()

    private fun encodeModuleResults(results: Collection<ModuleResult>): String =
        results.joinToString(";") { "${it.moduleId}:${it.correctCount}:${it.totalCount}" }
}
