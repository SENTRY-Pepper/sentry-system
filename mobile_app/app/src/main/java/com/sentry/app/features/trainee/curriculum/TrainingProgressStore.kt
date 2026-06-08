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

@Singleton
class TrainingProgressStore @Inject constructor(
    @param:ApplicationContext private val context: Context,
) {
    companion object {
        private val KEY_COMPLETED_MODULES = stringSetPreferencesKey("completed_modules")
        private val KEY_SESSIONS_COMPLETED = intPreferencesKey("sessions_completed")
        private val KEY_AVERAGE_ACCURACY = floatPreferencesKey("average_accuracy")
        private val KEY_LAST_SESSION_ID = stringPreferencesKey("last_session_id")
        private val KEY_LAST_CORRECT = intPreferencesKey("last_correct")
        private val KEY_LAST_TOTAL = intPreferencesKey("last_total")
        private val KEY_LAST_POST_SCORE = floatPreferencesKey("last_post_score")
        private val KEY_LAST_DURATION = intPreferencesKey("last_duration")
        private val KEY_LAST_MISSED = stringPreferencesKey("last_missed")
    }

    suspend fun completedModuleIds(): Set<String> {
        val prefs = context.trainingProgressDataStore.data.first()
        return prefs[KEY_COMPLETED_MODULES] ?: emptySet()
    }

    suspend fun sessionsCompleted(): Int {
        val prefs = context.trainingProgressDataStore.data.first()
        return prefs[KEY_SESSIONS_COMPLETED] ?: 0
    }

    suspend fun averageAccuracy(): Float? {
        val prefs = context.trainingProgressDataStore.data.first()
        return prefs[KEY_AVERAGE_ACCURACY]
    }

    suspend fun lastResult(sessionId: String): LocalSessionResult? {
        val prefs = context.trainingProgressDataStore.data.first()
        if (prefs[KEY_LAST_SESSION_ID] != sessionId) return null

        val total = prefs[KEY_LAST_TOTAL] ?: return null
        val correct = prefs[KEY_LAST_CORRECT] ?: 0
        val postScore = prefs[KEY_LAST_POST_SCORE] ?: 0f
        val duration = prefs[KEY_LAST_DURATION] ?: 0
        val missed = prefs[KEY_LAST_MISSED]
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
        val currentSessions = sessionsCompleted()
        val currentAverage = averageAccuracy()
        val newSessionCount = currentSessions + 1
        val newAverage = if (currentAverage == null) {
            result.postScore
        } else {
            ((currentAverage * currentSessions) + result.postScore) / newSessionCount
        }

        context.trainingProgressDataStore.edit { prefs ->
            prefs[KEY_COMPLETED_MODULES] = completedModuleIds
            prefs[KEY_SESSIONS_COMPLETED] = newSessionCount
            prefs[KEY_AVERAGE_ACCURACY] = newAverage
            prefs[KEY_LAST_SESSION_ID] = result.sessionId
            prefs[KEY_LAST_CORRECT] = result.correctCount
            prefs[KEY_LAST_TOTAL] = result.totalCount
            prefs[KEY_LAST_POST_SCORE] = result.postScore
            prefs[KEY_LAST_DURATION] = result.durationSeconds
            prefs[KEY_LAST_MISSED] = result.missedModuleIds.joinToString(",")
        }
    }
}
