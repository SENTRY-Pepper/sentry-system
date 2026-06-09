package com.sentry.app.data.local

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.runBlocking
import timber.log.Timber
import javax.inject.Inject
import javax.inject.Singleton

private val Context.authDataStore: DataStore<Preferences>
        by preferencesDataStore(name = "sentry_auth")

@Singleton
class TokenManager @Inject constructor(
    @param:ApplicationContext private val context: Context,
) {
    companion object {
        private val KEY_TOKEN = stringPreferencesKey("auth_token")
        private val KEY_ID = stringPreferencesKey("participant_id")
        private val KEY_ROLE = stringPreferencesKey("user_role")
        private val KEY_ORG = stringPreferencesKey("organisation_id")
    }

    // Synchronous reads used by OkHttp interceptor (runs off main thread)
    fun getToken(): String? = runBlocking { dataFlow(KEY_TOKEN).first() }
    fun getRole(): String? = runBlocking { dataFlow(KEY_ROLE).first() }
    fun getParticipantId(): String? = runBlocking { dataFlow(KEY_ID).first() }
    fun getOrganisationId(): String? = runBlocking { dataFlow(KEY_ORG).first() }
    fun isAuthenticated(): Boolean = getToken() != null

    // Blocking clear for use from interceptor thread
    fun clearTokenBlocking() = runBlocking { clearToken() }

    // Reactive flows for ViewModels
    val tokenFlow: Flow<String?> = dataFlow(KEY_TOKEN)
    val roleFlow: Flow<String?> = dataFlow(KEY_ROLE)

    // Suspend writes — called from coroutine context only
    suspend fun saveSession(
        token: String,
        participantId: String,
        role: String,
        organisationId: String,
    ) {
        Timber.d("TokenManager: saving session for $participantId ($role)")
        context.authDataStore.edit {
            it[KEY_TOKEN] = token
            it[KEY_ID] = participantId
            it[KEY_ROLE] = role
            it[KEY_ORG] = organisationId
        }
    }

    suspend fun clearToken() {
        Timber.d("TokenManager: clearing session")
        context.authDataStore.edit { it.clear() }
    }

    private fun <T> dataFlow(key: Preferences.Key<T>): Flow<T?> =
        context.authDataStore.data.map { it[key] }
}
