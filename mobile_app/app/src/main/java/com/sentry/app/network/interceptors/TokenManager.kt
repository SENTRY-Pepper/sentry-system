package com.sentry.app.network.interceptors

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
    @ApplicationContext private val context: Context,
) {
    companion object {
        private val KEY_TOKEN           = stringPreferencesKey("auth_token")
        private val KEY_PARTICIPANT_ID  = stringPreferencesKey("participant_id")
        private val KEY_ROLE            = stringPreferencesKey("user_role")
        private val KEY_ORGANISATION_ID = stringPreferencesKey("organisation_id")
    }

    fun getToken(): String? = runBlocking {
        context.authDataStore.data.map { it[KEY_TOKEN] }.first()
    }
    fun getRole(): String? = runBlocking {
        context.authDataStore.data.map { it[KEY_ROLE] }.first()
    }
    fun getParticipantId(): String? = runBlocking {
        context.authDataStore.data.map { it[KEY_PARTICIPANT_ID] }.first()
    }
    fun getOrganisationId(): String? = runBlocking {
        context.authDataStore.data.map { it[KEY_ORGANISATION_ID] }.first()
    }
    fun isAuthenticated(): Boolean = getToken() != null

    val tokenFlow: Flow<String?> = context.authDataStore.data.map { it[KEY_TOKEN] }
    val roleFlow:  Flow<String?> = context.authDataStore.data.map { it[KEY_ROLE] }

    suspend fun saveCredentials(
        token: String, participantId: String, role: String, organisationId: String,
    ) {
        Timber.d("TokenManager: saving credentials for $participantId ($role)")
        context.authDataStore.edit { prefs ->
            prefs[KEY_TOKEN]           = token
            prefs[KEY_PARTICIPANT_ID]  = participantId
            prefs[KEY_ROLE]            = role
            prefs[KEY_ORGANISATION_ID] = organisationId
        }
    }

    suspend fun clearToken() {
        Timber.d("TokenManager: clearing credentials")
        context.authDataStore.edit { it.clear() }
    }
}
