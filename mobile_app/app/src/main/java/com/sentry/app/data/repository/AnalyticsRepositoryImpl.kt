package com.sentry.app.data.repository

import com.sentry.app.core.network.NetworkResult
import com.sentry.app.core.network.SentryKtorClient
import com.sentry.app.data.local.TokenManager
import com.sentry.app.data.models.request.UserCreateRequest
import com.sentry.app.data.models.response.ManagerOverviewAnalytics
import com.sentry.app.data.models.response.OrganisationAnalytics
import com.sentry.app.data.models.response.SessionSummary
import com.sentry.app.data.models.response.StudyAnalytics
import com.sentry.app.data.models.response.TraineeAnalytics
import com.sentry.app.data.models.response.UserResponse
import com.sentry.app.data.models.response.WeaknessAnalytics
import com.sentry.app.data.remote.api.createTrainee
import com.sentry.app.data.remote.api.deactivateTrainee
import com.sentry.app.data.remote.api.getOrganisationAnalytics
import com.sentry.app.data.remote.api.getManagerOverview
import com.sentry.app.data.remote.api.getManagerWeaknesses
import com.sentry.app.data.remote.api.getStudyAnalytics
import com.sentry.app.data.remote.api.listTrainees
import com.sentry.app.data.remote.api.listSessions
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class AnalyticsRepositoryImpl @Inject constructor(
    private val ktorClient: SentryKtorClient,
    private val tokenManager: TokenManager,
) : AnalyticsRepository {

    override suspend fun getStudyAnalytics(): NetworkResult<StudyAnalytics> =
        ktorClient.getStudyAnalytics()

    override suspend fun getOrganisationAnalytics(): NetworkResult<OrganisationAnalytics> {
        val orgId = tokenManager.getOrganisationId()
            ?: return NetworkResult.Error(message = "No organisation ID")
        return ktorClient.getOrganisationAnalytics(orgId)
    }

    override suspend fun listSessions(
        condition: String?,
        organisationId: String?,
        limit: Int,
    ): NetworkResult<List<SessionSummary>> {
        val role = tokenManager.getRole()
        val orgId = organisationId ?: if (role == "admin") null else tokenManager.getOrganisationId()
        return ktorClient.listSessions(
            condition = condition,
            organisationId = orgId,
            limit = limit,
        )
    }

    override suspend fun getManagerOverview(): NetworkResult<ManagerOverviewAnalytics> {
        val orgId = tokenManager.getOrganisationId()
            ?: return NetworkResult.Error(message = "No organisation ID")
        return ktorClient.getManagerOverview(orgId)
    }

    override suspend fun getManagerWeaknesses(
        limit: Int,
    ): NetworkResult<List<WeaknessAnalytics>> {
        val orgId = tokenManager.getOrganisationId()
            ?: return NetworkResult.Error(message = "No organisation ID")
        return ktorClient.getManagerWeaknesses(orgId, limit)
    }

    override suspend fun listTrainees(): NetworkResult<List<TraineeAnalytics>> {
        val orgId = tokenManager.getOrganisationId()
            ?: return NetworkResult.Error(message = "No organisation ID")
        return ktorClient.listTrainees(orgId)
    }

    override suspend fun createTrainee(
        request: UserCreateRequest,
    ): NetworkResult<UserResponse> {
        val orgId = tokenManager.getOrganisationId()
            ?: return NetworkResult.Error(message = "No organisation ID")
        return ktorClient.createTrainee(request.copy(organisationId = orgId))
    }

    override suspend fun deactivateTrainee(userId: String): NetworkResult<UserResponse> =
        ktorClient.deactivateTrainee(userId)
}
