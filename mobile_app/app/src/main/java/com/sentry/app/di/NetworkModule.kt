package com.sentry.app.di

import android.content.Context
import com.sentry.app.BuildConfig
import com.sentry.app.core.network.SentryKtorClient
import com.sentry.app.data.local.TokenManager
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    @Provides
    @Singleton
    fun provideTokenManager(
        @ApplicationContext context: Context,
    ): TokenManager = TokenManager(context)

    @Provides
    @Singleton
    fun provideSentryKtorClient(
        tokenManager: TokenManager,
    ): SentryKtorClient = SentryKtorClient(
        tokenManager = tokenManager,
        baseUrl       = BuildConfig.MIDDLEWARE_BASE_URL,
    )
}
