package com.sentry.app.di

import com.sentry.app.data.local.TokenManager
import com.sentry.app.data.repository.*
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object RepositoryModule {

    @Provides
    @Singleton
    fun provideAuthRepository(tm: TokenManager) = AuthRepository(tm)

}