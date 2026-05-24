package com.sentry.app.di

import android.content.Context
import com.sentry.app.network.interceptors.TokenManager
import com.sentry.app.repository.AuthRepository
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides @Singleton
    fun provideTokenManager(@ApplicationContext context: Context): TokenManager =
        TokenManager(context)

    @Provides @Singleton
    fun provideAuthRepository(tokenManager: TokenManager): AuthRepository =
        AuthRepository(tokenManager)
}
