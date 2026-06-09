package com.sentry.app.di

import com.sentry.app.data.repository.QueryRepository
import com.sentry.app.data.repository.QueryRepositoryImpl
import com.sentry.app.data.repository.SessionRepository
import com.sentry.app.data.repository.SessionRepositoryImpl
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
abstract class SessionModule {

    @Binds
    @Singleton
    abstract fun bindSessionRepository(
        impl: SessionRepositoryImpl,
    ): SessionRepository

    @Binds
    @Singleton
    abstract fun bindQueryRepository(
        impl: QueryRepositoryImpl,
    ): QueryRepository
}