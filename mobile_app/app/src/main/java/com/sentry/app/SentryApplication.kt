package com.sentry.app

import android.app.Application
import dagger.hilt.android.HiltAndroidApp
import timber.log.Timber

/**
 * Application entry point.
 * @HiltAndroidApp triggers Hilt's code generation and sets up
 * the dependency injection graph for the entire app.
 */

@HiltAndroidApp
class SentryApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // Timber logging — debug builds only.
        // ProGuard strips Timber.d/v/i calls from release APK.
        if (BuildConfig.DEBUG) {
            Timber.plant(Timber.DebugTree())
        }
    }
}