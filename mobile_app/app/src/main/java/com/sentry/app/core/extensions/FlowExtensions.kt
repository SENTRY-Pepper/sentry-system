package com.sentry.app.core.extensions

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.onStart
import timber.log.Timber

fun <T> Flow<T>.withLogging(tag: String): Flow<T> =
    this
        .onStart { Timber.d("[$tag] Flow started") }
        .catch { e ->
            Timber.e(e, "[$tag] Flow error")
            throw e
        }