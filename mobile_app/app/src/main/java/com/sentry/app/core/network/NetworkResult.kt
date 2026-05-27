package com.sentry.app.core.network

sealed class NetworkResult<out T> {
    data class Success<T>(val data: T)                    : NetworkResult<T>()
    data class Error(val code: Int, val message: String)  : NetworkResult<Nothing>()
    data class Exception(val e: Throwable)                : NetworkResult<Nothing>()
    data object Loading                                   : NetworkResult<Nothing>()

    val isSuccess get() = this is Success
    val isError   get() = this is Error || this is Exception
}