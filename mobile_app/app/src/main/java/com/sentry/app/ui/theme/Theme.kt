package com.sentry.app.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val SentryColorScheme = lightColorScheme(
    primary            = SentryCyan,
    onPrimary          = White,
    primaryContainer   = SentryCyanLt,
    onPrimaryContainer = SentryCyanDk,
    secondary          = Teal500,
    onSecondary        = White,
    surface            = White,
    onSurface          = Neutral900,
    background         = Neutral50,
    onBackground       = Neutral900,
    error              = Red500,
    onError            = White,
)

@Composable
fun SentryTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = SentryColorScheme,
        typography  = SentryTypography,
        content     = content,
    )
}