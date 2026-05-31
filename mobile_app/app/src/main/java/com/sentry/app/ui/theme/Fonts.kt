package com.sentry.app.ui.theme

import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.googlefonts.Font
import androidx.compose.ui.text.googlefonts.GoogleFont

private val provider = GoogleFont.Provider(
    providerAuthority = "com.google.android.gms.fonts",
    providerPackage   = "com.google.android.gms",
    certificates      = com.sentry.app.R.array.com_google_android_gms_fonts_certs,
)

val ItimFont = FontFamily(
    Font(
        googleFont     = GoogleFont("Itim"),
        fontProvider   = provider,
        weight         = FontWeight.Normal,
    )
)

val PhilosopherFont = FontFamily(
    Font(
        googleFont     = GoogleFont("Philosopher"),
        fontProvider   = provider,
        weight         = FontWeight.Normal,
    ),
    Font(
        googleFont     = GoogleFont("Philosopher"),
        fontProvider   = provider,
        weight         = FontWeight.Bold,
    )
)