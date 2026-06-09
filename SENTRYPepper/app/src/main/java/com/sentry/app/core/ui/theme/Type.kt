package com.sentry.app.core.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import com.sentry.app.R

val PhilosopherFont = FontFamily(
    Font(R.font.philosopher_regular, FontWeight.Normal),
    Font(R.font.philosopher_bold, FontWeight.Bold),
    Font(R.font.philosopher_italic, FontWeight.Normal, FontStyle.Italic),
    Font(R.font.philosopher_bold_italic, FontWeight.Bold, FontStyle.Italic),
)

val ItimFont = FontFamily(Font(R.font.itim_regular, FontWeight.Normal))


val Typography = Typography(
    displayLarge = TextStyle(
        fontFamily = PhilosopherFont,
        fontWeight = FontWeight.Bold,
        fontSize = 30.sp,
    ),
    displayMedium = TextStyle(
        fontFamily = PhilosopherFont,
        fontWeight = FontWeight.Bold,
        fontSize = 24.sp,
    ),
    titleLarge = TextStyle(
        fontFamily = PhilosopherFont,
        fontWeight = FontWeight.Bold,
        fontSize = 20.sp,
    ),
    titleMedium = TextStyle(
        fontFamily = PhilosopherFont,
        fontWeight = FontWeight.Bold,
        fontSize = 16.sp,
    ),
    bodyLarge = TextStyle(
        fontFamily = PhilosopherFont,
        fontWeight = FontWeight.Normal,
        fontSize = 16.sp,
    ),
    bodyMedium = TextStyle(
        fontFamily = PhilosopherFont,
        fontWeight = FontWeight.Normal,
        fontSize = 14.sp,
    ),
    labelSmall = TextStyle(
        fontFamily = PhilosopherFont,
        fontWeight = FontWeight.Normal,
        fontSize = 12.sp,
    ),
)