package com.sentry.app.core.ui.components.texts

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.TextUnit
import com.sentry.app.core.ui.constants.sentryTextDisplay
import com.sentry.app.core.ui.constants.sentryTextHero
import com.sentry.app.core.ui.constants.sentryTextLg
import com.sentry.app.core.ui.constants.sentryTextMd
import com.sentry.app.core.ui.constants.sentryTextSm
import com.sentry.app.core.ui.constants.sentryTextXl
import com.sentry.app.core.ui.constants.sentryTextXs
import com.sentry.app.core.ui.models.SentryTextAlign
import com.sentry.app.core.ui.models.SentryTextSize
import com.sentry.app.core.ui.theme.PhilosopherFont


@Composable
fun SentryText(
    text: Any,
    modifier: Modifier = Modifier,
    size: SentryTextSize = SentryTextSize.Md,
    align: SentryTextAlign = SentryTextAlign.Start,
    color: Color = MaterialTheme.colorScheme.onBackground,
    weight: FontWeight = FontWeight.Normal,
    maxLines: Int = Int.MAX_VALUE,
    overflow: TextOverflow = TextOverflow.Ellipsis,
    lineHeight: TextUnit = TextUnit.Unspecified,
) {
    val fontSize = when (size) {
        SentryTextSize.Xs -> sentryTextXs
        SentryTextSize.Sm -> sentryTextSm
        SentryTextSize.Md -> sentryTextMd
        SentryTextSize.Lg -> sentryTextLg
        SentryTextSize.Xl -> sentryTextXl
        SentryTextSize.Display -> sentryTextDisplay
        SentryTextSize.Hero -> sentryTextHero
    }

    val textAlign = when (align) {
        SentryTextAlign.Start -> TextAlign.Start
        SentryTextAlign.Center -> TextAlign.Center
        SentryTextAlign.End -> TextAlign.End
    }

    val style = TextStyle(
        fontFamily = PhilosopherFont,
        fontWeight = weight,
        fontSize = fontSize,
        textAlign = textAlign,
        lineHeight = lineHeight,
    )

    when (text) {
        is AnnotatedString -> Text(
            text = text,
            modifier = modifier,
            color = color,
            style = style,
            maxLines = maxLines,
            overflow = overflow,
        )

        is String -> Text(
            text = text,
            modifier = modifier,
            color = color,
            style = style,
            maxLines = maxLines,
            overflow = overflow,
        )

        else -> error("SentryText only supports String or AnnotatedString")
    }
}