package com.sentry.app.features.splash

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import com.sentry.app.R
import com.sentry.app.core.navigation.UserRole
import com.sentry.app.ui.theme.ItimFont
import com.sentry.app.ui.theme.PhilosopherFont
import kotlinx.coroutines.delay

val SentryCyan = Color(0xFF00BCD4)

@Composable
fun SplashScreen(
    onAuthenticated: (UserRole) -> Unit,
    onUnauthenticated: () -> Unit,
    vm: SplashViewModel = hiltViewModel(),
) {
    val alpha = remember { Animatable(0f) }

    LaunchedEffect(Unit) {
        alpha.animateTo(1f, animationSpec = tween(600))
        delay(2500)
        if (vm.isAuthenticated()) onAuthenticated(vm.getRole())
        else onUnauthenticated()
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(SentryCyan),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            modifier = Modifier
                .alpha(alpha.value)
                .padding(horizontal = 48.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            // Pepper robot in white rounded card
            Box(
                modifier = Modifier
                    .size(width = 220.dp, height = 160.dp)
                    .clip(RoundedCornerShape(20.dp))
                    .background(Color.White),
                contentAlignment = Alignment.Center,
            ) {
                Image(
                    painter = painterResource(id = R.drawable.pepper_robot),
                    contentDescription = "Pepper robot — SENTRY cybersecurity tutor",
                    modifier = Modifier
                        .size(width = 200.dp, height = 145.dp),
                    contentScale = ContentScale.Fit,
                )
            }

            Spacer(Modifier.height(24.dp))

            // SENTRY wordmark — Itim font
            Text(
                text = "SENTRY",
                fontFamily = ItimFont,
                fontSize = 42.sp,
                fontWeight = FontWeight.Normal,
                color = Color.White,
                letterSpacing = 6.sp,
            )

            Spacer(Modifier.height(10.dp))

            // Subtitle — Philosopher font
            Text(
                text = "Your personalised grounded\nCybersecurity AI tutor",
                fontFamily = PhilosopherFont,
                fontSize = 15.sp,
                fontWeight = FontWeight.Normal,
                color = Color.White.copy(alpha = 0.85f),
                textAlign = TextAlign.Center,
                lineHeight = 22.sp,
            )

            Spacer(Modifier.height(48.dp))

            LoadingBar(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 40.dp),
            )
        }
    }
}

@Composable
private fun LoadingBar(modifier: Modifier = Modifier) {
    val progress = remember { Animatable(0f) }

    LaunchedEffect(Unit) {
        progress.animateTo(
            targetValue = 1f,
            animationSpec = tween(durationMillis = 2200, delayMillis = 400),
        )
    }

    Box(
        modifier = modifier
            .height(3.dp)
            .clip(RoundedCornerShape(2.dp))
            .background(Color.White.copy(alpha = 0.3f)),
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth(progress.value)
                .height(3.dp)
                .clip(RoundedCornerShape(2.dp))
                .background(Color.White),
        )
    }
}