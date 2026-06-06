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
import androidx.compose.material3.MaterialTheme
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
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavHostController
import com.sentry.app.R
import com.sentry.app.core.navigation.UserRole
import com.sentry.app.core.navigation.navigateAndClear
import com.sentry.app.core.ui.components.texts.SentryText
import com.sentry.app.core.ui.models.SentryTextAlign
import com.sentry.app.core.ui.models.SentryTextSize
import com.sentry.app.core.ui.theme.PhilosopherFont
import kotlinx.coroutines.delay

@Composable
fun SplashScreen(
    navController: NavHostController,
    vm: SplashViewModel = hiltViewModel(),
) {
    val scheme = MaterialTheme.colorScheme
    val alpha = remember { Animatable(0f) }

    LaunchedEffect(Unit) {
        alpha.animateTo(1f, animationSpec = tween(600))
        delay(2500)
        if (vm.isAuthenticated()) {
            val route = if (vm.getRole() == UserRole.ADMIN) "adminHome" else "traineeHome"
            navController.navigateAndClear(route, popUpToRoute = "splash")
        } else {
            navController.navigateAndClear("auth", popUpToRoute = "splash")
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(scheme.primary),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            modifier = Modifier
                .alpha(alpha.value)
                .padding(horizontal = 48.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            // pepper robot card
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
                    modifier = Modifier.size(width = 200.dp, height = 145.dp),
                    contentScale = ContentScale.Fit,
                )
            }

            Spacer(Modifier.height(24.dp))

            // SENTRY wordmark
            SentryText(
                text = "SENTRY",
                size = SentryTextSize.Hero,
                weight = FontWeight.Bold,
                color = Color.White,
                align = SentryTextAlign.Center,
                fontFamily = PhilosopherFont
            )

            Spacer(Modifier.height(10.dp))

            // subtitle
            SentryText(
                text = "Your personalised grounded\nCybersecurity AI tutor",
                size = SentryTextSize.Md,
                color = Color.White.copy(alpha = 0.85f),
                align = SentryTextAlign.Center,
                maxLines = 3,
            )

            Spacer(Modifier.height(48.dp))

            /*  LoadingBar(
                  modifier = Modifier
                      .fillMaxWidth()
                      .padding(horizontal = 40.dp),
              )*/
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