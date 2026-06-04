package com.sentry.app.features.trainee.home

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sentry.app.R
import com.sentry.app.features.splash.SentryCyan
import com.sentry.app.ui.theme.PhilosopherFont

private val SentryGreen    = Color(0xFF4CAF50)
private val SentryGreenDk  = Color(0xFF388E3C)
private val SentryCyanDk   = Color(0xFF0097A7)
private val BackgroundGray = Color(0xFFF5F5F5)
private val CardBorder     = Color(0xFFE0E0E0)
private val TextPrimary    = Color(0xFF212121)
private val TextSecondary  = Color(0xFF757575)

data class ModuleProgress(
    val name: String,
    val progress: Float,
    val status: String,
    val statusColor: Color,
)

@Composable
fun TraineeHomeScreen(
    onStartSession: (String) -> Unit,
    onOpenChat: () -> Unit,
    onOpenSettings: () -> Unit,
    vm: TraineeHomeViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsStateWithLifecycle()

    LaunchedEffect(state.sessionStarted) {
        state.sessionStarted?.let { onStartSession(it) }
    }

    val modules = listOf(
        ModuleProgress("Phishing Detection",       1.0f, "Complete",    SentryGreen),
        ModuleProgress("USB Drop Simulation",      1.0f, "Complete",    SentryGreen),
        ModuleProgress("Password Hygiene",         0.4f, "In Progress", TextSecondary),
        ModuleProgress("Voice Social Engineering", 0.0f, "Not Started", TextSecondary),
        ModuleProgress("Network Hygiene",          0.0f, "Not Started", TextSecondary),
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White),
    ) {
        // ── Top bar ──────────────────────────────────────────────────
        TopBar(
            participantId = "EMP_042",
            onOpenChat    = onOpenChat,
            onOpenSettings = onOpenSettings,
        )

        // ── Body ─────────────────────────────────────────────────────
        Row(
            modifier = Modifier
                .fillMaxSize()
                .background(BackgroundGray)
                .padding(20.dp),
            horizontalArrangement = Arrangement.spacedBy(20.dp),
        ) {
            // Left column — progress
            Column(
                modifier = Modifier
                    .weight(1.4f)
                    .fillMaxHeight()
                    .verticalScroll(rememberScrollState()),
            ) {
                WelcomeHeader(participantId = "EMP_042")
                Spacer(Modifier.height(16.dp))
                ProgressCard(modules = modules)
            }

            // Right column — stats + CTA
            Column(
                modifier = Modifier
                    .weight(0.8f)
                    .fillMaxHeight()
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                StatCard(value = "2",   label = "Sessions Completed")
                StatCard(value = "78%", label = "Avg Accuracy")
                StatCard(value = "3",   label = "Modules Left")
                CtaCard(
                    loading  = state.loading,
                    error    = state.error,
                    onStart  = { vm.startSession() },
                )
            }
        }
    }
}

@Composable
private fun TopBar(
    participantId: String,
    onOpenChat: () -> Unit,
    onOpenSettings: () -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(68.dp)
            .background(SentryCyan),
    ) {
        // Left — avatar + name
        Row(
            modifier          = Modifier
                .align(Alignment.CenterStart)
                .padding(start = 16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                modifier         = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(Color.White.copy(alpha = 0.25f)),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text       = participantId.take(2).uppercase(),
                    fontSize   = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color      = Color.White,
                )
            }
            Spacer(Modifier.width(10.dp))
            Column {
                Text(
                    text       = participantId,
                    fontSize   = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color      = Color.White,
                )
                Text(
                    text     = "Trainee",
                    fontSize = 11.sp,
                    color    = Color.White.copy(alpha = 0.8f),
                )
            }
        }

        // Centre — SENTRY logo + Tap to ASK
        Column(
            modifier             = Modifier.align(Alignment.Center),
            horizontalAlignment  = Alignment.CenterHorizontally,
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Image(
                    painter            = painterResource(R.drawable.pepper_robot),
                    contentDescription = null,
                    modifier           = Modifier
                        .size(28.dp)
                        .clip(CircleShape)
                        .background(Color.White),
                    contentScale       = ContentScale.Crop,
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    text       = "SENTRY",
                    fontFamily = com.sentry.app.ui.theme.ItimFont,
                    fontSize   = 22.sp,
                    fontWeight = FontWeight.Bold,
                    color      = Color.White,
                    letterSpacing = 3.sp,
                )
            }
            // Tap to ASK button
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .background(Color.White.copy(alpha = 0.2f))
                    .clickable { onOpenChat() }
                    .padding(horizontal = 12.dp, vertical = 3.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    // Traffic light dots
                    listOf(Color(0xFF4CAF50), Color(0xFFFFC107), Color(0xFFF44336))
                        .forEach { colour ->
                            Box(
                                modifier = Modifier
                                    .size(8.dp)
                                    .clip(CircleShape)
                                    .background(colour),
                            )
                            Spacer(Modifier.width(4.dp))
                        }
                    Text(
                        text     = "Tap to ASK",
                        fontSize = 11.sp,
                        color    = Color.White,
                        fontWeight = FontWeight.Medium,
                    )
                }
            }
        }

        // Right — settings icon
        IconButton(
            onClick  = onOpenSettings,
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .padding(end = 8.dp),
        ) {
            Icon(
                imageVector        = Icons.Default.Settings,
                contentDescription = "Settings",
                tint               = Color.White,
                modifier           = Modifier.size(26.dp),
            )
        }
    }
}

@Composable
private fun WelcomeHeader(participantId: String) {
    Column {
        Text(
            text       = "Welcome back, $participantId",
            fontFamily = PhilosopherFont,
            fontSize   = 20.sp,
            fontWeight = FontWeight.Bold,
            color      = TextPrimary,
        )
        Text(
            text     = "Heritage Insurance  ·  ${currentDate()}",
            fontSize = 12.sp,
            color    = TextSecondary,
        )
    }
}

@Composable
private fun ProgressCard(modules: List<ModuleProgress>) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(Color.White)
            .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
            .padding(20.dp),
    ) {
        Column {
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment     = Alignment.CenterVertically,
            ) {
                Text(
                    text       = "Your Progress",
                    fontFamily = PhilosopherFont,
                    fontSize   = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color      = TextPrimary,
                )
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(20.dp))
                        .background(Color(0xFF9C27B0))
                        .padding(horizontal = 14.dp, vertical = 5.dp),
                ) {
                    Text(
                        text     = "40% Complete",
                        fontSize = 12.sp,
                        color    = Color.White,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }

            Spacer(Modifier.height(16.dp))

            modules.forEach { module ->
                ModuleRow(module = module)
                Spacer(Modifier.height(12.dp))
            }
        }
    }
}

@Composable
private fun ModuleRow(module: ModuleProgress) {
    Column {
        Row(
            modifier              = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment     = Alignment.CenterVertically,
        ) {
            Text(
                text     = module.name,
                fontSize = 13.sp,
                color    = TextPrimary,
                fontWeight = FontWeight.Medium,
            )
            Text(
                text     = module.status,
                fontSize = 12.sp,
                color    = module.statusColor,
            )
        }
        Spacer(Modifier.height(5.dp))
        LinearProgressIndicator(
            progress          = { module.progress },
            modifier          = Modifier
                .fillMaxWidth()
                .height(7.dp)
                .clip(RoundedCornerShape(4.dp)),
            color             = if (module.progress > 0f) SentryGreen else Color(0xFFBDBDBD),
            trackColor        = Color(0xFFE0E0E0),
            strokeCap         = StrokeCap.Round,
        )
    }
}

@Composable
private fun StatCard(value: String, label: String) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(Color.White)
            .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
            .padding(vertical = 20.dp, horizontal = 16.dp),
    ) {
        Column(horizontalAlignment = Alignment.Start) {
            Text(
                text       = value,
                fontFamily = PhilosopherFont,
                fontSize   = 32.sp,
                fontWeight = FontWeight.Bold,
                color      = TextPrimary,
            )
            Text(
                text     = label,
                fontSize = 13.sp,
                color    = TextSecondary,
            )
        }
    }
}

@Composable
private fun CtaCard(
    loading: Boolean,
    error: String,
    onStart: () -> Unit,
) {
    Column {
        // Continue training button
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(14.dp))
                .background(SentryCyan)
                .clickable { if (!loading) onStart() }
                .padding(vertical = 16.dp, horizontal = 12.dp),
            contentAlignment = Alignment.Center,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text       = if (loading) "Starting…" else "Continue Your Training",
                    fontFamily = PhilosopherFont,
                    fontSize   = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color      = Color.White,
                    textAlign  = TextAlign.Center,
                )
                if (!loading) {
                    Text(
                        text     = "Next: Password Hygiene",
                        fontSize = 11.sp,
                        color    = Color.White.copy(alpha = 0.85f),
                    )
                }
            }
        }

        if (error.isNotBlank()) {
            Spacer(Modifier.height(8.dp))
            Text(
                text     = error,
                fontSize = 11.sp,
                color    = Color(0xFFF44336),
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth(),
                fontFamily = PhilosopherFont
            )
        }
    }
}

private fun currentDate(): String {
    val cal = java.util.Calendar.getInstance()
    val days   = arrayOf("Sun","Mon","Tue","Wed","Thu","Fri","Sat")
    val months = arrayOf("Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec")
    val dow = days[cal.get(java.util.Calendar.DAY_OF_WEEK) - 1]
    val dom = cal.get(java.util.Calendar.DAY_OF_MONTH)
    val mon = months[cal.get(java.util.Calendar.MONTH)]
    val yr  = cal.get(java.util.Calendar.YEAR)
    return "$dow $dom $mon $yr"
}