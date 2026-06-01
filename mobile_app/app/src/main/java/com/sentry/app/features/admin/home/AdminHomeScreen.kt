package com.sentry.app.features.admin.home

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
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sentry.app.R
import com.sentry.app.features.splash.SentryCyan
import com.sentry.app.ui.theme.ItimFont
import com.sentry.app.ui.theme.PhilosopherFont

private val BackgroundGray = Color(0xFFF5F5F5)
private val CardBorder     = Color(0xFFE0E0E0)
private val TextPrimary    = Color(0xFF212121)
private val TextSecondary  = Color(0xFF757575)
private val CorrectGreen   = Color(0xFF4CAF50)
private val WarnAmber      = Color(0xFFFF9800)
private val DangerRed      = Color(0xFFF44336)

private data class VulnerabilityItem(
    val name: String,
    val accuracy: Float,
    val color: Color,
)

private data class SessionRow(
    val participantId: String,
    val condition: String,
    val score: String,
    val gain: String,
    val status: String,
    val statusColor: Color,
    val statusBg: Color,
)

@Composable
fun AdminHomeScreen(
    onOpenAnalytics: () -> Unit,
    onOpenChat: () -> Unit,
    onOpenSettings: () -> Unit,
    vm: AdminHomeViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsStateWithLifecycle()

    val vulnerabilities = listOf(
        VulnerabilityItem("Voice Social Engineering", 0.42f, DangerRed),
        VulnerabilityItem("Network Hygiene",          0.58f, WarnAmber),
        VulnerabilityItem("Password Hygiene",         0.63f, WarnAmber),
        VulnerabilityItem("Phishing Detection",       0.81f, CorrectGreen),
        VulnerabilityItem("USB Drop Simulation",      0.88f, CorrectGreen),
    )

    val sessions = listOf(
        SessionRow("EMP_042", "Grounded", "80%", "+27%", "Done",    CorrectGreen,       Color(0xFFE8F5E9)),
        SessionRow("EMP_031", "Grounded", "60%", "+15%", "Done",    CorrectGreen,       Color(0xFFE8F5E9)),
        SessionRow("EMP_018", "Baseline", "40%", "+2%",  "Live",    SentryCyan,         Color(0xFFE3F2FD)),
        SessionRow("EMP_055", "Grounded", "—",   "—",    "Pending", Color(0xFF9E9E9E),  Color(0xFFF5F5F5)),
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White),
    ) {
        // ── Top bar ──────────────────────────────────────────────────
        AdminTopBar(
            onOpenChat     = onOpenChat,
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
            // Left — vulnerability hotspots
            Column(
                modifier = Modifier
                    .weight(1.4f)
                    .fillMaxHeight()
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                AdminWelcomeHeader()
                MetricRow()
                VulnerabilityCard(vulnerabilities = vulnerabilities)
            }

            // Right — recent sessions + analytics CTA
            Column(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                RecentSessionsCard(
                    sessions        = sessions,
                    onViewAll       = onOpenAnalytics,
                )
            }
        }
    }
}

@Composable
private fun AdminTopBar(
    onOpenChat: () -> Unit,
    onOpenSettings: () -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(68.dp)
            .background(SentryCyan),
    ) {
        // Left — avatar
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
                    text       = "AD",
                    fontSize   = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color      = Color.White,
                )
            }
            Spacer(Modifier.width(10.dp))
            Column {
                Text(
                    text       = "Admin",
                    fontSize   = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color      = Color.White,
                )
                Text(
                    text     = "Manager",
                    fontSize = 11.sp,
                    color    = Color.White.copy(alpha = 0.8f),
                )
            }
        }

        // Centre — SENTRY wordmark
        Column(
            modifier            = Modifier.align(Alignment.Center),
            horizontalAlignment = Alignment.CenterHorizontally,
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
                    text          = "SENTRY",
                    fontFamily    = ItimFont,
                    fontSize      = 22.sp,
                    fontWeight    = FontWeight.Bold,
                    color         = Color.White,
                    letterSpacing = 3.sp,
                )
            }
            Text(
                text     = "Admin Panel",
                fontSize = 11.sp,
                color    = Color.White.copy(alpha = 0.8f),
            )
        }

        // Right — chat + settings
        Row(
            modifier          = Modifier
                .align(Alignment.CenterEnd)
                .padding(end = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(8.dp))
                    .background(Color.White.copy(alpha = 0.2f))
                    .clickable { onOpenChat() }
                    .padding(horizontal = 10.dp, vertical = 6.dp),
            ) {
                Text(
                    text     = "Chat",
                    fontSize = 12.sp,
                    color    = Color.White,
                    fontWeight = FontWeight.Medium,
                )
            }
            IconButton(onClick = onOpenSettings) {
                Icon(
                    Icons.Default.Settings,
                    contentDescription = "Settings",
                    tint               = Color.White,
                    modifier           = Modifier.size(24.dp),
                )
            }
        }
    }
}

@Composable
private fun AdminWelcomeHeader() {
    Column {
        Text(
            text       = "Organisation overview",
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
private fun MetricRow() {
    Row(
        modifier              = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        listOf(
            Triple("24",  "Total sessions",     "+6 this week"),
            Triple("71%", "Mean accuracy",      "+8% vs last week"),
            Triple("18",  "Employees trained",  "of 30 total"),
        ).forEach { (value, label, sub) ->
            Box(
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(14.dp))
                    .background(Color.White)
                    .border(1.dp, CardBorder, RoundedCornerShape(14.dp))
                    .padding(16.dp),
            ) {
                Column {
                    Text(
                        text       = value,
                        fontFamily = PhilosopherFont,
                        fontSize   = 26.sp,
                        fontWeight = FontWeight.Bold,
                        color      = TextPrimary,
                    )
                    Text(text = label, fontSize = 12.sp, color = TextPrimary)
                    Text(text = sub,   fontSize = 11.sp, color = TextSecondary)
                }
            }
        }
    }
}

@Composable
private fun VulnerabilityCard(vulnerabilities: List<VulnerabilityItem>) {
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
                    text       = "Vulnerability hotspots",
                    fontFamily = PhilosopherFont,
                    fontSize   = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color      = TextPrimary,
                )
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(20.dp))
                        .background(Color(0xFFFFEBEE))
                        .padding(horizontal = 12.dp, vertical = 4.dp),
                ) {
                    Text(
                        text     = "3 areas at risk",
                        fontSize = 11.sp,
                        color    = DangerRed,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }

            Spacer(Modifier.height(16.dp))

            vulnerabilities.forEach { item ->
                Column(modifier = Modifier.padding(bottom = 12.dp)) {
                    Row(
                        modifier              = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        Text(
                            text     = item.name,
                            fontSize = 13.sp,
                            color    = TextPrimary,
                            fontWeight = FontWeight.Medium,
                        )
                        Text(
                            text     = "${(item.accuracy * 100).toInt()}% accuracy",
                            fontSize = 12.sp,
                            color    = item.color,
                            fontWeight = FontWeight.Bold,
                        )
                    }
                    Spacer(Modifier.height(5.dp))
                    LinearProgressIndicator(
                        progress   = { item.accuracy },
                        modifier   = Modifier
                            .fillMaxWidth()
                            .height(7.dp)
                            .clip(RoundedCornerShape(4.dp)),
                        color      = item.color,
                        trackColor = Color(0xFFE0E0E0),
                        strokeCap  = StrokeCap.Round,
                    )
                }
            }
        }
    }
}

@Composable
private fun RecentSessionsCard(
    sessions: List<SessionRow>,
    onViewAll: () -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(Color.White)
            .border(1.dp, CardBorder, RoundedCornerShape(16.dp)),
    ) {
        Column {
            // Header
            Row(
                modifier              = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 14.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment     = Alignment.CenterVertically,
            ) {
                Text(
                    text       = "Recent sessions",
                    fontFamily = PhilosopherFont,
                    fontSize   = 15.sp,
                    fontWeight = FontWeight.Bold,
                    color      = TextPrimary,
                )
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(SentryCyan)
                        .clickable { onViewAll() }
                        .padding(horizontal = 10.dp, vertical = 5.dp),
                ) {
                    Text(
                        text     = "View all",
                        fontSize = 12.sp,
                        color    = Color.White,
                        fontWeight = FontWeight.Medium,
                    )
                }
            }

            // Divider
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(0.5.dp)
                    .background(CardBorder),
            )

            // Table header
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFFF9F9F9))
                    .padding(horizontal = 16.dp, vertical = 8.dp),
            ) {
                TableHeaderCell("ID",        Modifier.weight(1f))
                TableHeaderCell("Mode",      Modifier.weight(1f))
                TableHeaderCell("Score",     Modifier.weight(0.8f))
                TableHeaderCell("Gain",      Modifier.weight(0.8f))
                TableHeaderCell("Status",    Modifier.weight(1f))
            }

            // Rows
            sessions.forEach { session ->
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(0.5.dp)
                        .background(CardBorder),
                )
                Row(
                    modifier          = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(session.participantId, fontSize = 12.sp, color = TextPrimary,   modifier = Modifier.weight(1f))
                    Text(session.condition,     fontSize = 12.sp, color = TextSecondary, modifier = Modifier.weight(1f))
                    Text(session.score,         fontSize = 12.sp, color = TextPrimary,   modifier = Modifier.weight(0.8f))
                    Text(
                        text     = session.gain,
                        fontSize = 12.sp,
                        color    = if (session.gain.startsWith("+")) CorrectGreen else TextSecondary,
                        modifier = Modifier.weight(0.8f),
                    )
                    Box(
                        modifier = Modifier
                            .weight(1f),
                    ) {
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(12.dp))
                                .background(session.statusBg)
                                .padding(horizontal = 8.dp, vertical = 3.dp),
                        ) {
                            Text(
                                text     = session.status,
                                fontSize = 11.sp,
                                color    = session.statusColor,
                                fontWeight = FontWeight.Bold,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun TableHeaderCell(text: String, modifier: Modifier) {
    Text(
        text       = text,
        fontSize   = 11.sp,
        color      = TextSecondary,
        fontWeight = FontWeight.Bold,
        modifier   = modifier,
    )
}

private fun currentDate(): String {
    val cal    = java.util.Calendar.getInstance()
    val months = arrayOf("Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec")
    return "${cal.get(java.util.Calendar.DAY_OF_MONTH)} " +
            "${months[cal.get(java.util.Calendar.MONTH)]} " +
            "${cal.get(java.util.Calendar.YEAR)}"
}