package com.sentry.app.features.auth

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sentry.app.R
import com.sentry.app.core.navigation.UserRole
import com.sentry.app.features.splash.SentryCyan
import com.sentry.app.ui.theme.ItimFont
import com.sentry.app.ui.theme.PhilosopherFont

private val SentryGreen     = Color(0xFF4CAF50)
private val SentryGreenDk   = Color(0xFF388E3C)
private val SentryGreenLt   = Color(0xFFE8F5E9)
private val BorderGreen     = Color(0xFF66BB6A)
private val TextFieldBorder = Color(0xFF66BB6A)

@Composable
fun AuthScreen(
    state: AuthUiState,
    onLoginClick: (String, String, String, String) -> Unit,
    onLoginSuccess: (UserRole) -> Unit,
) {
    var participantId  by remember { mutableStateOf("") }
    var password       by remember { mutableStateOf("") }
    var organisation   by remember { mutableStateOf("") }
    var selectedRole   by remember { mutableStateOf(UserRole.TRAINEE) }

    LaunchedEffect(state.success) {
        state.success?.let { onLoginSuccess(it) }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {

        // ── Cyan header bar ──────────────────────────────────────────
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(90.dp)
                .background(SentryCyan),
        )

        // ── Pepper robot card — overlaps the cyan bar ────────────────
        Box(
            modifier = Modifier
                .padding(top = 0.dp)
                .size(width = 180.dp, height = 130.dp)
                .clip(RoundedCornerShape(16.dp))
                .background(Color.White)
                .border(1.dp, Color(0xFFE0E0E0), RoundedCornerShape(16.dp)),
            contentAlignment = Alignment.Center,
        ) {
            Image(
                painter            = painterResource(id = R.drawable.pepper_robot),
                contentDescription = "Pepper robot",
                modifier           = Modifier.size(width = 160.dp, height = 115.dp),
                contentScale       = ContentScale.Fit,
            )
        }

        Spacer(Modifier.height(16.dp))

        // ── Welcome text ─────────────────────────────────────────────
        Text(
            text       = "Welcome to SENTRY, your personalised and\ngrounded AI tutor in Cybersecurity.",
            fontFamily = PhilosopherFont,
            fontSize   = 15.sp,
            color      = Color(0xFF212121),
            textAlign  = TextAlign.Center,
            lineHeight = 22.sp,
            modifier   = Modifier.padding(horizontal = 32.dp),
        )

        Spacer(Modifier.height(20.dp))

        // ── Role toggle ──────────────────────────────────────────────
        Row(
            modifier            = Modifier.padding(horizontal = 80.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            RoleButton(
                label    = "Trainee",
                selected = selectedRole == UserRole.TRAINEE,
                onClick  = { selectedRole = UserRole.TRAINEE },
                modifier = Modifier.weight(1f),
            )
            RoleButton(
                label    = "Admin",
                selected = selectedRole == UserRole.ADMIN,
                onClick  = { selectedRole = UserRole.ADMIN },
                modifier = Modifier.weight(1f),
            )
        }

        Spacer(Modifier.height(16.dp))

        // ── Form card ────────────────────────────────────────────────
        Box(
            modifier = Modifier
                .padding(horizontal = 32.dp)
                .fillMaxWidth()
                .clip(RoundedCornerShape(16.dp))
                .border(1.5.dp, BorderGreen, RoundedCornerShape(16.dp))
                .background(Color.White)
                .padding(20.dp),
        ) {
            Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {

                // Left: form fields
                Column(modifier = Modifier.weight(1f)) {
                    if (selectedRole == UserRole.ADMIN) {
                        AuthField(
                            label       = "Organisation:",
                            value       = organisation,
                            onValueChange = { organisation = it },
                        )
                        Spacer(Modifier.height(12.dp))
                    }
                    AuthField(
                        label         = "Participant ID:",
                        value         = participantId,
                        onValueChange = { participantId = it },
                    )
                    Spacer(Modifier.height(12.dp))
                    AuthField(
                        label         = "Password:",
                        value         = password,
                        onValueChange = { password = it },
                        isPassword    = true,
                    )
                    Spacer(Modifier.height(8.dp))
                    Text(
                        text     = "Forgot Password?",
                        fontSize = 12.sp,
                        color    = Color(0xFFF44336),
                        modifier = Modifier
                            .align(Alignment.End)
                            .clickable { },
                    )
                }

                Spacer(Modifier.width(8.dp))

                // Right: info panel + proceed button
                Column(modifier = Modifier.weight(1f)) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .background(SentryGreen)
                            .padding(16.dp),
                    ) {
                        Column {
                            Text(
                                text       = if (selectedRole == UserRole.TRAINEE)
                                    "Signing in as an Employee"
                                else
                                    "Sign in as an Admin",
                                fontFamily = PhilosopherFont,
                                fontSize   = 15.sp,
                                fontWeight = FontWeight.Bold,
                                color      = Color.White,
                            )
                            Spacer(Modifier.height(10.dp))
                            InfoItem("Real-world scenarios")
                            InfoItem("AI grounded explanations")
                            InfoItem("Legal context from Kenyan Law")
                            if (selectedRole == UserRole.ADMIN) {
                                InfoItem("Organisational analytics")
                            }
                        }
                    }

                    Spacer(Modifier.height(12.dp))

                    // Proceed button
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .background(SentryGreenDk)
                            .clickable {
                                val org = if (selectedRole == UserRole.ADMIN)
                                    organisation else "SENTRY_STUDY"
                                onLoginClick(
                                    participantId,
                                    password,
                                    selectedRole.name.lowercase(),
                                    org,
                                )
                            }
                            .padding(vertical = 14.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            text       = if (selectedRole == UserRole.TRAINEE)
                                "Proceed to sessions ⊞"
                            else
                                "Proceed to Admin Panel ⊞",
                            fontFamily = PhilosopherFont,
                            fontSize   = 14.sp,
                            fontWeight = FontWeight.Bold,
                            color      = Color.White,
                        )
                    }

                    // Error message
                    if (state.error.isNotBlank()) {
                        Spacer(Modifier.height(8.dp))
                        Text(
                            text     = state.error,
                            fontSize = 12.sp,
                            color    = Color(0xFFF44336),
                            textAlign = TextAlign.Center,
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                }
            }
        }

        Spacer(Modifier.height(32.dp))
    }
}

@Composable
private fun RoleButton(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(24.dp))
            .background(if (selected) SentryGreen else Color.White)
            .border(
                width = 1.5.dp,
                color = if (selected) SentryGreen else Color(0xFFBDBDBD),
                shape = RoundedCornerShape(24.dp),
            )
            .clickable { onClick() }
            .padding(vertical = 10.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text       = label,
            fontFamily = PhilosopherFont,
            fontSize   = 14.sp,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
            color      = if (selected) Color.White else Color(0xFF424242),
        )
    }
}

@Composable
private fun AuthField(
    label: String,
    value: String,
    onValueChange: (String) -> Unit,
    isPassword: Boolean = false,
) {
    Column {
        Text(
            text     = label,
            fontSize = 13.sp,
            color    = Color(0xFF424242),
            fontWeight = FontWeight.Medium,
        )
        Spacer(Modifier.height(4.dp))
        OutlinedTextField(
            value         = value,
            onValueChange = onValueChange,
            modifier      = Modifier.fillMaxWidth(),
            singleLine    = true,
            shape         = RoundedCornerShape(10.dp),
            colors        = OutlinedTextFieldDefaults.colors(
                focusedBorderColor   = TextFieldBorder,
                unfocusedBorderColor = Color(0xFFBDBDBD),
            ),
            visualTransformation = if (isPassword) PasswordVisualTransformation()
            else androidx.compose.ui.text.input.VisualTransformation.None,
            keyboardOptions = if (isPassword)
                KeyboardOptions(keyboardType = KeyboardType.Password)
            else
                KeyboardOptions.Default,
        )
    }
}

@Composable
private fun InfoItem(text: String) {
    Row(
        modifier       = Modifier.padding(vertical = 3.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text("i.  ", fontSize = 12.sp, color = Color.White)
        Text(
            text       = text,
            fontFamily = PhilosopherFont,
            fontSize   = 12.sp,
            color      = Color.White,
        )
    }
}