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
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
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
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavHostController
import com.sentry.app.R
import com.sentry.app.core.navigation.UserRole
import com.sentry.app.core.navigation.navigateAndClear
import com.sentry.app.core.ui.components.texts.SentryText
import com.sentry.app.core.ui.models.SentryTextAlign
import com.sentry.app.core.ui.models.SentryTextSize
import com.sentry.app.core.ui.theme.LocalBrandColors
import com.sentry.app.core.ui.theme.PhilosopherFont

// component-specific tokens
private val AuthGreen = Color(0xFF34C659)
private val AuthGreenDark = Color(0xFF34C659)
private val AuthGreenBorder = Color(0xFF34C659)
private val NeutralBorder = Color(0xFFBDBDBD)
private val NeutralText = Color(0xFF424242)
private val CardBorder = Color(0xFFE0E0E0)

// shared corner radius used everywhere on this screen
private val AuthRadius = RoundedCornerShape(12.dp)

@Composable
fun AuthScreen(
    navController: NavHostController,
    vm: AuthViewModel = hiltViewModel(),
) {
    val state by vm.uiState.collectAsStateWithLifecycle()
    val scheme = MaterialTheme.colorScheme
    val brand = LocalBrandColors.current

    var participantId by remember { mutableStateOf("") }
    var pin by remember { mutableStateOf("") }
    var organisation by remember { mutableStateOf("") }
    var selectedRole by remember { mutableStateOf(UserRole.TRAINEE) }

    // collect one-shot nav events
    LaunchedEffect(Unit) {
        vm.events.collect { event ->
            when (event) {
                is AuthEvent.NavigateToHome -> {
                    val route = if (event.role == UserRole.ADMIN) "adminHome" else "traineeHome"
                    navController.navigateAndClear(route, popUpToRoute = "auth")

                }
            }
        }
    }
    Column {
        Box(
            Modifier
                .fillMaxWidth()
                .height(60.dp)
                // .padding(bottom = 60.dp)
                .background(MaterialTheme.colorScheme.primary)
            //.shadow(elevation = 10.dp)
        )
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(scheme.background)
                .imePadding()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 24.dp, vertical = 20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {

            // ── Branding row ─────────────────────────────────────────────
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // pepper robot card
                Box(
                    modifier = Modifier
                        .size(width = 90.dp, height = 72.dp)
                        .clip(AuthRadius)
                        .background(Color.White)
                        .border(1.dp, CardBorder, AuthRadius),
                    contentAlignment = Alignment.Center,
                ) {
                    Image(
                        painter = painterResource(id = R.drawable.pepper_robot),
                        contentDescription = "Pepper robot",
                        modifier = Modifier.size(width = 80.dp, height = 64.dp),
                        contentScale = ContentScale.Fit,
                    )
                }

                Spacer(Modifier.width(20.dp))

                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    SentryText(
                        text = "Welcome to SENTRY",
                        fontFamily = PhilosopherFont,
                        size = SentryTextSize.Xl,
                        weight = FontWeight.Bold,
                        color = scheme.onBackground,
                    )
                    SentryText(
                        text = "Your personalised and grounded AI tutor in Cybersecurity.",
                        size = SentryTextSize.Sm,
                        color = scheme.outline,
                        maxLines = 2,
                    )
                }
            }

            Spacer(Modifier.height(20.dp))

            // ── Role toggle ───────────────────────────────────────────────
            Row(
                modifier = Modifier
                    .fillMaxWidth(0.5f)
                    .clip(AuthRadius)
                    .background(scheme.inverseOnSurface)
                    .padding(4.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                RoleButton(
                    icon = R.drawable.users,
                    label = "Trainee",
                    selected = selectedRole == UserRole.TRAINEE,
                    onClick = { selectedRole = UserRole.TRAINEE },
                    modifier = Modifier.weight(1f),
                )
                RoleButton(
                    icon = R.drawable.briefcase,
                    label = "Admin",
                    selected = selectedRole == UserRole.ADMIN,
                    onClick = { selectedRole = UserRole.ADMIN },
                    modifier = Modifier.weight(1f),
                )
            }

            Spacer(Modifier.height(20.dp))

            // ── Form card ─────────────────────────────────────────────────
            Box(
                modifier = Modifier
                    .fillMaxWidth(0.65f)
                    .clip(AuthRadius)
                    .border(1.5.dp, AuthGreenBorder, AuthRadius)
                    .background(scheme.background)
                    .padding(20.dp),
            ) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(20.dp),
                    verticalAlignment = Alignment.Top,
                ) {

                    // left — form fields
                    Column(modifier = Modifier.weight(1f)) {
                        SentryText(
                            text = "Sign in",
                            size = SentryTextSize.Xl,
                            weight = FontWeight.Bold,
                            color = scheme.onBackground,
                        )
                        Spacer(Modifier.height(2.dp))
                        SentryText(
                            text = "Enter your credentials to continue",
                            size = SentryTextSize.Sm,
                            color = scheme.outline,
                        )

                        Spacer(Modifier.height(16.dp))

                        // organisation — admin only
                        if (selectedRole == UserRole.ADMIN) {
                            AuthField(
                                label = "Organisation:",
                                value = organisation,
                                onValueChange = { organisation = it },
                            )
                            Spacer(Modifier.height(12.dp))
                        }

                        AuthField(
                            label = "Participant ID:",
                            value = participantId,
                            onValueChange = { participantId = it },
                        )

                        Spacer(Modifier.height(12.dp))

                        AuthField(
                            label = "PIN:",
                            value = pin,
                            onValueChange = { pin = it },
                            isPassword = true,
                        )

                        Spacer(Modifier.height(4.dp))

                        SentryText(
                            text = "Forgot PIN?",
                            size = SentryTextSize.Sm,
                            color = brand.red,
                            modifier = Modifier.clickable { },
                        )
                    }

                    // right — info panel + proceed button
                    Column(modifier = Modifier.weight(1f)) {

                        // info panel
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(AuthRadius)
                                .background(AuthGreen)
                                .padding(16.dp),
                        ) {
                            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                SentryText(
                                    text = if (selectedRole == UserRole.TRAINEE)
                                        "Signing in as an Employee"
                                    else
                                        "Sign in as an Admin",
                                    size = SentryTextSize.Md,
                                    weight = FontWeight.Bold,
                                    color = Color.White,
                                    align = SentryTextAlign.Center,
                                    modifier = Modifier.fillMaxWidth(),
                                )

                                Spacer(Modifier.height(4.dp))

                                InfoItem("Real-world scenarios")
                                InfoItem("AI grounded explanations")
                                InfoItem("Legal context from Kenyan Law")
                                if (selectedRole == UserRole.ADMIN) {
                                    InfoItem("Organisational analytics")
                                }
                            }
                        }

                        Spacer(Modifier.height(12.dp))

                        // proceed button
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(AuthRadius)
                                .background(
                                    if (state.loading) scheme.outline else AuthGreenDark
                                )
                                .clickable(enabled = !state.loading) {
                                    val org = if (selectedRole == UserRole.ADMIN)
                                        organisation else "SENTRY_STUDY"
                                    vm.login(
                                        participantId = participantId,
                                        pin = pin,
                                        role = selectedRole.name.lowercase(),
                                        organisationId = org,
                                    )
                                }
                                .padding(vertical = 14.dp),
                            horizontalArrangement = Arrangement.spacedBy(
                                10.dp,
                                Alignment.CenterHorizontally
                            )
                        ) {
                            if (state.loading) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(20.dp),
                                    color = Color.White,
                                    strokeWidth = 2.dp,
                                )
                            } else {
                                SentryText(
                                    text = if (selectedRole == UserRole.TRAINEE)
                                        "Proceed to sessions"
                                    else
                                        "Proceed to Admin Panel",
                                    size = SentryTextSize.Lg,
                                    weight = FontWeight.Bold,
                                    color = Color.White,
                                )
                                Icon(
                                    painter = painterResource(R.drawable.grid),
                                    tint = CardBorder,
                                    contentDescription = "",
                                    modifier = Modifier.size(23.dp)
                                )
                            }
                        }

                        // error
                        if (state.error.isNotBlank()) {
                            Spacer(Modifier.height(8.dp))
                            SentryText(
                                text = state.error,
                                size = SentryTextSize.Sm,
                                color = brand.red,
                                align = SentryTextAlign.Center,
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }
                    }
                }
            }

            Spacer(Modifier.height(24.dp))
        }
    }
    // outer column fills the screen; top branding, then the form card
}


@Composable
private fun RoleButton(
    icon: Int,
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier
            .clip(AuthRadius)
            // .shadow(elevation = 2.dp, shape = AuthRadius)
            .background(if (selected) Color.White else Color.Transparent)
            .clickable { onClick() }
            .padding(vertical = 10.dp, horizontal = 16.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp, Alignment.CenterHorizontally),
        verticalAlignment = Alignment.CenterVertically
    ) {
        SentryText(
            text = label,
            size = SentryTextSize.Lg,
            weight = if (selected) FontWeight.Bold else FontWeight.Normal,
            color = if (selected) AuthGreen else NeutralText,
        )
        Icon(
            painter = painterResource(icon),
            tint = if (selected) AuthGreen else NeutralText,
            contentDescription = "",
            modifier = Modifier.size(25.dp)
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
        SentryText(
            text = label,
            size = SentryTextSize.Sm,
            weight = FontWeight.Medium,
            color = NeutralText,
        )
        Spacer(Modifier.height(4.dp))
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            shape = AuthRadius,
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = AuthGreenBorder,
                unfocusedBorderColor = NeutralBorder,
            ),
            visualTransformation = if (isPassword) PasswordVisualTransformation()
            else VisualTransformation.None,
            keyboardOptions = if (isPassword)
                KeyboardOptions(keyboardType = KeyboardType.NumberPassword)
            else
                KeyboardOptions.Default,
        )
    }
}

@Composable
private fun InfoItem(text: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.padding(vertical = 2.dp),
    ) {
        SentryText(
            text = "\u2713  ",
            size = SentryTextSize.Md,
            weight = FontWeight.Bold,
            color = Color.White,
        )
        SentryText(
            text = text,
            size = SentryTextSize.Sm,
            color = Color.White,
        )
    }
}