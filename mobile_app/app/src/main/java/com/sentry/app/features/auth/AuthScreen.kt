package com.sentry.app.features.auth

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.CircularProgressIndicator
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
import com.sentry.app.core.navigation.navigateSingleTop
import com.sentry.app.core.ui.components.texts.SentryText
import com.sentry.app.core.ui.models.SentryTextAlign
import com.sentry.app.core.ui.models.SentryTextSize
import com.sentry.app.core.ui.theme.LocalBrandColors

// component-specific tokens — green accent is auth-screen only, not a global brand token
private val AuthGreen = Color(0xFF4CAF50)
private val AuthGreenDark = Color(0xFF388E3C)
private val AuthGreenBorder = Color(0xFF66BB6A)
private val NeutralBorder = Color(0xFFBDBDBD)
private val NeutralText = Color(0xFF424242)
private val CardBorder = Color(0xFFE0E0E0)

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

    Row(
        modifier = Modifier
            .fillMaxSize()
            .background(scheme.background),
    ) {
        // ── Left column — branding + role toggle ─────────────────────
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight()
                .background(scheme.primary),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            // pepper robot card
            Box(
                modifier = Modifier
                    .size(width = 180.dp, height = 140.dp)
                    .clip(RoundedCornerShape(16.dp))
                    .background(Color.White)
                    .border(1.dp, CardBorder, RoundedCornerShape(16.dp)),
                contentAlignment = Alignment.Center,
            ) {
                Image(
                    painter = painterResource(id = R.drawable.pepper_robot),
                    contentDescription = "Pepper robot",
                    modifier = Modifier.size(width = 160.dp, height = 120.dp),
                    contentScale = ContentScale.Fit,
                )
            }

            Spacer(Modifier.height(24.dp))

            SentryText(
                text = "Welcome to SENTRY",
                size = SentryTextSize.Display,
                weight = FontWeight.Bold,
                color = Color.White,
                align = SentryTextAlign.Center,
                modifier = Modifier.padding(horizontal = 24.dp),
            )

            Spacer(Modifier.height(8.dp))

            SentryText(
                text = "Your personalised and grounded\nAI tutor in Cybersecurity",
                size = SentryTextSize.Sm,
                color = Color.White.copy(alpha = 0.85f),
                align = SentryTextAlign.Center,
                maxLines = 3,
                modifier = Modifier.padding(horizontal = 32.dp),
            )

            Spacer(Modifier.height(32.dp))

            // role toggle
            Row(
                modifier = Modifier.padding(horizontal = 32.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                RoleButton(
                    label = "Trainee",
                    selected = selectedRole == UserRole.TRAINEE,
                    onClick = { selectedRole = UserRole.TRAINEE },
                    modifier = Modifier.weight(1f),
                )
                RoleButton(
                    label = "Admin",
                    selected = selectedRole == UserRole.ADMIN,
                    onClick = { selectedRole = UserRole.ADMIN },
                    modifier = Modifier.weight(1f),
                )
            }

            Spacer(Modifier.height(24.dp))

            // info panel
            Box(
                modifier = Modifier
                    .padding(horizontal = 32.dp)
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(Color.White.copy(alpha = 0.12f))
                    .padding(16.dp),
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    SentryText(
                        text = if (selectedRole == UserRole.TRAINEE)
                            "Signing in as an Employee"
                        else
                            "Signing in as an Admin",
                        size = SentryTextSize.Md,
                        weight = FontWeight.Bold,
                        color = Color.White,
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
        }

        // ── Right column — login form ─────────────────────────────────
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight()
                .background(scheme.background),
            contentPadding = PaddingValues(32.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            item {
                SentryText(
                    text = "Sign in",
                    size = SentryTextSize.Xl,
                    weight = FontWeight.Bold,
                    color = scheme.onBackground,
                )
                Spacer(Modifier.height(4.dp))
                SentryText(
                    text = "Enter your credentials to continue",
                    size = SentryTextSize.Sm,
                    color = scheme.outline,
                )
            }

            // organisation field — admin only
            if (selectedRole == UserRole.ADMIN) {
                item {
                    AuthField(
                        label = "Organisation:",
                        value = organisation,
                        onValueChange = { organisation = it },
                    )
                }
            }

            item {
                AuthField(
                    label = "Participant ID:",
                    value = participantId,
                    onValueChange = { participantId = it },
                )
            }

            item {
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
                    modifier = Modifier
                        //.align(Alignment.End)
                        .clickable { },
                )
            }

            // proceed button
            item {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .background(
                            if (state.loading) scheme.outline
                            else AuthGreenDark
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
                    contentAlignment = Alignment.Center,
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
                                "Proceed to sessions ⊞"
                            else
                                "Proceed to Admin Panel ⊞",
                            size = SentryTextSize.Md,
                            weight = FontWeight.Bold,
                            color = Color.White,
                        )
                    }
                }
            }

            // error
            if (state.error.isNotBlank()) {
                item {
                    SentryText(
                        text = state.error,
                        size = SentryTextSize.Sm,
                        color = brand.red,
                        align = SentryTextAlign.Center,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }

            item { Spacer(Modifier.height(80.dp)) }
        }
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
            .background(
                if (selected) Color.White else Color.Transparent
            )
            .border(
                width = 1.5.dp,
                color = if (selected) Color.White else Color.White.copy(alpha = 0.5f),
                shape = RoundedCornerShape(24.dp),
            )
            .clickable { onClick() }
            .padding(vertical = 10.dp),
        contentAlignment = Alignment.Center,
    ) {
        SentryText(
            text = label,
            size = SentryTextSize.Md,
            weight = if (selected) FontWeight.Bold else FontWeight.Normal,
            color = if (selected) AuthGreen else Color.White,
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
    val scheme = MaterialTheme.colorScheme

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
            shape = RoundedCornerShape(10.dp),
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
            text = "i.  ",
            size = SentryTextSize.Sm,
            color = Color.White.copy(alpha = 0.7f),
        )
        SentryText(
            text = text,
            size = SentryTextSize.Sm,
            color = Color.White,
        )
    }
}