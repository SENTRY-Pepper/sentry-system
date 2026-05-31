package com.sentry.app.features.chat

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sentry.app.R
import com.sentry.app.features.splash.SentryCyan
import com.sentry.app.ui.theme.PhilosopherFont

private val UserBubble    = Color(0xFF29B6F6)
private val PepperBubble  = Color(0xFFE3F2FD)
private val PepperBorder  = Color(0xFFBBDEFB)
private val SourceTagBg   = Color(0xFFE8EAF6)
private val SourceTagText = Color(0xFF3949AB)
private val TextPrimary   = Color(0xFF212121)
private val TextSecondary = Color(0xFF757575)

@Composable
fun ChatScreen(
    onBack: () -> Unit,
    vm: ChatViewModel = hiltViewModel(),
) {
    val state     by vm.uiState.collectAsStateWithLifecycle()
    val listState = rememberLazyListState()

    // Auto-scroll to bottom when new message arrives
    LaunchedEffect(state.messages.size) {
        if (state.messages.isNotEmpty()) {
            listState.animateScrollToItem(state.messages.lastIndex)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF5F5F5))
            .imePadding(),
    ) {
        // ── Top bar ──────────────────────────────────────────────────
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(64.dp)
                .background(SentryCyan),
        ) {
            IconButton(
                onClick  = onBack,
                modifier = Modifier.align(Alignment.CenterStart),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint = Color.White,
                    )
                    Text(
                        text       = "BACK",
                        color      = Color.White,
                        fontSize   = 13.sp,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }
            Column(
                modifier            = Modifier.align(Alignment.Center),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(
                    text       = "Chat with Pepper",
                    fontFamily = PhilosopherFont,
                    fontSize   = 17.sp,
                    fontWeight = FontWeight.Bold,
                    color      = Color.White,
                )
                Text(
                    text     = "Grounded AI · OWASP + Kenyan law",
                    fontSize = 11.sp,
                    color    = Color.White.copy(alpha = 0.85f),
                )
            }
        }

        // ── Message list ─────────────────────────────────────────────
        LazyColumn(
            state           = listState,
            modifier        = Modifier
                .weight(1f)
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            items(state.messages) { message ->
                if (message.isUser) {
                    UserMessage(text = message.text)
                } else {
                    PepperMessage(
                        text    = message.text,
                        sources = message.sources,
                    )
                }
            }

            if (state.loading) {
                item {
                    PepperTypingIndicator()
                }
            }
        }

        // ── Input bar ────────────────────────────────────────────────
        Row(
            modifier          = Modifier
                .fillMaxWidth()
                .background(Color.White)
                .border(0.5.dp, Color(0xFFE0E0E0), RoundedCornerShape(0.dp))
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            OutlinedTextField(
                value         = state.inputText,
                onValueChange = { vm.onInputChanged(it) },
                placeholder   = {
                    Text(
                        "Ask Pepper about cybersecurity…",
                        fontSize = 13.sp,
                        color    = Color(0xFFBDBDBD),
                    )
                },
                modifier      = Modifier.weight(1f),
                singleLine    = true,
                shape         = RoundedCornerShape(24.dp),
                colors        = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor   = SentryCyan,
                    unfocusedBorderColor = Color(0xFFE0E0E0),
                ),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { vm.sendMessage() }),
            )

            Spacer(Modifier.width(8.dp))

            Box(
                modifier = Modifier
                    .size(44.dp)
                    .clip(CircleShape)
                    .background(SentryCyan),
                contentAlignment = Alignment.Center,
            ) {
                IconButton(onClick = { vm.sendMessage() }) {
                    Icon(
                        Icons.AutoMirrored.Filled.Send,
                        contentDescription = "Send",
                        tint               = Color.White,
                        modifier           = Modifier.size(20.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun UserMessage(text: String) {
    Row(
        modifier              = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.End,
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 480.dp)
                .clip(RoundedCornerShape(18.dp, 18.dp, 4.dp, 18.dp))
                .background(UserBubble)
                .padding(horizontal = 16.dp, vertical = 12.dp),
        ) {
            Text(
                text      = text,
                fontSize  = 14.sp,
                color     = Color.White,
                lineHeight = 21.sp,
            )
        }
    }
}

@Composable
private fun PepperMessage(text: String, sources: List<String>) {
    Row(
        modifier          = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.Bottom,
        horizontalArrangement = Arrangement.Start,
    ) {
        // Pepper avatar
        Box(
            modifier = Modifier
                .size(36.dp)
                .clip(CircleShape)
                .background(Color.White)
                .border(1.dp, Color(0xFFE0E0E0), CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            Image(
                painter            = painterResource(R.drawable.pepper_robot),
                contentDescription = "Pepper",
                modifier           = Modifier
                    .size(30.dp)
                    .clip(CircleShape),
                contentScale       = ContentScale.Crop,
            )
        }

        Spacer(Modifier.width(8.dp))

        Column(modifier = Modifier.widthIn(max = 480.dp)) {
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(18.dp, 18.dp, 18.dp, 4.dp))
                    .background(PepperBubble)
                    .border(
                        0.5.dp,
                        PepperBorder,
                        RoundedCornerShape(18.dp, 18.dp, 18.dp, 4.dp),
                    )
                    .padding(horizontal = 16.dp, vertical = 12.dp),
            ) {
                Text(
                    text      = text,
                    fontSize  = 14.sp,
                    color     = TextPrimary,
                    lineHeight = 21.sp,
                )
            }

            // Sources
            if (sources.isNotEmpty()) {
                Spacer(Modifier.height(6.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(
                        text     = "Sources:",
                        fontSize = 10.sp,
                        color    = TextSecondary,
                    )
                    sources.take(3).forEach { source ->
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(10.dp))
                                .background(SourceTagBg)
                                .padding(horizontal = 8.dp, vertical = 3.dp),
                        ) {
                            Text(
                                text     = source,
                                fontSize = 10.sp,
                                color    = SourceTagText,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun PepperTypingIndicator() {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(36.dp)
                .clip(CircleShape)
                .background(Color.White)
                .border(1.dp, Color(0xFFE0E0E0), CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            Image(
                painter            = painterResource(R.drawable.pepper_robot),
                contentDescription = null,
                modifier           = Modifier.size(30.dp).clip(CircleShape),
                contentScale       = ContentScale.Crop,
            )
        }
        Spacer(Modifier.width(8.dp))
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(18.dp))
                .background(PepperBubble)
                .border(0.5.dp, PepperBorder, RoundedCornerShape(18.dp))
                .padding(horizontal = 16.dp, vertical = 12.dp),
        ) {
            Row(
                verticalAlignment     = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                CircularProgressIndicator(
                    modifier    = Modifier.size(14.dp),
                    color       = SentryCyan,
                    strokeWidth = 2.dp,
                )
                Text(
                    text     = "Pepper is thinking…",
                    fontSize = 13.sp,
                    color    = TextSecondary,
                )
            }
        }
    }
}