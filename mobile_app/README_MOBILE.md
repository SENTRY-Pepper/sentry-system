# SENTRY Mobile App
## Timothy Wachira | SCT212-0178/2021
### Android App for Pepper's Tablet — Setup, Development & Deployment Guide

---

## Overview

The SENTRY mobile app is an Android application built in Kotlin that runs on
Pepper's chest tablet. It is the visual and interactive face of the training
system — displaying scenarios, capturing employee responses, and showing
AI-generated feedback received from Derick's middleware.

The app communicates with the FastAPI middleware server over the local WiFi
network. It does not contain any AI logic — it is purely a client that
sends queries and displays responses.

```
Employee
    ↓ taps / speaks
Android App (Pepper Tablet)
    ↓ HTTP POST
FastAPI Middleware (Derick's laptop — port 8000)
    ↓ RAG pipeline
GPT-4 + ChromaDB
    ↑ grounded response JSON
Android App
    ↑ displays feedback to employee
```

---

## Prerequisites

| Tool | Version | Download |
|---|---|---|
| Android Studio | Ladybug or newer | https://developer.android.com/studio |
| JDK | 17 | Bundled with Android Studio |
| Android SDK | API 21+ (Android 5.0) | Via Android Studio SDK Manager |
| ADB | Latest | Bundled with Android Studio |
| Kotlin | 1.9+ | Bundled with Android Studio |

---

## Project Setup in Android Studio

### Step 1: Create the project

1. Open Android Studio
2. **File → New → New Project**
3. Select **Empty Activity**
4. Configure:
   - **Name:** SENTRY
   - **Package name:** `com.sentry.app`
   - **Save location:** `C:\Users\user\Desktop\SENTRY\sentry-system\mobile_app`
   - **Language:** Kotlin
   - **Minimum SDK:** API 21 (Android 5.0) — covers Pepper's tablet
5. Click **Finish**

### Step 2: Add dependencies to `app/build.gradle`

Inside the `dependencies { }` block add:

```gradle
// Retrofit — HTTP client for API calls
implementation 'com.squareup.retrofit2:retrofit:2.9.0'
implementation 'com.squareup.retrofit2:converter-gson:2.9.0'

// OkHttp — underlying HTTP engine + logging
implementation 'com.squareup.okhttp3:okhttp:4.12.0'
implementation 'com.squareup.okhttp3:logging-interceptor:4.12.0'

// Coroutines — async API calls without blocking the UI
implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'

// ViewModel and LiveData — MVVM architecture
implementation 'androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0'
implementation 'androidx.lifecycle:lifecycle-livedata-ktx:2.7.0'

// Jetpack Compose (optional — use if building modern UI)
implementation 'androidx.activity:activity-compose:1.8.2'
implementation 'androidx.compose.ui:ui:1.6.0'
implementation 'androidx.compose.material3:material3:1.2.0'
```

Sync the project after saving: **File → Sync Project with Gradle Files**

### Step 3: Add internet permission to `AndroidManifest.xml`

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

Also add this inside the `<application>` tag to allow HTTP (not just HTTPS)
since the middleware runs on HTTP locally:

```xml
android:usesCleartextTraffic="true"
```

---

## Project File Structure

```
mobile_app/
└── app/
    └── src/
        └── main/
            ├── java/com/sentry/app/
            │   │
            │   ├── network/
            │   │   ├── ApiClient.kt          # Retrofit instance + base URL
            │   │   ├── ApiService.kt         # Endpoint interface definitions
            │   │   └── NetworkResult.kt      # Sealed class for API states
            │   │
            │   ├── models/
            │   │   ├── QueryRequest.kt       # Maps to Derick's request model
            │   │   ├── QueryResponse.kt      # Maps to Derick's response model
            │   │   ├── SessionRequest.kt     # Session start/end payloads
            │   │   └── RetrievedChunk.kt     # Source chunk from RAG pipeline
            │   │
            │   ├── viewmodels/
            │   │   ├── SessionViewModel.kt   # Manages session state
            │   │   └── ScenarioViewModel.kt  # Manages scenario flow
            │   │
            │   ├── ui/
            │   │   ├── WelcomeActivity.kt    # Session start screen
            │   │   ├── ScenarioActivity.kt   # Attack scenario display
            │   │   ├── LoadingActivity.kt    # Waiting for AI response
            │   │   ├── FeedbackActivity.kt   # Grounded AI response display
            │   │   └── ResultsActivity.kt    # Post-session score summary
            │   │
            │   └── MainActivity.kt           # Entry point + navigation
            │
            └── res/
                ├── layout/                   # XML layouts per screen
                ├── values/
                │   ├── colors.xml            # SENTRY brand colours
                │   ├── strings.xml           # All UI text strings
                │   └── themes.xml            # App theme
                └── drawable/                 # Icons and images
```

---

## Connecting to Derick's Middleware

### Finding the middleware IP address

Derick runs the FastAPI server on his laptop. Both the laptop and Pepper's
tablet must be on the **same WiFi network**.

Derick runs this to find his local IP:
```powershell
ipconfig
# Look for: Wireless LAN adapter Wi-Fi → IPv4 Address
# Example: 192.168.1.105
```

### ApiClient.kt

```kotlin
package com.sentry.app.network

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

object ApiClient {

    // ---------------------------------------------------------------
    // IMPORTANT: Change this to Derick's laptop IP before each session
    // Both devices must be on the same WiFi network
    // ---------------------------------------------------------------
    private const val BASE_URL = "http://192.168.x.x:8000/"

    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }

    private val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)  // AI responses can take up to 20s
        .writeTimeout(10, TimeUnit.SECONDS)
        .build()

    val retrofit: Retrofit = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .client(okHttpClient)
        .addConverterFactory(GsonConverterFactory.create())
        .build()
}
```

### ApiService.kt

```kotlin
package com.sentry.app.network

import com.sentry.app.models.QueryRequest
import com.sentry.app.models.QueryResponse
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

interface ApiService {

    // Health check — call on app startup to confirm backend is ready
    @GET("health")
    suspend fun healthCheck(): Response<Map<String, Any>>

    // Grounded RAG query — primary endpoint used during training
    @POST("api/v1/query")
    suspend fun groundedQuery(
        @Body request: QueryRequest
    ): Response<QueryResponse>

    // Baseline query — used during evaluation study control condition
    @POST("api/v1/query/baseline")
    suspend fun baselineQuery(
        @Body request: QueryRequest
    ): Response<QueryResponse>

    // Session management (Timothy's backend routes — add after Derick sets up)
    @POST("api/v1/sessions/start")
    suspend fun startSession(
        @Body request: Map<String, String>
    ): Response<Map<String, Any>>

    @POST("api/v1/sessions/end")
    suspend fun endSession(
        @Body request: Map<String, Any>
    ): Response<Map<String, Any>>
}
```

### Models — matching Derick's API exactly

```kotlin
// QueryRequest.kt
package com.sentry.app.models

import com.google.gson.annotations.SerializedName

data class QueryRequest(
    @SerializedName("query") val query: String,
    @SerializedName("scenario_id") val scenarioId: String? = null,
    @SerializedName("doc_type_filter") val docTypeFilter: String? = null
)

// QueryResponse.kt
package com.sentry.app.models

import com.google.gson.annotations.SerializedName

data class QueryResponse(
    @SerializedName("query") val query: String,
    @SerializedName("mode") val mode: String,
    @SerializedName("response") val response: String,
    @SerializedName("sources") val sources: List<String>,
    @SerializedName("chunks_used") val chunksUsed: Int,
    @SerializedName("total_ms") val totalMs: Double,
    @SerializedName("prompt_tokens") val promptTokens: Int,
    @SerializedName("completion_tokens") val completionTokens: Int,
    @SerializedName("retrieval_ms") val retrievalMs: Double,
    @SerializedName("generation_ms") val generationMs: Double,
    @SerializedName("scenario_id") val scenarioId: String?,
    @SerializedName("retrieved_chunks") val retrievedChunks: List<RetrievedChunk>
)

// RetrievedChunk.kt
package com.sentry.app.models

import com.google.gson.annotations.SerializedName

data class RetrievedChunk(
    @SerializedName("source") val source: String,
    @SerializedName("doc_type") val docType: String,
    @SerializedName("score") val score: Double,
    @SerializedName("chunk_index") val chunkIndex: Int
)
```

---

## Screen Flow

```
WelcomeActivity
    ↓ Employee enters participant ID, app pings /health
ScenarioActivity
    ↓ Displays attack scenario prompt
    ↓ Employee makes decision (button tap or voice)
    ↓ App calls POST /api/v1/query
LoadingActivity  (shown while waiting for AI response)
    ↓ Response received
FeedbackActivity
    ↓ Displays grounded response + source attribution
    ↓ "Based on: OWASP Top 10, Computer Misuse Act"
    ↓ Employee presses Next
    ↓ Loop back to ScenarioActivity for next scenario
ResultsActivity
    ↓ Shows session summary — score, accuracy, risk areas
    ↓ App calls POST /api/v1/sessions/end
```

---

## Figma Design Handoff

1. Design all five screens in Figma
2. Export assets (icons, images) as PNG/SVG into `res/drawable/`
3. Use Figma's **Inspect** panel to get exact:
   - Colour hex codes → `res/values/colors.xml`
   - Font sizes and weights → `res/values/themes.xml`
   - Spacing and padding values → XML layouts
4. Implement layouts in `res/layout/` as XML files matching the Figma designs

---

## ADB Deployment to Pepper's Tablet

### One-time setup on Pepper's tablet

1. On Pepper's tablet go to **Settings → About tablet**
2. Tap **Build number** seven times to enable Developer Options
3. Go to **Settings → Developer Options**
4. Enable **USB Debugging**

### Connect and deploy

```bash
# Connect laptop to Pepper's tablet via USB cable

# Verify tablet is detected
adb devices
# Should show something like: HT1234567890    device

# Install the APK (built from Android Studio: Build → Build APK)
adb install app\build\outputs\apk\debug\app-debug.apk

# Launch the app
adb shell am start -n com.sentry.app/.MainActivity

# To update (uninstall first)
adb uninstall com.sentry.app
adb install app\build\outputs\apk\debug\app-debug.apk

# View logs in real time (useful for debugging)
adb logcat -s "SENTRY"
```

### Build the APK in Android Studio

1. **Build → Build Bundle(s) / APK(s) → Build APK(s)**
2. APK location: `app/build/outputs/apk/debug/app-debug.apk`
3. Android Studio will show a notification with a **locate** link when done

---

## Network Checklist Before Each Session

- [ ] Derick's laptop and Pepper's tablet on the same WiFi network
- [ ] Middleware server running: `uvicorn middleware.main:app --host 0.0.0.0 --port 8000`
- [ ] `BASE_URL` in `ApiClient.kt` matches Derick's current laptop IP
- [ ] `/health` returns `pipeline_ready: true`
- [ ] Windows Firewall allows inbound connections on port 8000

### Allow port 8000 through Windows Firewall (run once)

```powershell
# Run as Administrator
New-NetFirewallRule -DisplayName "SENTRY Middleware" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 8000 `
    -Action Allow
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| App cannot connect to server | Wrong IP or different WiFi | Check `ipconfig`, update `BASE_URL` |
| `adb devices` shows nothing | USB debugging not enabled | Enable in Developer Options |
| APK installs but crashes | API level mismatch | Check `minSdk` in `build.gradle` |
| Responses take too long | GPT-4 latency | Show `LoadingActivity` with animation |
| `CLEARTEXT not permitted` | Missing manifest flag | Add `usesCleartextTraffic="true"` |