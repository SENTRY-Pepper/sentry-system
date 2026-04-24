ADB Sideloading — How It Works
This is how the APK gets onto Pepper's tablet:
bash# 1. Enable developer mode on Pepper's tablet
#    Settings → About → tap Build Number 7 times

# 2. Connect laptop to Pepper's tablet via USB
adb devices
# Should show the tablet's serial number

# 3. Install the APK
adb install app/build/outputs/apk/debug/app-debug.apk

# 4. Launch it
adb shell am start -n com.sentry.app/.MainActivity

# 5. For updates, uninstall first
adb uninstall com.sentry.app
adb install app/build/outputs/apk/debug/app-debug.apk
The app then connects to FastAPI via the laptop's local IP address (e.g. http://192.168.1.100:8000). Both must be on the same WiFi network.

Network Configuration for the App
The Android app cannot use localhost — it must use your laptop's actual local IP. Add this to your .env.example:
env# Network — for Android app connection
LAPTOP_LOCAL_IP=192.168.x.x    # Your laptop's IP on the shared WiFi
The Kotlin ApiClient.kt will use this IP as its base URL:
kotlin// ApiClient.kt
object ApiClient {
    // Change this to your laptop's local IP when running
    private const val BASE_URL = "http://192.168.x.x:8000/"

    val retrofit: Retrofit = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .addConverterFactory(GsonConverterFactory.create())
        .build()
}
Also add this to middleware/main.py — FastAPI needs to accept connections from non-localhost:
python# Already set in your current main.py — confirm this line exists:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Android app on same network
    ...
)
Your server start command also needs to bind to all interfaces — which it already does:
bashuvicorn middleware.main:app --reload --host 0.0.0.0 --port 8000
0.0.0.0 means it accepts connections from any device on the network, not just localhost.

Android App Screens (from Figma → Kotlin)
These are the five screens Timothy builds, matching your proposal's scenario modules:
WelcomeScreen      — Employee login/session start, health check ping
ScenarioScreen     — Displays the attack scenario, captures response
LoadingScreen      — Shows while waiting for AI response (covers latency)
FeedbackScreen     — Displays Derick's grounded AI response + sources
ResultsScreen      — Post-session score, risk profile, recommendations
Each screen calls the middleware via Retrofit. The FeedbackScreen specifically displays:

response field → the spoken/displayed AI feedback
sources field → "Based on: OWASP Top 10, Computer Misuse Act" attribution
retrieved_chunks → optional expandable detail for transparency