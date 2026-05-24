-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt
-keep,includedescriptorclasses class com.sentry.app.**$$serializer { *; }
-keepclassmembers class com.sentry.app.** { *** Companion; }
-keepclasseswithmembers class com.sentry.app.** {
    kotlinx.serialization.KSerializer serializer(...);
}
-keepattributes Signature
-keepattributes Exceptions
-keep class retrofit2.** { *; }
-keepclasseswithmembers class * { @retrofit2.http.* <methods>; }
-dontwarn okhttp3.**
-dontwarn okio.**
-keep class okhttp3.** { *; }
-keep class com.sentry.app.models.** { *; }
-keep class com.sentry.app.network.api.** { *; }
-assumenosideeffects class timber.log.Timber {
    public static *** v(...);
    public static *** d(...);
    public static *** i(...);
}