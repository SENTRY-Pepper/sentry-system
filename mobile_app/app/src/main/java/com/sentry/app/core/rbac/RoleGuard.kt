package com.sentry.app.core.rbac

import com.sentry.app.core.navigation.UserRole

/**
 * Centralised role-based access control.
 * All permission checks go through here — never scattered across screens.
 */
object RoleGuard {

    fun canAccessAdminDashboard(role: UserRole) = role == UserRole.ADMIN
    fun canStartSession(role: UserRole)          = role == UserRole.TRAINEE
    fun canViewOwnResults(role: UserRole)        = role == UserRole.TRAINEE
    fun canAccessChat(role: UserRole)            = true  // both roles
    fun canAccessSettings(role: UserRole)        = true  // both roles

    /**
     * Throws if the role does not have permission.
     * Call this at the top of any ViewModel that serves admin-only data.
     */
    fun requireAdmin(role: UserRole) {
        check(role == UserRole.ADMIN) {
            "Access denied: admin role required"
        }
    }
}