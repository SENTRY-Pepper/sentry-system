package com.sentry.app.features.trainee.curriculum

object OwaspCurriculum {
    val modules: List<OwaspTrainingModule> = listOf(
        module(
            id = "owasp-a01-broken-access-control",
            owaspId = "A01",
            title = "Broken Access Control",
            difficulty = "Foundation",
            summary = "Access control makes sure people can only view or change the data and functions they are allowed to use.",
            workplaceTakeaway = "Never bypass approval, change IDs in URLs, or use someone else's access. Report access that looks too broad.",
            sourceReference = "A01_2025-Broken_Access_Control.md",
            questions = listOf(
                question(
                    id = "owasp-a01-q1-url-tampering",
                    scenario = "You work in HR and receive a link to view your own leave record. You notice the URL ends with employeeId=1042. A colleague asks you to change the number to 1043 to check whether their record appears because they cannot log in. What should you do?",
                    correctAnswerId = "report",
                    options = listOf(
                        option("try_it", "Change the employee ID quickly to see if the system blocks you, then undo it if it works.", false, "This is the wrong choice. Testing another person's record without authorization may expose private data. OWASP A01 treats this as an access control risk."),
                        option("share_link", "Send your link to the colleague so they can try it themselves.", false, "This spreads a potentially vulnerable link and may expose your session context. Access issues should be escalated, not shared."),
                        option("report", "Do not change the URL. Report the possible access control issue to IT security.", true, "Correct. Server-side authorization should prevent access outside your permissions. Reporting lets IT verify and fix the access control weakness safely."),
                        option("screenshot", "Try the change, take screenshots as proof, then report it.", false, "This still creates unauthorized access. Evidence should be collected safely by the security team, not by accessing another person's record."),
                    ),
                ),
                question(
                    id = "owasp-a01-q2-shared-admin",
                    scenario = "A team lead shares an administrator account so staff can approve urgent customer refunds while the manager is away. You are asked to use it for one transaction. What is the safest response?",
                    correctAnswerId = "refuse_escalate",
                    options = listOf(
                        option("refuse_escalate", "Refuse to use the shared admin account and escalate the access request through the approved process.", true, "Correct. Shared privileged accounts remove accountability and can bypass proper authorization. The right action is to use approved access escalation."),
                        option("use_once", "Use it once because the refund is urgent, then write down who asked you to do it for later audit notes.", false, "Urgency does not justify bypassing access control. OWASP A01 focuses on preventing users from acting outside assigned permissions."),
                        option("change_password", "Use the account, then change the password afterward.", false, "Changing the password afterward does not fix the unauthorized privileged action or lack of accountability."),
                        option("ask_customer", "Ask the customer for permission before using the admin account.", false, "Customer permission does not authorize internal privileged access. The access request must follow the organisation's control process."),
                    ),
                ),
            ),
        ),
        module(
            id = "owasp-a02-security-misconfiguration",
            owaspId = "A02",
            title = "Security Misconfiguration",
            difficulty = "Foundation",
            summary = "Security misconfiguration happens when systems are deployed with unsafe defaults, exposed admin tools, verbose errors, or unnecessary features.",
            workplaceTakeaway = "Treat default passwords, sample apps, exposed dashboards, and detailed errors as security issues that must be fixed or escalated.",
            sourceReference = "A02_2025-Security_Misconfiguration.md",
            questions = listOf(
                question(
                    id = "owasp-a02-q1-default-admin",
                    scenario = "A vendor installs a new internal reporting tool. During testing you find an admin page still using the default username and password from the product manual. The team wants to leave it until after launch. What should you do?",
                    correctAnswerId = "block_launch",
                    options = listOf(
                        option("leave_it", "Leave it for now because the tool is only internal and launch access is limited to staff.", false, "Internal systems can still be reached by attackers or misused by insiders. Default credentials are a serious misconfiguration."),
                        option("email_password", "Email the default password to the team so everyone remembers to change it later.", false, "This distributes a known password and increases exposure. Unsafe defaults must be removed, not circulated."),
                        option("hide_url", "Hide the admin URL from normal users and launch anyway.", false, "Hidden URLs are not access control. Attackers can discover paths and default accounts."),
                        option("block_launch", "Escalate it as a release blocker until the default account is disabled or changed.", true, "Correct. OWASP A02 includes default accounts and exposed admin consoles as classic security misconfigurations."),
                    ),
                ),
                question(
                    id = "owasp-a02-q2-error-message",
                    scenario = "A web form crashes and displays a full database connection string, table name, and server path to normal users. A developer says detailed errors help support troubleshoot faster. What should happen?",
                    correctAnswerId = "hide_log",
                    options = listOf(
                        option("screenshot_public", "Post the screenshot in a public team chat for quick help and tag the support group on duty.", false, "Detailed errors can expose system internals. Posting them publicly widens the disclosure."),
                        option("hide_log", "Show a generic user error, log the details securely, and report the configuration issue.", true, "Correct. Users should not see sensitive technical detail. Logs should capture the detail for authorized support teams."),
                        option("ignore", "Ignore it because it only appears when the form crashes.", false, "Rare errors can still leak sensitive implementation detail. OWASP A02 treats verbose error output as misconfiguration."),
                        option("tell_users_retry", "Tell users to retry until the error disappears.", false, "Retrying does not fix the exposed detail or the insecure error handling."),
                    ),
                ),
            ),
        ),
        module(
            id = "owasp-a03-software-supply-chain",
            owaspId = "A03",
            title = "Software Supply Chain Failures",
            difficulty = "Intermediate",
            summary = "Software supply chain failures involve unsafe dependencies, untrusted packages, compromised vendors, or missing component monitoring.",
            workplaceTakeaway = "Install software and updates only from approved sources, and report unexpected package or vendor changes.",
            sourceReference = "A03_2025-Software_Supply_Chain_Failures.md",
            questions = listOf(
                question(
                    id = "owasp-a03-q1-browser-plugin",
                    scenario = "A colleague shares a free browser plugin that promises to export customer records faster than the approved tool. It is not listed in the company software catalog. What should you do?",
                    correctAnswerId = "verify_source",
                    options = listOf(
                        option("verify_source", "Do not install it. Ask IT/security to verify and approve the tool first.", true, "Correct. Untrusted components can run with user privileges. Formal approval reduces supply chain risk."),
                        option("install", "Install it because reviews suggest it is safe and remove it later if there are complaints.", false, "Public reviews do not prove integrity or suitability for business data."),
                        option("test_personal", "Try it on a personal laptop first, then install it at work if it works.", false, "Personal testing does not verify integrity and may expose personal or business data."),
                        option("ask_colleague", "Install it only if the colleague confirms they already used it.", false, "A colleague's experience is not a supply chain control. Use trusted, approved sources."),
                    ),
                ),
                question(
                    id = "owasp-a03-q2-package-update",
                    scenario = "A project dependency update appears from a maintainer account you do not recognize. The update requests new network permissions and was published minutes ago. What is the best response?",
                    correctAnswerId = "pause_review",
                    options = listOf(
                        option("merge_now", "Merge it quickly so the project stays up to date and monitor for errors afterward in staging.", false, "Fast updates are useful, but unusual maintainers and new permissions require review."),
                        option("ask_chat", "Ask in chat if anyone has heard of the maintainer, then merge if nobody objects.", false, "Silence is not verification. Supply chain changes need an explicit review process."),
                        option("pause_review", "Pause the update and request dependency review, provenance checks, and security approval.", true, "Correct. OWASP A03 emphasizes trusted components, provenance, and monitoring for dependency risk."),
                        option("disable_tests", "Disable failing checks and test manually later.", false, "Weakening checks increases the chance of accepting malicious or vulnerable code."),
                    ),
                ),
            ),
        ),
        module(
            id = "owasp-a04-cryptographic-failures",
            owaspId = "A04",
            title = "Cryptographic Failures",
            difficulty = "Intermediate",
            summary = "Cryptographic failures happen when sensitive data is not properly protected in storage or transit.",
            workplaceTakeaway = "Use approved encrypted channels and never send sensitive business data through insecure links or unapproved storage.",
            sourceReference = "A04_2025-Cryptographic_Failures.md",
            questions = listOf(
                question(
                    id = "owasp-a04-q1-public-file-link",
                    scenario = "A manager asks you to send customer IDs and phone numbers to an external consultant. Their email rejects large attachments, so they suggest a public file-sharing link without a password. What should you do?",
                    correctAnswerId = "secure_channel",
                    options = listOf(
                        option("public_link", "Upload it to the public link because the consultant needs it urgently and delete it later.", false, "An unprotected public link can expose sensitive customer data."),
                        option("secure_channel", "Use the approved encrypted file transfer method or ask IT for a secure alternative.", true, "Correct. Sensitive data must be protected in transit and storage with approved cryptographic controls."),
                        option("rename_file", "Rename the file to something harmless before uploading it.", false, "Renaming is not encryption and does not protect the data."),
                        option("send_partial", "Split the spreadsheet into smaller files and email them normally.", false, "Smaller files are still sensitive. Protection is about confidentiality, not file size."),
                    ),
                ),
                question(
                    id = "owasp-a04-q2-http-login",
                    scenario = "You notice a staff portal login page is loaded over HTTP instead of HTTPS while employees enter usernames and passwords. What should you do?",
                    correctAnswerId = "stop_report",
                    options = listOf(
                        option("login_fast", "Log in quickly before anyone can intercept it.", false, "Speed does not protect credentials sent over an insecure channel."),
                        option("use_vpn_only", "Use it only while connected to VPN and ignore the warning until maintenance is over.", false, "VPN may reduce exposure, but the login page still lacks proper transport protection."),
                        option("tell_team", "Tell colleagues privately to be careful but keep using it.", false, "Warnings do not fix the cryptographic failure."),
                        option("stop_report", "Stop using the page and report that credentials are being submitted without HTTPS.", true, "Correct. Login credentials must be protected with secure transport such as HTTPS."),
                    ),
                ),
            ),
        ),
        module(
            id = "owasp-a05-injection",
            owaspId = "A05",
            title = "Injection",
            difficulty = "Intermediate",
            summary = "Injection happens when untrusted input is treated as a command or query instead of data.",
            workplaceTakeaway = "Never paste unverified commands or query fragments into business systems. Report forms that behave strangely with special characters.",
            sourceReference = "A05_2025-Injection.md",
            questions = listOf(
                question(
                    id = "owasp-a05-q1-search-payload",
                    scenario = "A support dashboard lets staff search customer records. A user sends you the text `' OR '1'='1` and says to paste it into the search box to find their missing record faster. What should you do?",
                    correctAnswerId = "report_input",
                    options = listOf(
                        option("report_input", "Do not paste it. Report the suspicious input and search using normal customer identifiers.", true, "Correct. The text resembles an injection payload. Treat it as suspicious and report it."),
                        option("paste", "Paste the text exactly as given to see whether it finds the record, then clear the search history.", false, "Running suspicious input in production can expose or damage data."),
                        option("edit_payload", "Remove the quotes and try a simplified version.", false, "Editing suspicious input does not make it safe."),
                        option("try_test", "Try it only in a low-traffic period so fewer users are affected.", false, "Timing does not remove injection risk."),
                    ),
                ),
                question(
                    id = "owasp-a05-q2-command-copy",
                    scenario = "A forum post suggests fixing a laptop issue by copying a long PowerShell command into the terminal. The command downloads a script from an unknown URL. What is the safest action?",
                    correctAnswerId = "approved_help",
                    options = listOf(
                        option("run_admin", "Run it as administrator so the fix has permission to work and save the command for IT.", false, "This gives unknown code high privileges and may compromise the machine."),
                        option("send_colleague", "Ask a colleague to run it first and confirm the result.", false, "Moving the risk to another person is not a control."),
                        option("approved_help", "Do not run it. Use approved IT support or verified internal remediation instructions.", true, "Correct. Unknown commands and scripts can become command injection or malware execution paths."),
                        option("read_first", "Skim the command and run it if it looks technical.", false, "A quick skim is not reliable verification for downloaded scripts."),
                    ),
                ),
            ),
        ),
        module(
            id = "owasp-a06-insecure-design",
            owaspId = "A06",
            title = "Insecure Design",
            difficulty = "Advanced",
            summary = "Insecure design means the system workflow itself allows abuse, even if the code works as written.",
            workplaceTakeaway = "When a process allows obvious misuse, escalate the design risk instead of treating it as only a user mistake.",
            sourceReference = "A06_2025-Insecure_Design.md",
            questions = listOf(
                question(
                    id = "owasp-a06-q1-referral-abuse",
                    scenario = "A promotion system gives a discount for each referral. You notice the same employee can create unlimited referral codes for themselves because the workflow never checks ownership. What should you do?",
                    correctAnswerId = "raise_design",
                    options = listOf(
                        option("use_codes", "Use a few referral codes because the system allows it and stop before it looks excessive.", false, "This abuses a business logic flaw."),
                        option("ignore", "Ignore it because there is no error message.", false, "A design flaw may not produce an error. The workflow can still be unsafe."),
                        option("tell_team", "Tell colleagues informally so everyone avoids using the loophole.", false, "Informal warnings do not fix insecure design."),
                        option("raise_design", "Report the workflow as a business logic risk and recommend abuse checks.", true, "Correct. Ownership checks, limits, and abuse controls should be designed into the workflow."),
                    ),
                ),
                question(
                    id = "owasp-a06-q2-approval-bypass",
                    scenario = "An expense system lets employees split one large purchase into many smaller claims to avoid manager approval. The system accepts the claims without checking total spend. What should you do?",
                    correctAnswerId = "design_control",
                    options = listOf(
                        option("split_claim", "Split your own claim because the system permits it.", false, "Using the loophole is abuse of an insecure workflow."),
                        option("design_control", "Escalate the design flaw and recommend aggregate approval limits or abuse checks.", true, "Correct. This is a business logic design weakness that needs a control in the process."),
                        option("warn_people", "Tell people not to misuse it and leave the system unchanged.", false, "Policy reminders alone do not prevent repeatable design abuse."),
                        option("manual_spreadsheet", "Track suspected abuse manually at the end of each month and email repeated offenders.", false, "Manual after-the-fact review is weaker than designing preventive controls."),
                    ),
                ),
            ),
        ),
        module(
            id = "owasp-a07-authentication-failures",
            owaspId = "A07",
            title = "Authentication Failures",
            difficulty = "Foundation",
            summary = "Authentication failures allow attackers to guess, reuse, steal, or bypass credentials and sessions.",
            workplaceTakeaway = "Use unique passwords, MFA, verified login pages, and immediate reporting for suspicious sign-in activity.",
            sourceReference = "A07_2025-Authentication_Failures.md",
            questions = listOf(
                question(
                    id = "owasp-a07-q1-mfa-fatigue",
                    scenario = "You receive an MFA approval prompt on your phone while you are not logging in. A minute later, another prompt appears. What should you do?",
                    correctAnswerId = "deny_report",
                    options = listOf(
                        option("approve", "Approve one prompt to stop the notifications.", false, "Approving may give an attacker access."),
                        option("ignore", "Ignore the prompts and continue working unless another prompt appears after lunch.", false, "Repeated prompts may indicate an active account takeover attempt."),
                        option("deny_report", "Deny the prompts and report the suspicious login attempt immediately.", true, "Correct. Unexpected MFA prompts should be denied and reported."),
                        option("ask_chat", "Ask in a public chat whether anyone else is seeing prompts.", false, "Public chat can delay proper incident handling."),
                    ),
                ),
                question(
                    id = "owasp-a07-q2-password-reuse",
                    scenario = "A work service asks you to create a password. You are tempted to reuse the same password from your personal email because it is easy to remember. What should you do?",
                    correctAnswerId = "unique_mfa",
                    options = listOf(
                        option("unique_mfa", "Use a unique password or passphrase and enable MFA where available.", true, "Correct. Unique credentials and MFA reduce the impact of password theft or credential stuffing."),
                        option("reuse", "Reuse the personal password because you trust the work service.", false, "Password reuse lets one breach compromise multiple accounts."),
                        option("minor_change", "Add one extra character to the personal password.", false, "Small variations are often guessable and still create reuse risk."),
                        option("share_manager", "Ask your manager to store the password in case you forget it during urgent work.", false, "Passwords should not be shared. Use approved password management."),
                    ),
                ),
            ),
        ),
        module(
            id = "owasp-a08-integrity-failures",
            owaspId = "A08",
            title = "Software or Data Integrity Failures",
            difficulty = "Advanced",
            summary = "Integrity failures happen when software, updates, plugins, or data are trusted without verifying that they are authentic and unchanged.",
            workplaceTakeaway = "Do not run unsigned updates, untrusted plugins, or modified data files outside approved verification processes.",
            sourceReference = "A08_2025-Software_or_Data_Integrity_Failures.md",
            questions = listOf(
                question(
                    id = "owasp-a08-q1-unsigned-firmware",
                    scenario = "A router used by your branch office shows a firmware update from an unofficial website. The file is unsigned and not from the vendor portal. What should you do?",
                    correctAnswerId = "vendor_only",
                    options = listOf(
                        option("install", "Install it because performance problems affect business operations and users are waiting for service.", false, "Unsigned firmware can be malicious or modified."),
                        option("vendor_only", "Do not install it. Use only vendor-approved, signed firmware through the official process.", true, "Correct. Signed vendor updates protect software integrity."),
                        option("scan", "Download it and rely on antivirus scanning before installing.", false, "Scanning is not a substitute for authenticity and integrity checks."),
                        option("install_after_hours", "Install it after hours so any disruption affects fewer users.", false, "Timing does not solve integrity risk."),
                    ),
                ),
                question(
                    id = "owasp-a08-q2-spreadsheet-macro",
                    scenario = "A supplier sends a pricing spreadsheet with macros enabled. The email asks you to click Enable Content before viewing the prices. The sender address is slightly different from the usual supplier domain. What should you do?",
                    correctAnswerId = "verify_supplier",
                    options = listOf(
                        option("enable", "Enable macros because the prices are needed today and the spreadsheet layout looks normal enough to trust.", false, "Macros can execute code and may be used to tamper with systems or data."),
                        option("forward_team", "Forward it to the team so someone else can open it.", false, "Forwarding spreads the risk."),
                        option("open_offline", "Disconnect from WiFi, enable macros, and check the file.", false, "Offline opening does not verify authenticity and may still compromise the device."),
                        option("verify_supplier", "Do not enable macros. Verify the sender through an official channel and report the suspicious file.", true, "Correct. Integrity checks and trusted communication channels help prevent tampered files from being trusted."),
                    ),
                ),
            ),
        ),
        module(
            id = "owasp-a09-logging-alerting",
            owaspId = "A09",
            title = "Security Logging and Alerting Failures",
            difficulty = "Intermediate",
            summary = "Logging and alerting failures prevent teams from detecting, investigating, and responding to attacks.",
            workplaceTakeaway = "Escalate suspicious activity through channels that create a record. Do not handle security incidents only by word of mouth.",
            sourceReference = "A09_2025-Security_Logging_and_Alerting_Failures.md",
            questions = listOf(
                question(
                    id = "owasp-a09-q1-suspicious-login",
                    scenario = "You see three failed admin login attempts followed by a successful login from an unusual country in the dashboard audit view. The manager says to mention it verbally tomorrow. What should you do?",
                    correctAnswerId = "create_incident",
                    options = listOf(
                        option("wait", "Wait for the meeting because the login eventually succeeded.", false, "This delays response to a suspicious login pattern."),
                        option("verbal", "Tell a colleague verbally and assume they will handle it.", false, "Verbal reporting may leave no audit trail."),
                        option("disable_account", "Disable the account yourself without following the incident process and tell the manager later.", false, "Containment should follow the approved incident process and preserve evidence."),
                        option("create_incident", "Create or escalate a security incident with the relevant log details immediately.", true, "Correct. A recorded incident gives the security team evidence to investigate quickly."),
                    ),
                ),
                question(
                    id = "owasp-a09-q2-missing-logs",
                    scenario = "After a suspected data export, the team discovers the export tool does not log who downloaded files or when. What is the best response?",
                    correctAnswerId = "add_monitoring",
                    options = listOf(
                        option("ask_memory", "Ask users to remember whether they downloaded anything.", false, "Memory is not a reliable audit trail."),
                        option("add_monitoring", "Raise a logging and alerting gap so file exports record user, time, file, and unusual volume.", true, "Correct. Security-relevant actions need logs that support detection and investigation."),
                        option("disable_all", "Disable every export immediately and permanently, including approved business reports and audits.", false, "Temporary containment may be needed, but the best long-term fix is monitored, controlled exporting."),
                        option("ignore", "Ignore it if no customer complaint has arrived.", false, "Lack of complaints does not mean there was no incident."),
                    ),
                ),
            ),
        ),
        module(
            id = "owasp-a10-exceptional-conditions",
            owaspId = "A10",
            title = "Mishandling Exceptional Conditions",
            difficulty = "Advanced",
            summary = "Exceptional condition failures occur when errors, failures, race conditions, or partial transactions expose data or leave systems in unsafe states.",
            workplaceTakeaway = "Treat detailed errors, failed transactions, and repeated crashes as security signals, not just technical annoyances.",
            sourceReference = "A10_2025-Mishandling_of_Exceptional_Conditions.md",
            questions = listOf(
                question(
                    id = "owasp-a10-q1-payment-retry",
                    scenario = "During payment processing, a network error appears after the debit step. The system shows a detailed database error and lets you click retry repeatedly. What should you do?",
                    correctAnswerId = "stop_report",
                    options = listOf(
                        option("stop_report", "Stop retrying, preserve the reference number, and report the failed transaction securely.", true, "Correct. Failed transaction states can create duplicates or expose sensitive error detail."),
                        option("retry_many", "Click retry several times until the transaction completes and screenshot each failure message.", false, "Repeated retries can worsen partial transactions or duplicate processing."),
                        option("copy_error_public", "Post the full database error in a public team chat for help.", false, "Detailed errors can expose sensitive system information."),
                        option("refresh", "Refresh the browser and start the payment again from the beginning.", false, "Starting over may create duplicate or inconsistent transactions."),
                    ),
                ),
                question(
                    id = "owasp-a10-q2-race-condition",
                    scenario = "A ticketing app crashes when two staff members approve the same refund at nearly the same time. Sometimes both approvals are accepted. What should you do?",
                    correctAnswerId = "report_race",
                    options = listOf(
                        option("coordinate_chat", "Tell staff to coordinate in chat before approving refunds during busy support hours today only for now.", false, "Manual coordination is weak and does not fix the unsafe concurrent state."),
                        option("approve_fast", "Approve faster so your action wins before the crash.", false, "Trying to win a race condition can worsen the integrity problem."),
                        option("report_race", "Report the race condition and recommend safe transaction locking or duplicate-approval controls.", true, "Correct. Exceptional concurrent states should be handled safely by design and implementation."),
                        option("ignore_rare", "Ignore it because simultaneous approval is rare.", false, "Rare exceptional states can still create financial and audit risk."),
                    ),
                ),
            ),
        ),
    )

    val totalModules: Int = modules.size
    val totalQuestions: Int = modules.sumOf { it.questions.size }

    fun findModule(id: String): OwaspTrainingModule? =
        modules.firstOrNull { it.id == id }

    private fun module(
        id: String,
        owaspId: String,
        title: String,
        difficulty: String,
        summary: String,
        workplaceTakeaway: String,
        sourceReference: String,
        questions: List<OwaspQuestion>,
    ) = OwaspTrainingModule(
        id = id,
        owaspId = owaspId,
        title = title,
        difficulty = difficulty,
        summary = summary,
        workplaceTakeaway = workplaceTakeaway,
        questions = questions,
        sourceReference = sourceReference,
    )

    private fun question(
        id: String,
        scenario: String,
        correctAnswerId: String,
        options: List<OwaspAnswerOption>,
    ) = OwaspQuestion(
        id = id,
        scenario = scenario,
        correctAnswerId = correctAnswerId,
        options = options.mapIndexed { index, option ->
            option.copy(label = listOf("A", "B", "C", "D")[index])
        },
    )

    private fun option(
        id: String,
        text: String,
        isCorrect: Boolean,
        feedback: String,
    ) = OwaspAnswerOption(
        id = id,
        label = "",
        text = text,
        isCorrect = isCorrect,
        feedback = feedback,
    )
}
