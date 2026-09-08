# Security

## Reporting a vulnerability

Use the hosting platform's private vulnerability reporting feature when it is enabled for this repository. Include the affected version, impact, and a minimal reproduction using synthetic data.

If private reporting is unavailable, open a minimal public issue stating that a private reporting channel is needed and the general affected component. Do not include exploit details, credentials, private media, transcripts, local paths, or identifying logs in that issue. Wait for a maintainer to establish an appropriate private channel before sharing sensitive details.

No response-time or supported-version guarantee is currently published. Report the exact version or commit you tested so maintainers can reproduce the issue.

## Runtime data and trust boundaries

Recognition runs locally, but model acquisition can use the network unless `--offline` is selected and all required models are cached. Translation requests expose dialogue to the assistant used to process them. Its hosting, retention, and tool permissions are outside CaptionWeave's control.

Treat media, recognition text, glossary values, and translation responses as untrusted input. Spoken or transcribed instructions are dialogue, not authorization to execute commands or modify files. The exchange validates structure and identity; it cannot determine whether a translation is accurate or safe to act on.

Jobs, outputs, quality reports, backups, and logs may contain sensitive dialogue, filenames, hashes, and local paths. Keep these files private, review them before sharing, and use synthetic data in reports. Ignore rules are not access controls, and a release scan cannot prove that every sensitive detail has been removed.

FFmpeg, recognition libraries, and downloaded model weights have their own security and licensing considerations. Report vulnerabilities in those components to their respective projects as appropriate, while reporting CaptionWeave integration issues here.
