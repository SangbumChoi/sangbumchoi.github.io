---
title: Portfolio Conversation Collector
sdk: docker
app_port: 7860
---

Write-only opt-in collector. Set HF_TOKEN as a Space secret, and set
CONVERSATION_HF_REPO and CONVERSATION_ORIGINS as environment variables.
The dataset must already exist and be private. A successful receipt means
the Hub upload completed; no visitor audio is captured.

This service requires a public ingress with request-rate and body-size limits
before it is exposed to the internet. CORS is not authentication. Deploy the
Dockerfile from the repository root so the feedback package is available.
