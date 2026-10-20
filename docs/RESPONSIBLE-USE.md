# Responsible Use Policy

ABISGPT is dual-use technology. Biometric infrastructure can automate
lawful, consent-based identity workflows (eKYC, border control, banking
onboarding, device unlock) **and** can be repurposed to conduct mass
surveillance without anyone's knowledge or consent.

## Prohibited use

You may **not** use ABISGPT, in whole or in derived form, to:

1. Identify or track an individual **without their informed consent** and a
   lawful basis under applicable data-protection law.
2. Build covert surveillance of protests, religious practice, sexual
   orientation, immigration status, or political affiliation.
3. Infer sensitive attributes (health, race, emotion, criminality) from
   biometric samples outside a strictly consented, narrowly-scoped workflow.
4. Deploy in a jurisdiction where usage is prohibited without an appropriate
   licensing or legal-compliance regime in place.

## Required controls for any real-world deployment

- Explicit notice and consent capture at the point of collection.
- Published retention schedule; hard deletion of raw media post-inference.
- Human review in the loop for consequential decisions (denials, detentions,
  account lockouts).
- Independent bias audit across demographic groups before production
  rollout; publish the results.
- Regulation-specific compliance sign-off: GDPR Art. 9 + DPIA, CCPA, BIPA
  (state statutory damages apply), India DPDP Act, PIPL, etc.

## Fairness statement

We do not claim demographic parity for the reference encoders. Face and
fingerprint error rates are known to vary by skin tone, age, and gender for
most published models. Any use in consequential contexts demands per-subgroup
FMR/FNMR measurement against your own population, not this repo's synthetic
tests.

## Liveness claim boundaries

Combining rPPG + VLM semantic scoring raises the bar against *scalable* spoof
attacks (printed photos, screen replays, single-frame deepfakes), but no
passive PAD stack is 100% robust:

- High-end hardware wax / silicone masks remain challenging without ToF or
  multi-spectral depth.
- Dedicated high-quality injection attacks (camera-hijack middleware, HDMI
  replay) defeat software-only liveness — hardware attestation is required.
- AI-generated video with rendered vascular plausibility is an active
  threat; we recommend pairing this stack with certified challenge-response
  in high-assurance settings (ISO/IEC 30107-3 L2/L3 in-scope only when a
  certified lab confirms it).

Treat all claims as a research *baseline*, not a guarantee.

