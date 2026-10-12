# Data Directory

This directory is intended for authorized sample video recordings used during development, testing, and live demonstrations.

## Directory Structure

```text
data/
├── README.md
└── sample_videos/
    └── .gitkeep
```

## Recommended Test Videos

For a comprehensive evaluation during HackZen 2.0, place test recordings into `data/sample_videos/`:

1. `single_person_entry.mp4`: A single person entering the gate with a clearly visible ID card.
2. `crowd_entry_normal.mp4`: 3-5 students entering simultaneously, with some overlap and visible badges.
3. `occluded_id.mp4`: A student with a bag strap partially covering the ID card.
4. `no_id_entry.mp4`: A student entering without an ID card (to verify the `ID not detected — verify` review prompt).
5. `low_light_blur.mp4`: Fast motion or low illumination causing frame motion blur (to verify the `Uncertain — manual review` state).

## Privacy & Safety Notice

- **No Student PII:** Do not upload videos containing readable student roll numbers, phone numbers, or residential addresses.
- **Authorized Recordings Only:** Use staged recordings with participant consent.
- **Local Storage Only:** Videos are processed strictly on the local machine and are never uploaded to any remote server or cloud API.
