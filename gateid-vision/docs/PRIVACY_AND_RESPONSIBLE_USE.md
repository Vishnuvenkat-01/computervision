# Responsible AI, Privacy & Ethical Governance Guide
**GateID Vision — Campus Entry ID Card Monitoring**

---

## 1. Ethical Mission & Fundamental Principles

GateID Vision is built under a strict design mandate: **Decision Support, Not Automated Discipline**.

Campus entrances are busy public thoroughfares where students, staff, and visitors pass continuously. Automated surveillance systems in educational settings risk overreach if deployed without clear boundaries. GateID Vision addresses this by adhering to four core ethical pillars:

```
[1. Non-Biometric]       No facial recognition or identity extraction
[2. Decision-Support]    Assists human guards; does not deny entry or punish
[3. Objective Fallback]  Missing detection != deliberate violation
[4. Data Minimization]   Transient tracking IDs; configurable retention
```

---

## 2. Core Privacy Guarantees

### 2.1 Strictly Non-Biometric Processing
* **No Face Recognition:** The system does not analyze facial landmarks, extract biometric embeddings, or match faces against student databases.
* **No OCR on Personal Details:** The system localizes the physical presence of an ID badge or lanyard. It **does not read** student names, registration numbers, contact numbers, or blood groups.

### 2.2 Transient Tracking Identifiers
* Entrants are assigned temporary tracking labels (`Person 1`, `Person 2`) solely to maintain tracking across successive frames.
* Tracking IDs are volatile, reset upon video completion, and cannot be correlated across different recordings or dates.

### 2.3 Absence of Automated Denial or Penalties
* GateID Vision **never denies gate access** or generates automatic disciplinary citations.
* A flag of `verification_required` is merely a prompt for physical security staff to perform a courtesy check at the turnstile.

---

## 3. The Non-Detection Philosophy

A fundamental principle implemented in GateID Vision is:
$$\text{Non-Detection } \neq \text{ Proof of Non-Compliance}$$

There are numerous legitimate physical reasons why an ID card may not be detected in a recorded video:
1. The lanyard strap is tucked inside a jacket, hoodie, or heavy winter coat.
2. A backpack strap, tote bag, or laptop sleeve obscures the chest area.
3. The entrant momentarily turns away from the camera or bends down to adjust a shoe.
4. Glare from evening gate floodlights reflects across the plastic card pouch.
5. The card flipped around, showing an unprinted magnetic stripe.

Therefore, GateID Vision categorizes these scenarios as **`uncertain`** or **`verification_required`**, explicitly instructing staff that visual occlusion is expected and normal.

---

## 4. Data Retention, Storage & Security

### 4.1 Local Processing Mandate
* GateID Vision processes videos strictly on the local host machine.
* Video data and captured snapshots are **never transmitted** to third-party cloud APIs, external analytics vendors, or remote storage buckets.

### 4.2 Snapshot Minimization & Lifecycle
* Snapshots are captured exclusively when a state transition occurs (e.g., entrance confirmed or flagged for review).
* The rate-limiting engine enforces a **$3.0\text{s}$ cooldown** and a **cap of 2 snapshots per individual**, preventing disk flooding.
* Security operators should establish a regular purging schedule:
  ```powershell
  # Operator command to purge old demonstration snapshots
  Remove-Item "outputs\snapshots\*\*.jpg" -Force
  ```

---

## 5. Security Personnel Operational Guidelines

Hostel and campus security staff using the GateID Vision dashboard should follow these operational instructions:

1. **Verify Physically with Courtesy:** When a case appears in the *Verification Queue*, security staff should politely ask the student to present their ID card physically.
2. **Do Not Treat the AI as Infallible:** Staff decisions always take precedence over system flags.
3. **Log Human Outcomes Independently:** Use the dashboard review buttons (*Confirm Physically*, *Dismiss*) to document physical verification outcomes separately from automated model classifications.
4. **Maintain Camera Hygiene:** Ensure gate cameras are positioned at eye level, clean lenses regularly, and avoid positioning cameras facing directly into morning sunrise or headlights.
