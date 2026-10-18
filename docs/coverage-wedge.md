# Coverage wedge

The coverage wedge is the region around the receiver where this node has been **proven** to
detect aircraft, built only from ADS-B-verified detections. Nothing in it comes from a declared
antenna aim or beam width. Alongside it, the tracker draws each unlabelled track's delay ellipse
as an arc that is solid where it crosses the wedge and faint elsewhere. retina-gui's Tracker page
shows both on a map (its `docs/features/tracker.md`, "Wedge view").

Code: `retina_tracker/coverage.py` (`Coverage`, `verified_position`, the geometry helpers),
served by `retina_tracker/control.py`, fed from `retina_tracker/server.py`.

## Geometry

The receiver and transmitter positions come from the node's blah2 config, the `location.rx` and
`location.tx` blocks that blah2's own ADS-B matcher uses. `load_blah2_config` reads them and the
CLI copies them into the tracker's `location` config section; a file-mode config can set
`location` directly. A node without both ends placed has no geometry: `/coverage` reports
`enabled: false` and no arcs are drawn.

All coverage geometry is spherical (Earth radius 6371 km): bearing, haversine ground range and
the forward geodesic. That matches retina-analytics' helpers rather than the tracker's WGS84
`geometry.py`; the two differ by about 0.2 degrees of bearing and 0.3% of range, far inside a
5 degree bin, and keeping it spherical keeps the module self-contained and vectorisable.

## What counts as a verified position

`verified_position(detection, track_hex)` returns a position only when all of these hold:

- The detection carries blah2's ADS-B match (`detection["adsb"]`), and its `hex` equals the
  **track's** `adsb_hex`. A track keeps its hex after it swaps onto another target, so the
  detection's own tag is what proves the position belongs to this detection.
- `position_age` is present and at most `coverage.max_position_age_s` (20 s). blah2 projects
  each aircraft to the detection's timestamp before matching and reports how far it projected
  (`blah2-arm/api/lib/extrapolation.js`, `projectForMatch`). A missing age means blah2 could not
  bind the fix to the detection, so it is refused rather than trusted.
- `lat`/`lon` are finite and in range.

The limit is blah2's own projection cap (`MATCH_MAX_POSITION_AGE_S`), not a tighter one, because
the age measures how far blah2 projected the fix, not how wrong the result is. It started at
retina-server's 10 s, which is the right bound for an unprojected fix (2.5 km stale at 250 m/s)
but refused 22% of correctly tagged detections in a test node's first 20 h: ADS-B arrives in
batches about 8.6 s apart, so ages run in a sawtooth (median 8 s, 95th percentile 13.5 s, 20 s at
most). Measured on the same node, as the jump in position when a fresh fix replaces a projected
one, the projection error was a median 0.07-0.14 km and a 90th percentile of about 0.2 km at
every age up to 20 s, against a 5 degree slice 1.7 km wide at 20 km. Ages over 12 s were rare (19
samples), so this is worth rechecking on other nodes.

The position is never the tracker's own estimate. The same rule decides the `lat`/`lon` that
track points carry on `/events` (see `history.py`), so a trail on the map is drawn from exactly
the evidence the wedge is built from.

These rules are borrowed from retina-server's calibration recorder, where each one was learned
from contaminated coverage polygons: positions from solves rather than ADS-B, points recorded
from coasting tracks, and a track's stale hex painting another aircraft's path.

## Passes and the max-hold

The unit of evidence is a **pass**: one track's verified detections in one bearing bin
(`coverage.bin_deg`, 5 degrees). A pass records its aircraft hex, detection count, first and
last time, and the furthest ground range it reached. Positions under 0.5 km from the receiver
are dropped as uninformative. Each event repeats a window of recent detections, so the recorder
keeps a per-track high-water mark and counts each detection once.

A pass counts once it has `coverage.min_pass_detections` (3) detections, so one blip cannot move
a boundary. For each bin, the held range is the **second-best** range among distinct aircraft:
each aircraft contributes its furthest qualifying pass, and the bin holds the second-highest of
those. A boundary therefore moves only once two different aircraft have reached it, and a single
wrong-aircraft match cannot push it out. Two passes of the same aircraft are still one aircraft.

The hold slides over `coverage.hold_s` (24 h). A pass leaves the hold when its last detection is
older than that, and the bin falls back to the best evidence still inside it. A full day weights
every hour equally; a shorter hold makes the wedge's size depend on the time of day, because
traffic and its routes follow a daily cycle. The held bins are recomputed at most every 10 s.

This deliberately does not mirror retina-server's evidence-only polygon (retina-analytics
`empirical_coverage.py`). Its thresholds count points, which at about one point a second measure
dwell time rather than independent evidence; its 200-point per-bin FIFO cannot express a time
window; and drawing vertices at bin centres makes every lobe 5 degrees narrower than its
evidence. Here each bin is drawn edge to edge.

The constants (`min_pass_detections`, `max_position_age_s`, the second-best rule) have not yet
been validated across sites. Check them against several nodes' recordings before relying on the
wedge for anything beyond display.

## The processed-range limit

blah2 only searches delays up to its `delayMax` (bins, converted with the sample rate to a
differential range; `config.DELAY_MAX_KM`). No detection can lie beyond the ellipse of that
differential, so where the wedge reaches it, its reach is set by the node's configuration and
says nothing about the antenna. On a test node with a short baseline, the main lobe's held
ranges sat at 87-98% of that ellipse.

`/coverage` therefore also serves `span_ring`, the ellipse at `max_differential_km`, and
`limited`, per bin: true when the held range is at least `SPAN_LIMITED_FRACTION` (0.85) of the
ellipse's range at the bin centre. The margin is there because traffic does not fly exactly to
the edge and flat ranges sit a little inside it for aircraft at altitude. Both are empty when
the node states no `delayMax`.

## Reset and persistence

`POST /coverage/reset` clears every pass and records when and why (`reset_reason: manual`). It
does not touch tracking or the history. The wedge also starts again, with reason `geometry`, when
it is loaded under a receiver or transmitter position other than the one it was built under, and
with reason `format` when the saved schema or bin width differs.

The state is saved to `coverage.json` beside the `-s` events file (override with
`--coverage-state`; memory only when events go to stdout), every 60 s and on reset, written to a
temporary file and renamed so a crash cannot leave half a file. A restart therefore loses at most
a minute of passes, unlike `events.jsonl`, which a restart truncates.

`since_ms` in the state is the start of the evidence the wedge can hold: the later of 24 h ago
and the last reset (or the moment a fresh state began).

## Arcs

`Coverage.arc(differential_km)` draws the locus of the measured bistatic range: the ellipse with
foci at receiver and transmitter, whose ground range from the receiver at angle psi off the
receiver-to-transmitter bearing is

```
r(psi) = D (D + 2L) / (2 ((L + D) - L cos psi))
```

for differential range D (the detection's delay, in km) and baseline L. It is sampled every half
bin, so every sample falls on a bin edge or centre, and each segment is solid when it lies within
its bin's held range and faint otherwise. A segment that crosses the held range is split at the
crossing. An empty wedge, as after a reset, makes every arc faint.

Arcs are **flat**, as retina-server's are since 2026-08: target altitude never enters them, so an
aircraft at altitude sits a few kilometres off its own arc. That is the honest raw-measurement
picture. Differentials under 3 km are not drawn: near the baseline the ellipse degenerates into a
sliver that reads as a blob.

Arcs are cached by differential (quantised to 30 m) and the held wedge, up to 4096 entries. A
replay asks for at most 240 arcs spread evenly over a track (`replay_arcs`); at 5 decimal places
an arc is about 3 KB, so a replay is up to roughly 750 KB.

## Endpoints

| Route | Returns |
| --- | --- |
| `GET /coverage` | `enabled`, `now_ms` (the node's clock, so a consumer can tell a full hold from a recent start without trusting its own), `geometry` (rx, tx, `baseline_km`), `bin_deg`, `hold_s`, `min_pass_detections`, `since_ms`, `reset_ms`, `reset_reason`, `held_km` (one per bin, 0 where unproven), `proven_bins`, `aircraft`, `lobes` (rings of `[lat, lon]`, each starting and ending at the receiver), `max_differential_km`, `span_ring` (closed ring of `[lat, lon]`, empty when unknown), `limited` (one bool per bin) and `limited_bins`. |
| `GET /coverage/arcs?window=<s>` | `{"arcs": [{"id", "t", "delay", "arc"}]}`: the newest point's arc for each **unlabelled** track in the window. Labelled tracks are drawn from their positions instead. |
| `GET /coverage/track/<id>` | `{"id", "adsb_hex", "arcs": [{"t", "delay", "arc"}]}` for the replay. 400 for an id outside `[A-Za-z0-9-]`, 404 when the track is not held. |
| `POST /coverage/reset` | `{"ok": true}` |

Each `arc` is `{"solid": [[lat, lon], ...] lines, "faint": [...]}` or null when not drawable. All
coverage routes answer 503 when the tracker runs without coverage.
