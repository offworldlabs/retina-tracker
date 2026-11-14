# Outside store

Every confirmed track that took a detection beyond ±300 Hz of Doppler is kept on the node's disk,
with every detection around it while it lived, so it can be looked at long after the 4 h history
has dropped it. Code: `retina_tracker/outside.py`. retina-gui will read it through the routes
below; it holds no copy of its own.

## What is stored, and when

**Trigger.** A track qualifies when it was confirmed (it has an id) and at least one detection it
took had |Doppler| > `OUTSIDE_DOPPLER_HZ` (300 Hz, `track.py`). The test is on the measured
Doppler of each detection, never the Kalman state, and counts every detection of the track's life
including those before it was confirmed. `Track.n_outside` and `Track.peak_doppler_hz` keep the
count and the largest |Doppler| as detections arrive.

**Length limit.** A track that lived longer than `MAX_TRACK_MS` (5 minutes, first to last
detection) is not stored, only counted (`too_long`). The check runs in `retire`, before anything is
queued, so a long track never costs a history read. It bounds the neighbour window: building a
record copies the window while holding the history lock the frame thread also needs, and a track
lasting hours (a CW tone beyond 300 Hz builds exactly that) would hold it for seconds on a node.
5 minutes keeps every track that crossed 300 Hz in the October collection, the longest of which
lasted 218 s (384 on jn2, one on jn1, none on ffb), and matches the tracker's own per-track buffer
of 600 frames. The longest confirmed track of any kind in that data lasted 16 minutes, on ffb.

Nothing else is filtered on the node. Tracks built on interference, shadow tracks and ADS-B
aircraft are all kept, with their fractions and hex, so a reader can filter them. Filtering here
would lose the evidence for good.

The limit is a literal 300 Hz on every node, matching retina-gui's narrow Doppler view. It is
not a speed. Doppler is bistatic range rate over wavelength, so 300 Hz is 422 m/s of range rate at
213 MHz but 165 m/s at 545 MHz, where ordinary airliners cross it. Measured on the October
collection, which ran at ±1000 Hz:

| Node | Frequency | Days | Tracks beyond ±300 Hz | ADS-B labelled |
| --- | --- | --- | --- | --- |
| jn1 | 213 MHz | 4.2 | 1 | 0 |
| ffb | 213 MHz | 1.3 | 0 | 0 |
| jn2 | 545 MHz | 1.8 | 83 | 52 |

A speed-based limit was considered and set aside for now: one node only proves a target's speed
is at least |range rate| / 2, so a boundary at airliner top speed (about 340 m/s) needs a range
rate of 680 m/s. That is 1,236 Hz at 545 MHz, outside the ±1000 Hz span every node processes.

**When.** At retirement: the tracker's deletion step, 5 s after the track's last detection, calls
`retire_sink.retire(track, run_id)`. That runs on the frame thread, so it only copies a summary of
the track and queues it. Nothing touches disk or the history there.

The writer thread (`start_writer`, once a second) then waits for the neighbour window to settle
before building the record:

- Detections reach the history only once their classification is final, which is usually at
  once but can take up to 60 frames (`MAX_PENDING_CLASSIFICATION_FRAMES`) when a tentative
  track nearby is unresolved.
- The window runs `MARGIN_MS` (10 s) past the last detection, and those seconds have to happen.

A record is written once `DetectionHistory.released_ms()` reaches the last detection plus the
margin, or `MAX_WAIT_S` (60 s) after it was queued, whichever comes first. The second case is a
feed that stopped: the record is written with what the history holds.

## What a record holds

| Field | Meaning |
| --- | --- |
| `run`, `track_id` | The key. Track ids repeat after a restart; the run id does not |
| `stitch_id`, `adsb_hex` | As in events.jsonl (docs/stitching.md) |
| `limit_hz`, `n_outside`, `peak_doppler_hz` | Why it was stored: the limit then, how many detections were beyond it, and the signed Doppler of the largest |
| `birth_ms`, `last_ms` | First and last detection |
| `n_detections`, `n_frames`, `mean_snr`, `max_velocity_ms`, `shadow_fraction`, `interference_fraction` | What the tracker thought of it |
| `n_points`, `complete` | Points the history held for it, and whether that is every detection it took. False when the history was cleared or had evicted it |
| `track` | Its points as columns `t`, `delay` (km), `doppler` (Hz), `snr` (dB), `lat`, `lon` (verified ADS-B position or null), plus `stride` |
| `neighbours` | `start_ms`, `end_ms` (the track's span plus 10 s each side), `total`, `stride`, `detections` (every detection on the node in the window, by class, as columns, less the track's own) and `tracks` (every other track's points in the window, with its meta) |
| `radar`, `location` | The radar settings and RX/TX positions the tracker ran with, so a record still makes sense after a re-tower |
| `schema`, `stored_ms`, `id` | Format version, wall-clock write time, store id |

Columns are the history's wire form, rounded the same way, so whatever draws `/events` can draw a
record.

**Neighbours are the whole map.** Every detection the node classified in the window is kept,
across its whole delay and Doppler span, with every other track's points in the window. What is
going on around a track, a mirror image, a second aircraft, a burst of interference, is often
nowhere near it in delay-Doppler. The track's own detections are left out of `detections`, since
`track` already holds them; nothing else is filtered. A track the history no longer holds still
gets its window, from the birth and last times the tracker reported.

Measured over 4 h history snapshots with a ±10 s window, a track has 290-440 neighbours at the
median and about 1,000 at p90 (jn1, ffb, jn2), 10-35 kB of JSON before compression.

A tube was tried on 2026-11-14: keeping only detections within 5 km and 100 Hz of where the track
was at the time, projected along its line in the margins. It kept 3-8% of the window and cut
records to about half a kilobyte, and was set aside because the whole map is what a reader wants.

**Per-record cap.** `track` and the neighbours each stay under `MAX_RECORD_POINTS` (50,000) by
keeping every `stride`-th point, `stride` recorded. Only a track lasting hours reaches it. Thinning
by stride keeps the whole time span drawable, where truncation would lose its end.

## Storage

SQLite (stdlib `sqlite3`), one file, `outside_tracks.sqlite` beside the `-s` events file. On a
node that is `/app/output`, the host mount `/data/retina-node/retina-tracker/output`, which
survives container restarts. Chosen over append-only files because it gives atomic commits that
survive power loss, deletion of the oldest rows without rewriting files, and indexed paging for
the reader, with no new dependency.

- One table, `records`: the summary fields as columns, so the list never decompresses anything,
  and the record as zlib-compressed JSON in `payload`. `UNIQUE (run, track_id)`.
- `id` is `AUTOINCREMENT`, so an id is never reused after the oldest rows are deleted, and a page
  cursor (`before`) stays valid.
- `journal_mode=WAL` with `synchronous=FULL`: every record is fsynced when committed. At tens of
  records a day that costs nothing, and SD card wear is negligible.
- `journal_size_limit` is 4 MiB, so the WAL file is truncated back after each checkpoint.
- `PRAGMA user_version` holds the schema version (1).

**Across releases.** Nodes only ever move forward a release at a time, and the store must outlive
every one of them. The rules for any change to it:

- Additive changes keep the version: a new column must have a default, and a new record field must
  be optional to readers. Every statement here names its columns and readers ignore fields they do
  not know, so old rows and new code, or new rows and old code, both work.
- A change that cannot be additive bumps `SCHEMA` and migrates the existing file in place, inside
  one transaction, when the new release first opens it. It never sets the file aside or starts a
  fresh one: setting aside is for corruption only.
- Each record carries `schema` and the parameters it was built with (`limit_hz`, the window,
  `radar`), so changing a parameter later is not a schema change: old records still say
  what they were built under.
- `tests/fixtures/outside_store_schema1.sql` is a store written by this first release (two
  records, one with neighbours and one without). A test opens it with the current code, lists
  and reads its records and writes a new one. A later release that cannot do all of that with a
  file from this one fails the suite. Never edit the fixture; a schema bump adds a new one beside it.

A file whose schema is newer than the running code is left untouched and the store stands down
(`disabled`, with the reason). Releases only move forward, so this is never expected; it exists so
that nothing ever moves a healthy file.

The file lives beside the `-s` events file, so it stays put only while that path does. A compose
change that moves the events file without `--outside-store` would start a new, empty store in the
new place.

**Capacity and roll-over.** The cap is on the database file, measured as its pages in use:
`(page_count − freelist_count) × page_size`, which the connection reports exactly inside a
transaction, deletions included. An earlier version capped the sum of compressed payloads, which
undercounted the file by about 30%: small rows leave part of each 4 KiB page unused, and a store
holding 255 MB of payload measured 336 MiB on disk. Two limits, and the lower one binds:

- `max_bytes` (256 MiB, `--outside-max-mb`) bounds the pages in use.
- `min_free_bytes` (1 GiB, `MIN_FREE_BYTES`) is free space the store never writes into. Before
  each insert it reads the filesystem's free space (`os.statvfs`); the file may have in use only
  its current size plus whatever free space sits above the floor.

Each write, in one transaction:

1. Deletes the oldest records until the pages in use plus the new record and one page of slack fit
   the cap. Deleting first puts their pages on SQLite's free list, so the insert reuses them instead
   of growing the file. Without it, a disk at the floor let the file creep up a page per write.
2. Inserts the record.
3. Deletes more of the oldest, if needed, until the pages in use fit exactly. The record just
   written is never deleted: if nothing older is left and it still does not fit, the whole
   transaction rolls back and the record is skipped and counted (`skipped`).

A record that fits in pages the file already has is written even on a disk at the floor, because it
does not grow the file. If free space cannot be read, only `max_bytes` applies.

The file does not shrink after deletes: freed pages are reused rather than returned to the disk.
At the floor it therefore stays the size it was, give or take a page. The WAL beside it adds up to
4 MiB (`journal_size_limit`) and is not counted in the cap. Lowering `--outside-max-mb` later trims
the records to fit but not the file, which would need a VACUUM.

Sizing, from records built out of 4 h history snapshots (track, neighbours and other tracks,
compressed about 3.7 times): a median of 3.5-3.9 kB, p90 8-11 kB, the largest 49 kB. A real jn2
track with 521 neighbours stored at 5.0 kB. The 5-minute limit bounds the largest possible record.

| | Records a day | Per day | 256 MiB lasts |
| --- | --- | --- | --- |
| jn2 (545 MHz), October rate | about 46 | about 230 kB | about 3 years |
| jn1, ffb (213 MHz) | about one every few days | a few kB | indefinitely |
| An interferer beyond 300 Hz, at ffb's tone rate | about 4,000 | about 20 MB | about 13 days |

The cards measured on 2026-11-14 (owl-ded9 and jonathan-node-1) are 64 GB, with a 24 GB `/data`
partition holding 15-18 GB free, so 256 MiB is under 2% of it. Other nodes were not checked; the
floor is what keeps the store from pushing any card towards full, whatever else is filling it.

## Safety

The frame thread must never wait on the SD card or fail because of the store, and a damaged file
must never stop the tracker starting.

| Failure | What happens |
| --- | --- |
| Disk at the floor | The oldest records are deleted to make room, so the file does not grow. With nothing left to delete the record is skipped and counted (`skipped`) |
| Writer slower than retirements | The queue holds `MAX_PENDING` (1,000) tracks. Past that, `retire` drops and counts (`dropped`) rather than growing |
| Summary or write raises | Counted (`failed`), logged, the next record goes ahead. The writer thread catches everything so it never dies |
| Power loss mid-write | The transaction is lost whole. SQLite's WAL replays committed records at the next open |
| File corrupt when opened (`SQLITE_CORRUPT`, `SQLITE_NOTADB`, `quick_check` not ok) | Moved to `outside_tracks.sqlite.bad` (one slot, replacing any older one) and a fresh store started. Never a crash loop |
| File from a newer schema | Left untouched; the store is disabled with the reason |
| File cannot be opened for any other reason (missing directory, permissions, locked) | The store is disabled, not set aside: such failures say nothing about the data. Retirements are counted as dropped, `/outside` answers 503 with the error, tracking carries on |
| Corruption found while running | That write fails and is counted. The next restart's `quick_check` sets the file aside |

**Opening off the ingest path.** Constructing the store touches no disk. The writer thread opens
the file on its first pass, a second after startup, and that is where `quick_check` runs: it reads
the whole file, which on a full store on an SD card can take seconds, and the tracker has already
started taking detections by then. Until it is open, `retire` keeps queuing as normal (the queue is
bounded) and `/outside` answers 503 "outside store opening". If opening fails, whatever was queued
is counted as dropped.

`/health` carries `outside`: `state` (`opening`, `open` or `disabled`), `error`, `pending`,
`queued`, `written`, `dropped`, `failed`, `rolled`, `skipped`, `too_long`, `records`,
`payload_bytes` (compressed record payloads), `used_bytes` (pages in use, what the cap applies to),
`file_bytes` (the whole database file, free pages included), `max_bytes`, `min_free_bytes`, and
`free_bytes` (the free space seen at the last write, null before one).

## Reading

| Route | Answer |
| --- | --- |
| `GET /outside?before=<id>&limit=<n>` | `{"schema": 1, "limit_hz": 300, "records": [...]}`: the summary columns, newest first. `limit` defaults to 50, at most 500. `before` pages back |
| `GET /outside/<id>` | One whole record, decompressed |

400 for a malformed id, `before` or `limit`; 404 for an id not held; 503 when the store is off,
still opening or disabled; 500 if a stored record cannot be decompressed. Loopback only, like every control
route.

## Limits

- A qualifying track still alive, or waiting to settle, when the tracker stops is lost. That is at
  most about 70 s of retirements (5 s to delete, up to 60 s to settle).
- `POST /reset` discards live tracks without retiring them, so none of them are stored.
- A track longer than 5 minutes is counted (`too_long`), not stored, however far beyond 300 Hz it
  went.
- The radar context is what the tracker read at startup. Auto-calibrate retunes the radio live and
  only resets the tracker, so records made during a search carry the old `center_frequency`.
- The limit is in Hz, so how often it fires depends on the band (see the table above).
- No delete or annotate. Records leave only by rolling off.
- retina-gui does not read the store yet.
