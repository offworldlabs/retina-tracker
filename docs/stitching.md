# Stitching

A tracker that only models well-constrained physics breaks a track whenever an
aircraft does something it does not model: crossing the zero-Doppler blank
(docs/zero-doppler-blank.md), dropping below the detection floor for longer than
`n_delete` frames, or manoeuvring hard. Stitching is how those fragments are
shown as one aircraft again, without loosening the filter.

Each fragment keeps its own track id. An id shared between fragments hides
fragmentation, because live_score counts ids per aircraft, and breaks the event
writer, whose high-water mark is per id (PR #31). A stitch adds a link instead.

## The contract

Every event in `events.jsonl` and every track's `meta` in the history stream
(`/events`) carries `stitch_id`: the track id of the first fragment of the chain
this fragment belongs to. An unstitched fragment carries its own id. Consumers
group by `stitch_id` to show one aircraft per chain, and order the fragments of a
chain by time; there is a gap between consecutive fragments, which should be
drawn as a gap rather than joined.

A fragment's `stitch_id` can change once: from its own id to its chain's first
fragment, when the link is decided after the fragment was first reported. The
history's `meta` is replaced on every event, so the next event carries it.
`Track.to_dict` carries it too.

## ADS-B stitching

The first kind of stitch, and the one that joins zero-Doppler crossings for
cooperative aircraft: fragments carrying the same ICAO hex are one aircraft.
`Tracker._stitch` decides it once per fragment, the first time the fragment is
confirmed and carries its hex on at least `MIN_STITCH_LABELS` = 2 detections.
It links to that hex's latest fragment if:

- that fragment's last detection came before this fragment's first, so the two
  never overlap; while the earlier one is only coasting the decision waits, and
  if it detects again the newer one starts its own chain;
- and it is still within the history window (`history.WINDOW_S`, 4 h), so a
  chain never names a fragment the history no longer holds.

Two detections rather than one, because labels are attached upstream by a
delay-Doppler match against ADS-B, and a single stray label lands on the wrong
object. On fairforest B on 17 October one did: aircraft a03038 crossed the blank
at 6.4 km, and an object hovering at 7.6 km and -19 Hz picked up one a03038 label
1.2 km from it, which would have stitched it into the aircraft's chain.

Fragments of one hex alive at the same time are never linked. They are a
duplicate track of the aircraft, or one of them carries a wrong label, and
joining them would interleave two tracks in one chain.

### Measured

Replayed on six datasets, the tracker's 4 h history from jn1, fairforest B and
jn2 on 11 and 17 October with ADS-B labels from the same runs. Tracking is
unchanged by stitching (identical to the replay without it); only the links are
new. A crossing is counted when an aircraft's labelled detections change Doppler
sign within 5 minutes.

| Dataset | Aircraft | Fragments per aircraft | Chains per aircraft | Links | Labelled crossings | In one track | In one chain | Implausible links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jn1 17 Oct | 53 | 1.89 | 1.04 | 45 | 27 | 2 | 26 | 0 |
| ffb 17 Oct | 66 | 2.76 | 1.26 | 99 | 36 | 0 | 32 | 0 |
| jn2 17 Oct | 22 | 1.05 | 1.00 | 1 | 7 | 5 | 6 | 0 |
| jn1 11 Oct | 45 | 1.91 | 1.00 | 41 | 26 | 4 | 26 | 0 |
| ffb 11 Oct | 135 | 2.37 | 1.07 | 175 | 46 | 0 | 45 | 0 |
| jn2 11 Oct | 14 | 1.07 | 1.00 | 1 | 5 | 5 | 5 | 0 |

An implausible link is one whose two ends are further apart in delay than 0.6 km
per second of gap plus 0.5 km, more than any aircraft's bistatic range rate
allows. The median link spans 11-15 s, the crossing or dropout itself; the
longest, 89 min and 2.7 h, are aircraft that left coverage and came back.

Most of fairforest B's remaining extra chains (13 of 17 on 17 October, 6 of 10 on
11 October) are a second fragment of the same hex alive alongside another, which
the overlap rule leaves apart on purpose; most of the rest carry their hex on
fewer than two detections.

Before the two-label and overlap rules, the same replays linked one fragment of
fairforest B 1.29 km in 1 s (the stray label above), and on 11 October put a
5-detection duplicate born during a 1 s gap in its aircraft's track into the
same chain.

## Not yet done

- **retina-gui** does not read `stitch_id`. It draws one row, colour and line per
  track id, so a chain still shows as its fragments until it groups by
  `stitch_id` and breaks the line between them.
- **Physics stitching**, joining fragments with no ADS-B, such as an aircraft
  without a transponder crossing the blank or a hard manoeuvre. Evidence for
  joining two confirmed tracklets under constant jerk is in
  docs/zero-doppler-blank.md. To be designed.
- `Tracker._merge_tracks` (5 s, 5 km, 50 Hz) still merges dead tracks in the
  internal archive, which no consumer reads. It predates stitching.
