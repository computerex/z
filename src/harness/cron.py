"""Cron expression parser — 5-field cron (minute hour dom month dow), next-fire calculation."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Sequence

# ── Public types ─────────────────────────────────────────────────────────────


@dataclass
class CronFields:
    """Expanded cron fields as sorted lists of matching values."""
    minute: List[int] = field(default_factory=list)
    hour: List[int] = field(default_factory=list)
    day_of_month: List[int] = field(default_factory=list)
    month: List[int] = field(default_factory=list)
    day_of_week: List[int] = field(default_factory=list)


# ── Field expansion ──────────────────────────────────────────────────────────

_FIELD_RANGES: List[tuple[int, int]] = [
    (0, 59),   # minute
    (0, 23),   # hour
    (1, 31),   # day of month
    (1, 12),   # month
    (0, 6),    # day of week (0=Sunday; 7 accepted as alias)
]


def _expand_field(field: str, rmin: int, rmax: int) -> Optional[List[int]]:
    """Parse a single cron field into a sorted list of matching values.

    Supports: ``*``, ``*/N``, ``N``, ``N-M``, ``N-M/S``, and ``A,B,C``.
    Returns ``None`` on invalid input.
    """
    out: set[int] = set()
    is_dow = rmin == 0 and rmax == 6  # day-of-week accepts 7 → 0

    for part in field.split(","):
        part = part.strip()
        if not part:
            return None

        # wildcard or star-slash-N
        if part.startswith("*"):
            step_str = part[1:]  # "" or "/N"
            step = 1
            if step_str:
                if not step_str.startswith("/"):
                    return None
                step = int(step_str[1:])
                if step < 1:
                    return None
            for i in range(rmin, rmax + 1, step):
                out.add(i)
            continue

        # N-M or N-M/S
        if "-" in part:
            parts = part.split("/", 1)
            range_part = parts[0]
            step = int(parts[1]) if len(parts) > 1 else 1
            if step < 1:
                return None
            segs = range_part.split("-", 1)
            if len(segs) != 2:
                return None
            lo, hi = int(segs[0]), int(segs[1])
            eff_max = 7 if is_dow else rmax
            if lo > hi or lo < rmin or hi > eff_max:
                return None
            for i in range(lo, hi + 1, step):
                out.add(0 if is_dow and i == 7 else i)
            continue

        # plain N
        n = int(part)
        if is_dow and n == 7:
            n = 0
        if n < rmin or n > rmax:
            return None
        out.add(n)

    if not out:
        return None
    return sorted(out)


# ── Expression parsing ───────────────────────────────────────────────────────


def parse_cron_expression(expr: str) -> Optional[CronFields]:
    """Parse a 5-field cron expression into expanded number arrays.

    Returns ``None`` if invalid or unsupported syntax.
    """
    parts = expr.strip().split()
    if len(parts) != 5:
        return None

    expanded: List[List[int]] = []
    for i, part in enumerate(parts):
        rmin, rmax = _FIELD_RANGES[i]
        result = _expand_field(part, rmin, rmax)
        if result is None:
            return None
        expanded.append(result)

    return CronFields(
        minute=expanded[0],
        hour=expanded[1],
        day_of_month=expanded[2],
        month=expanded[3],
        day_of_week=expanded[4],
    )


# ── Next-run computation ─────────────────────────────────────────────────────


def compute_next_cron_run(fields: CronFields, from_dt: Optional[datetime] = None) -> Optional[datetime]:
    """Compute the next datetime strictly after *from_dt* that matches
    *fields*, using the process's local timezone.

    Walks forward minute-by-minute. Bounded at 366 days; returns ``None``
    if no match found (impossible for valid cron, but safe).

    Standard cron semantics: when both day-of-month and day-of-week are
    constrained (neither is the full range), a date matches if **either**
    matches (OR semantics).

    DST: fixed-hour crons targeting a spring-forward gap (e.g. ``30 2 * * *``
    in a US timezone) skip the transition day — the gap hour never appears
    in local time, so the hour-set check fails and the loop moves on.
    Wildcard-hour crons (``30 * * * *``) fire at the first valid minute
    after the gap. Fall-back repeats fire once (step-forward jumps past
    the second occurrence).  This matches vixie-cron behaviour.
    """
    if from_dt is None:
        from_dt = datetime.now()

    minute_set = set(fields.minute)
    hour_set = set(fields.hour)
    dom_set = set(fields.day_of_month)
    month_set = set(fields.month)
    dow_set = set(fields.day_of_week)

    dom_wild = len(fields.day_of_month) == 31
    dow_wild = len(fields.day_of_week) == 7

    # Round up to the next whole minute (strictly after *from_dt*)
    t = from_dt.replace(second=0, microsecond=0) + timedelta(minutes=1)

    max_iter = 366 * 24 * 60
    for _ in range(max_iter):
        # Month check — jump to next month on failure
        if t.month not in month_set:
            # Jump to the 1st of next month
            next_month = t.month + 1
            year = t.year
            if next_month > 12:
                next_month = 1
                year += 1
            t = t.replace(year=year, month=next_month, day=1, hour=0, minute=0)
            continue

        # Day check — OR semantics when both are constrained
        dom_matches = t.day in dom_set
        dow_matches = t.weekday() in dow_set  # weekday(): Mon=0, Sun=6

        if dom_wild and dow_wild:
            day_ok = True
        elif dom_wild:
            day_ok = dow_matches
        elif dow_wild:
            day_ok = dom_matches
        else:
            day_ok = dom_matches or dow_matches

        if not day_ok:
            t = (t + timedelta(days=1)).replace(hour=0, minute=0)
            continue

        # Hour check
        if t.hour not in hour_set:
            t = t.replace(minute=0) + timedelta(hours=1)
            continue

        # Minute check
        if t.minute not in minute_set:
            t += timedelta(minutes=1)
            continue

        return t

    return None


def next_cron_run_ms(cron: str, from_ms: float) -> Optional[float]:
    """Convenience wrapper: next fire time in epoch ms given a cron string.

    Returns ``None`` if invalid or no match in the next 366 days.
    """
    fields = parse_cron_expression(cron)
    if fields is None:
        return None
    from_dt = datetime.fromtimestamp(from_ms / 1000)
    result = compute_next_cron_run(fields, from_dt)
    return result.timestamp() * 1000 if result else None


# ── Human-readable display ───────────────────────────────────────────────────


_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def cron_to_human(cron: str) -> str:
    """Convert a cron expression to a human-readable description.

    Falls through to the raw cron string for complex patterns.
    """
    parts = cron.strip().split()
    if len(parts) != 5:
        return cron

    minute_str, hour_str, dom_str, month_str, dow_str = parts

    # Every N minutes: */N * * * *
    if minute_str.startswith("*/") and hour_str == "*" and dom_str == "*" and month_str == "*" and dow_str == "*":
        n = int(minute_str[2:])
        return "Every minute" if n == 1 else f"Every {n} minutes"

    # Every hour: 0 * * * *  (or at :MM past the hour)
    if minute_str.isdigit() and hour_str == "*" and dom_str == "*" and month_str == "*" and dow_str == "*":
        m = int(minute_str)
        if m == 0:
            return "Every hour"
        return f"Every hour at :{m:02d}"

    # Every N hours: 0 */N * * *
    if minute_str.isdigit() and hour_str.startswith("*/") and dom_str == "*" and month_str == "*" and dow_str == "*":
        n = int(hour_str[2:])
        m = int(minute_str)
        suffix = "" if m == 0 else f" at :{m:02d}"
        return f"Every {n} hours{suffix}" if n > 1 else f"Every hour{suffix}"

    # Remaining cases reference specific hour+minute
    if not minute_str.isdigit() or not hour_str.isdigit():
        return cron
    m, h = int(minute_str), int(hour_str)

    def _fmt_time() -> str:
        return f"{h % 12 if h % 12 != 0 else 12}:{m:02d} {'AM' if h < 12 else 'PM'}"

    # Daily at specific time
    if dom_str == "*" and month_str == "*" and dow_str == "*":
        return f"Every day at {_fmt_time()}"

    # Specific day of week
    if dom_str == "*" and month_str == "*" and dow_str.isdigit():
        idx = (int(dow_str) % 7)
        day_name = _DAY_NAMES[idx - 1] if idx > 0 else _DAY_NAMES[6]
        return f"Every {day_name} at {_fmt_time()}"

    # Weekdays: 0 9 * * 1-5
    if dom_str == "*" and month_str == "*" and dow_str == "1-5":
        return f"Weekdays at {_fmt_time()}"

    return cron
