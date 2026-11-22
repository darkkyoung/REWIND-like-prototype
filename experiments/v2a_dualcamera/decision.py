"""Deployment tolerance and same-runtime exactness are independent observations."""
from .adapter import PINS

MODES = ("strict", "exploratory")


def decide(mode, clips):
    if mode not in MODES or set(clips) != set(PINS["clips"]):
        raise ValueError("mode or complete prespecified selection")
    exact = True
    published = True
    for name, row in clips.items():
        sequence, start = PINS["clips"][name]
        if row["sequence"] != sequence or row["frames"] != list(range(start,start+50)):
            raise ValueError("clip identity")
        exact &= all(row[k] is True for k in ("same_environment_xy_exact", "same_environment_local_hand_exact",
                                              "same_environment_whole62_exact"))
        status = row["published_parity"]["status"]
        if status not in ("within_tolerance", "outside_tolerance"):
            raise ValueError("unknown published reference result")
        published &= status == "within_tolerance"
    allow = exact and (published or mode == "exploratory")
    reason = ("same_runtime_mismatch" if not exact else "published_reference_mismatch"
              if not published and mode == "strict" else "paired_only_release_failed"
              if not published else "both_parity_checks_passed")
    return dict(mode=mode, historical_release="FAIL", same_runtime_exact=exact,
                strict_release="PASS" if exact and published else "FAIL",
                published_reference="PASS" if published else "FAIL", parity_allows_candidate=allow, reason=reason)
